import CoreGraphics
import CoreImage
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXNN
import MLXLMCommon
import MLXVLM
import Tokenizers

/// What one `locate` call returns: the boxes plus what the model actually wrote, for logs and gates.
public struct LocateResult: Sendable {
    public var boxes: [LocatedBox]
    public var text: String
    public var promptTokens: Int
    public var generatedTokens: Int
    /// The picture as the model saw it (after the side cap).
    public var seenWidth: Int
    public var seenHeight: Int
}

public enum LocateError: Error, CustomStringConvertible {
    case emptyPhrase
    case badImage

    public var description: String {
        switch self {
        case .emptyPhrase: "locate: the phrase is empty"
        case .badImage: "locate: the picture could not be read"
        }
    }
}

/// Text-to-boxes with a Qwen3-VL checkpoint: one picture and a phrase in, boxes out.
///
/// Greedy decoding, so the same picture and phrase give the same boxes. The picture's transparency is
/// composited over white and its longest side capped at `maxSide` (default 1024: the measured
/// envelope, ~1,100 prompt tokens); boxes are returned relative to the picture, so the cap does not
/// change their meaning.
public final class Locator: @unchecked Sendable {

    /// The default prompt: Qwen3-VL's multi-instance grounding prompt (one entry per instance). On the
    /// HairDeck yardstick (AB-A-0293) it beat the single-object prompt on hair, face and the head crop
    /// for both quants, and it splits plurals ("the hands" → two boxes). `{p}` is the phrase.
    public static let promptTemplate =
        #"Locate every instance that belongs to the following category: "{p}". Report bbox coordinates in JSON format: [{"bbox_2d": [x1, y1, x2, y2], "label": "{p}"}]."#

    /// The single-object prompt (kept for comparison; it tends to return one box for a plural phrase).
    public static let singleObjectTemplate =
        #"Locate {p} in the image and output its bounding box in JSON format: [{"bbox_2d": [x1, y1, x2, y2], "label": "{p}"}]."#

    public static let defaultMaxSide = 1024
    /// The floor of the generation budget. One box costs about 41 tokens with a four-word label
    /// (measured on the yardstick: each digit is a token), so this alone holds about six boxes.
    public static let defaultMaxTokens = 256
    /// The ceiling of the generation budget: bounds the run's time and its cache however long the
    /// phrase is. A list that would run past it is cut, and the parser drops the unfinished entry.
    public static let maxTokenBudget = 1024

    /// The generation budget for up to `boxes` boxes of `phrase`: the fence, then per box the
    /// `bbox_2d` entry (40 tokens budgeted; 16 boxes labelled "circle" took just under 640 in all)
    /// and the label, which repeats the phrase (at most one token per two UTF-8 bytes, a generous
    /// bound for Latin and CJK text). Never below `defaultMaxTokens`
    /// or above `maxTokenBudget`.
    public static func tokenBudget(boxes: Int, phrase: String) -> Int {
        let perBox = 40 + (phrase.utf8.count + 1) / 2
        return min(maxTokenBudget, max(defaultMaxTokens, 16 + max(1, boxes) * perBox))
    }

    private let container: ModelContainer
    /// The loaded module graph (for the engine's inference-mode gate). Read-only use.
    public let module: MLXNN.Module?

    private init(container: ModelContainer, module: MLXNN.Module?) {
        self.container = container
        self.module = module
    }

    private struct UncheckedModule: @unchecked Sendable { let module: MLXNN.Module? }

    /// Load a Qwen3-VL MLX checkpoint folder (config.json, safetensors, tokenizer, processor config).
    /// The weights are evaluated before this returns, so the first `locate` reads nothing from disk.
    public static func load(directory: URL) async throws -> Locator {
        let container = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(directory: directory))
        let graph = await container.perform { context in
            eval(context.model)
            return UncheckedModule(module: context.model as? MLXNN.Module)
        }
        return Locator(container: container, module: graph.module)
    }

    /// Find `phrase` in `image`. `maxTokens` nil sizes the generation for `maxBoxes` boxes
    /// (`tokenBudget`). `onToken` runs after every generated token (a cancellation point);
    /// throwing from it stops the run and rethrows.
    public func locate(
        image: CGImage, phrase: String, maxBoxes: Int = 16,
        maxSide: Int = Locator.defaultMaxSide, maxTokens: Int? = nil,
        template: String = Locator.promptTemplate,
        onToken: @escaping @Sendable () throws -> Void = {}
    ) async throws -> LocateResult {
        let phrase = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !phrase.isEmpty else { throw LocateError.emptyPhrase }
        guard let (picture, w, h) = Self.prepare(image, maxSide: maxSide) else { throw LocateError.badImage }
        let prompt = template.replacingOccurrences(of: "{p}", with: phrase)
        let cap = max(1, maxBoxes)

        // nil → room for `maxBoxes` boxes (v0.2.0's fixed 256 cut a list after about six).
        let budget = maxTokens ?? Self.tokenBudget(boxes: cap, phrase: phrase)
        let out = try await generate(picture: picture, prompt: prompt, maxTokens: budget,
                                     withOffsets: true, onToken: onToken)
        let boxes = GroundingParser.boxes(
            from: out.text, fallbackLabel: phrase, maxBoxes: cap, tokenEnds: out.ends, logprobs: out.logprobs)
        return LocateResult(boxes: boxes, text: out.text, promptTokens: out.promptTokens,
                            generatedTokens: out.logprobs.count, seenWidth: w, seenHeight: h)
    }

    /// Plain visual question answering with the same model (greedy).
    public func answer(
        image: CGImage, question: String, maxSide: Int = Locator.defaultMaxSide, maxTokens: Int = 512,
        onToken: @escaping @Sendable () throws -> Void = {}
    ) async throws -> String {
        guard let (picture, _, _) = Self.prepare(image, maxSide: maxSide) else { throw LocateError.badImage }
        return try await generate(picture: picture, prompt: question, maxTokens: maxTokens,
                                  withOffsets: false, onToken: onToken).text
    }

    private struct Generated: Sendable {
        var text: String
        var logprobs: [Float]
        var ends: [Int]
        var promptTokens: Int
    }

    private func generate(
        picture: CIImage, prompt: String, maxTokens: Int, withOffsets: Bool,
        onToken: @escaping @Sendable () throws -> Void
    ) async throws -> Generated {
        try await container.perform { context in
            var input = UserInput(chat: [.user(prompt, images: [.ciImage(picture)])])
            input.processing = .init()
            let lm = try await context.processor.prepare(input: input)
            let promptTokens = lm.text.tokens.size

            let sampler = RecordingGreedySampler()
            var iterator = try TokenIterator(
                input: lm, model: context.model, processor: nil, sampler: sampler, maxTokens: maxTokens)
            var eos = context.configuration.eosTokenIds
            if let id = context.tokenizer.eosTokenId { eos.insert(id) }
            for s in context.configuration.extraEOSTokens {
                if let id = context.tokenizer.convertTokenToId(s) { eos.insert(id) }
            }

            var tokens: [Int] = []
            while let t = iterator.next() {
                if eos.contains(t) { break }
                tokens.append(t)
                try onToken()
            }
            let logprobs = sampler.logprobs(count: tokens.count)

            // UTF-16 end offset of the decoded text after each token (prefix decodes; n is small).
            var ends: [Int] = []
            if withOffsets {
                ends.reserveCapacity(tokens.count)
                for i in tokens.indices {
                    ends.append((context.tokenizer.decode(tokenIds: Array(tokens[...i])) as NSString).length)
                }
            }
            return Generated(text: context.tokenizer.decode(tokenIds: tokens), logprobs: logprobs,
                             ends: ends, promptTokens: promptTokens)
        }
    }

    /// Transparency over white, longest side capped at `maxSide` (0 = no cap).
    static func prepare(_ cg: CGImage, maxSide: Int) -> (CIImage, Int, Int)? {
        var w = cg.width, h = cg.height
        guard w > 0, h > 0 else { return nil }
        if maxSide > 0, max(w, h) > maxSide {
            let k = Double(maxSide) / Double(max(w, h))
            w = max(1, Int((Double(w) * k).rounded()))
            h = max(1, Int((Double(h) * k).rounded()))
        }
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let out = ctx.makeImage() else { return nil }
        return (CIImage(cgImage: out), w, h)
    }
}

/// Greedy (argmax) sampling that keeps the chosen token's log-probability.
final class RecordingGreedySampler: LogitSampler, @unchecked Sendable {
    private var picked: [MLXArray] = []

    func sample(logits: MLXArray) -> MLXArray {
        let token = argMax(logits, axis: -1)
        let logp = logits.asType(.float32) - logSumExp(logits.asType(.float32), axis: -1, keepDims: true)
        let chosen = takeAlong(logp, token.expandedDimensions(axis: -1), axis: -1)
        asyncEval(chosen)
        picked.append(chosen)
        return token
    }

    /// The first `count` recorded log-probabilities (the iterator may sample one token past the end).
    func logprobs(count: Int) -> [Float] {
        let n = min(count, picked.count)
        guard n > 0 else { return [] }
        let all = concatenated(picked[..<n].map { $0.reshaped(-1) })
        return all.asArray(Float.self)
    }
}
