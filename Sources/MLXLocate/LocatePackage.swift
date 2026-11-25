import CoreGraphics
import Foundation
import ImageIO
import Locate
import MLX
import MLXNN
import MLXToolKit

/// Measured footprints: M5 Max 128 GB, macOS 27.3, Release, mlx-swift 0.32.4 (fleet fork),
/// mlx-swift-lm 3.32.3, `locate-cli mem-bench` on a 1022×1024 picture (1,081 prompt tokens, the
/// 1024 cap), one process per number, two processes per quant (2026-11-15, GPU shared with other
/// sessions). Floor = phys_footprint after load with the weights evaluated; peak = the ledger
/// phys_footprint peak over the query, MLX buffer cache uncapped (so an upper bound under the
/// engine's cache cap). Measured, in bytes:
///
///   quant  prompt          floor          peak           activation
///   int8   single-object   5,238,903,152  7,585,748,344  2,346,845,192
///   int4   single-object   3,195,554,672  5,604,052,760  2,408,498,088
///   int8   every-instance  5,238,378,864  7,591,466,288  2,353,087,424   (the shipped default)
///   int4   every-instance  3,197,569,976  5,614,374,776  2,416,804,800   (the shipped default)
///
/// Those are one-box answers. A 16-box answer (the `imageLocate` surface always asks the model for
/// up to 16, about 700 generated tokens) peaks higher. Measured 2026-11-25 through `MLXServeEngine`
/// (`locate-cli typed`, mlx-swift 0.33.1 fleet fork, mlx-swift-lm 3.33.0, a 1024×1024 picture of
/// 25 discs, "circle", count 16, two processes per quant, the larger peak; `receipts/typed/`):
///
///   quant  process peak   minus the floor above
///   int8   8,052,741,544  2,814,362,680
///   int4   6,122,426,352  2,926,871,680
///
/// Declarations round up: resident 5.4 / 3.35 GB, activation 3.0 GB for both (v0.2.0 declared 2.6,
/// which covered one box, not sixteen). The side cap (`measuredMaxSide`) bounds the visual tokens
/// and `Locator.maxTokenBudget` the generated ones, so no request can exceed it.
public enum LocateFootprint {
    /// The longest picture side the activation was measured at. `LocateConfiguration` clamps to it.
    public static let measuredMaxSide = 1024

    public static func residentBytes(for quant: Quant) -> UInt64 {
        switch quant {
        case .int4: 3_350_000_000
        default: 5_400_000_000
        }
    }

    public static let peakActivationBytes: UInt64 = 3_000_000_000
}

/// The `imageAnalysis` surface's `locate` mode: the prompt is a phrase, the answer is the boxes as
/// JSON (that surface's default). The typed way to ask is `ImageLocateRequest`.
extension Mode {
    public static let locate: Mode = "locate"
    /// Plain visual question answering with the same model.
    public static let answer: Mode = "answer"
}

public enum LocatePackageError: Error, CustomStringConvertible {
    case imageDecodeFailed(String)
    case weightsMissing

    public var description: String {
        switch self {
        case .imageDecodeFailed(let f): "locate: could not decode the \(f) picture"
        case .weightsMissing: "locate: no Qwen3-VL checkpoint found (materialize the weight source first)"
        }
    }
}

/// The JSON the `locate` mode returns in `ImageAnalysisResponse.text`: the `image.locate` output
/// shape, `{"boxes": [{x, y, width, height, label, score}]}`, fractions of the upright picture.
public struct LocateAnswer: Codable, Sendable, Equatable {
    public var boxes: [LocatedBox]
    public init(boxes: [LocatedBox]) { self.boxes = boxes }
}

/// An MLXEngine package that locates a phrase in a picture with Qwen3-VL-4B-Instruct (Apache-2.0).
///
/// Two surfaces on one loaded model:
/// - `imageLocate` (contract 1.66.0, AB-A-0295): `ImageLocateRequest(image:prompt:count:)` →
///   `ImageLocateResponse.boxes`, best first, at most `count` (nil → 1), every box scored. The
///   model is always asked for up to `LocatePackage.locateControls.maxCount` boxes and the answer
///   is `served(count:)`: clipped, in descending score, cut. An empty list is a success. The seed
///   is ignored (greedy decode).
/// - `imageAnalysis`: mode `answer` answers the prompt as text. Mode `locate` (or no mode) is the
///   pre-1.66.0 carrier, kept so v0.1.0 / v0.2.0 clients keep working: the response text is
///   `LocateAnswer` JSON in the model's own order, `metaData["count"]` caps it (default 16).
@InferenceActor
public final class LocatePackage: ModelPackage {
    public typealias Configuration = LocateConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Weights: Apache-2.0 (Qwen/Qwen3-VL-4B-Instruct LICENSE; the mlx-community conversions
            // carry the same tag). Training data undisclosed, so the declared licence governs
            // (AB-D-0105). Port code: Apache-2.0.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: LocateConfiguration.upstream,
                revision: LocateConfiguration.upstreamRevision, tier: 1),
            requirements: RequirementsManifest(
                footprints: LocateConfiguration.servableQuants.map {
                    QuantFootprint(quant: $0, residentBytes: LocateFootprint.residentBytes(for: $0),
                                   peakActivationBytes: LocateFootprint.peakActivationBytes)
                },
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil),
            specialties: [],
            surfaces: [
                ImageAnalysisContract.descriptor(
                    name: "qwen3-vl-locate",
                    summary: "Locate a phrase in a picture: mode `locate` (default) returns "
                        + "{\"boxes\": [{x, y, width, height, label, score}]} as JSON, fractions of the "
                        + "upright picture; mode `answer` answers the prompt as text.",
                    modes: [.locate, .answer]),
                ImageLocateContract.descriptor(
                    name: "qwen3-vl-locate-boxes",
                    summary: "Locate a phrase in a picture as typed boxes, best first: shares of the "
                        + "upright picture, each with the model's label and a score (the geometric mean "
                        + "probability of its coordinate tokens; for ranking, not calibrated). One box "
                        + "per instance the phrase names; an empty list means it is not there. Measured "
                        + "with English phrases; the picture's longest side is scaled to at most 1024 px.",
                    controls: locateControls),
            ])
    }

    /// What the `imageLocate` surface declares: up to 16 boxes a request (the generation budget is
    /// sized for 16, `Locator.tokenBudget`; measured on a 20-object picture), a score on every box,
    /// and no seed (greedy decode, so the same picture and phrase give the same boxes).
    public nonisolated static let locateControls = ImageLocateControls(maxCount: 16, scores: true, seeded: false)

    /// The core's boxes as the contract's, in the core's order.
    nonisolated static func contractBoxes(_ boxes: [LocatedBox]) -> [ImageLocateBox] {
        boxes.map { ImageLocateBox(x: $0.x, y: $0.y, width: $0.width, height: $0.height,
                                   label: $0.label, score: $0.score) }
    }

    private let configuration: Configuration
    private var locator: Locator?
    private var graph: MLXNN.Module?

    /// Test-facing seam for the engine's INF gate (C14). Empty before `load()`.
    var inferenceModeGraphs: [String: MLXNN.Module?] { ["qwen3vl": graph] }

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard locator == nil else { return }
        guard let dir = configuration.resolvedDirectory() else { throw LocatePackageError.weightsMissing }
        let l = try await Locator.load(directory: dir)
        graph = l.module
        locator = l
    }

    public func unload() async {
        locator = nil
        graph = nil
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the first act of run().
        try Task.checkCancellation()
        if let req = request as? ImageLocateRequest { return try await locate(req) }
        guard request.capability == .imageAnalysis, let req = request as? ImageAnalysisRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        guard let locator else { throw PackageError.notLoaded }
        let cg = try Self.decode(req.image)
        let mode = req.mode ?? .locate
        switch mode {
        case .locate:
            let count: Int = {
                if case .int(let n)? = req.metaData["count"] { return max(1, n) }
                if case .double(let n)? = req.metaData["count"] { return max(1, Int(n)) }
                return 16
            }()
            let result = try await locator.locate(
                image: cg, phrase: req.prompt, maxBoxes: count, maxSide: configuration.effectiveMaxSide,
                onToken: { try Task.checkCancellation() })
            let data = try JSONEncoder().encode(LocateAnswer(boxes: result.boxes))
            return ImageAnalysisResponse(text: String(decoding: data, as: UTF8.self))
        case .answer:
            let text = try await locator.answer(
                image: cg, question: req.prompt, maxSide: configuration.effectiveMaxSide,
                onToken: { try Task.checkCancellation() })
            return ImageAnalysisResponse(text: text)
        default:
            throw PackageError.unsupportedRequestFeature("mode \(mode.rawValue) (supported: locate, answer)")
        }
    }

    /// The typed surface. Refuses what the engine's pre-flight refuses (empty prompt, count below 1
    /// or above `maxCount`), so a directly driven package behaves the same.
    private func locate(_ req: ImageLocateRequest) async throws -> ImageLocateResponse {
        if let mode = req.mode, mode != .locate {
            throw PackageError.unsupportedRequestFeature(
                "mode \(mode.rawValue) (the imageLocate surface has no modes; `answer` is on imageAnalysis)")
        }
        if let reason = Self.locateControls.refusal(for: req) {
            throw PackageError.unsupportedRequestFeature(reason)
        }
        guard let locator else { throw PackageError.notLoaded }
        let cg = try Self.decode(req.image)
        // Ask for every box the surface may return, then keep the best `count`: cutting the model's
        // own list first would return the first boxes it wrote, not the ones it is surest of.
        let result = try await locator.locate(
            image: cg, phrase: req.prompt, maxBoxes: Self.locateControls.maxCount,
            maxSide: configuration.effectiveMaxSide, onToken: { try Task.checkCancellation() })
        return ImageLocateResponse(boxes: Self.contractBoxes(result.boxes).served(count: req.effectiveCount))
    }

    /// png / jpeg through ImageIO; raw BGRA8 and RGBA16 through CoreGraphics.
    static func decode(_ image: MLXToolKit.Image) throws -> CGImage {
        switch image.format {
        case .png, .jpeg:
            guard let src = CGImageSourceCreateWithData(image.data as CFData, nil),
                  let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
            else { throw LocatePackageError.imageDecodeFailed(image.format.rawValue) }
            return cg
        case .rawBGRA8, .rawRGBA16:
            guard let w = image.width, let h = image.height, w > 0, h > 0,
                  let provider = CGDataProvider(data: image.data as CFData),
                  let space = CGColorSpace(name: CGColorSpace.sRGB)
            else { throw LocatePackageError.imageDecodeFailed(image.format.rawValue) }
            let deep = image.format == .rawRGBA16
            let bpc = deep ? 16 : 8
            let stride = image.bytesPerRow ?? w * (deep ? 8 : 4)
            let info: CGBitmapInfo = deep
                ? CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
                    | CGBitmapInfo.byteOrder16Little.rawValue)
                : CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue)
            guard let cg = CGImage(width: w, height: h, bitsPerComponent: bpc, bitsPerPixel: bpc * 4,
                                   bytesPerRow: stride, space: space, bitmapInfo: info, provider: provider,
                                   decode: nil, shouldInterpolate: false, intent: .defaultIntent)
            else { throw LocatePackageError.imageDecodeFailed(image.format.rawValue) }
            return cg
        @unknown default:
            throw LocatePackageError.imageDecodeFailed(image.format.rawValue)
        }
    }
}

extension LocatePackage {
    public nonisolated static var registration: PackageRegistration { .of(LocatePackage.self) }
}
