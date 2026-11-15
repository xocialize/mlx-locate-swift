import Foundation
import MLXToolKit

/// Init-time configuration for `LocatePackage` (C9). Quant-keyed: the quant picks the mlx-community
/// checkpoint of Qwen3-VL-4B-Instruct.
public struct LocateConfiguration: PackageConfiguration, ModelStorable, WeightSourcing, QuantConfigured {
    /// The true upstream (provenance); the bytes come from `repo(for:)`.
    public static let upstream = "Qwen/Qwen3-VL-4B-Instruct"
    public static let upstreamRevision = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

    /// The lanes this package declares. `int8` is the default (12/12 head crops in tolerance on the
    /// yardstick); `int4` is the small lane (3.1 GB resident, 11/12).
    public static let servableQuants: [Quant] = [.int8, .int4]

    public static func repo(for quant: Quant) -> String? {
        switch quant {
        case .int8: "mlx-community/Qwen3-VL-4B-Instruct-8bit"
        case .int4: "mlx-community/Qwen3-VL-4B-Instruct-4bit"
        default: nil
        }
    }

    /// Pinned mlx-community commits (the files measured in AB-A-0293's receipts).
    public static func revision(for quant: Quant) -> String? {
        switch quant {
        case .int8: "0943db6e15185b86be368d3cf0704aec740b142b"
        case .int4: "2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b"
        default: nil
        }
    }

    public var quant: Quant
    /// The picture's longest side is scaled down to this before the model sees it. 1024 is the measured
    /// envelope (the declared footprint covers it); larger values are clamped to it.
    public var maxSide: Int
    /// An explicit checkpoint folder (gates, offline use). Wins over the store.
    public var modelDirectory: URL?
    /// Set by the engine from its `ModelStore`. Not persisted.
    public var modelsRootDirectory: URL?

    public init(quant: Quant = .int8, maxSide: Int = 1024, modelDirectory: URL? = nil,
                modelsRootDirectory: URL? = nil) {
        self.quant = quant
        self.maxSide = maxSide
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    /// The side the package actually uses: never above the measured 1024.
    public var effectiveMaxSide: Int { min(max(maxSide, 64), LocateFootprint.measuredMaxSide) }

    private enum CodingKeys: String, CodingKey { case quant, maxSide, modelDirectory }

    // Hand-written so a field added later cannot break a persisted registration.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        quant = try c.decodeIfPresent(Quant.self, forKey: .quant) ?? .int8
        maxSide = try c.decodeIfPresent(Int.self, forKey: .maxSide) ?? 1024
        modelDirectory = try c.decodeIfPresent(URL.self, forKey: .modelDirectory)
        modelsRootDirectory = nil
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(quant, forKey: .quant)
        try c.encode(maxSide, forKey: .maxSide)
        try c.encodeIfPresent(modelDirectory, forKey: .modelDirectory)
    }

    // MARK: WeightSourcing

    /// One repo holds everything load() opens. The globs skip the README and the duplicate
    /// `chat_template.json` (the `.jinja` one is what the tokenizer reads).
    public var weightSources: [WeightSource] {
        guard let repo = Self.repo(for: quant) else { return [] }
        return [
            WeightSource(
                role: "main", repo: repo, revision: Self.revision(for: quant),
                matching: [
                    "model.safetensors", "model.safetensors.index.json", "config.json",
                    "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json",
                    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                    "added_tokens.json", "special_tokens_map.json", "chat_template.jinja",
                ])
        ]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let dir = modelDirectory, WeightSourceProbe.flatDirectory(dir, satisfies: weightSources.first?.matching) {
            return []
        }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }

    /// The folder load() reads: the explicit directory, else the store's snapshot or flat layout.
    public func resolvedDirectory() -> URL? {
        if let dir = modelDirectory { return dir }
        guard let source = weightSources.first else { return nil }
        let store = ModelStore(root: modelsRootDirectory)
        if let snap = store.snapshotDirectory(for: source.repo, revision: source.revision),
           WeightSourceProbe.snapshot(snap, satisfies: source.matching) { return snap }
        if let flat = store.directory(for: source.repo),
           WeightSourceProbe.flatDirectory(flat, satisfies: source.matching) { return flat }
        return nil
    }
}
