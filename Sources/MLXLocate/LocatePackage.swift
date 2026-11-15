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
/// Declarations round up: resident 5.4 / 3.35 GB, activation 2.6 GB for both (activations do not
/// quantize). The side cap (`measuredMaxSide`) bounds the visual tokens, so no request can exceed it.
public enum LocateFootprint {
    /// The longest picture side the activation was measured at. `LocateConfiguration` clamps to it.
    public static let measuredMaxSide = 1024

    public static func residentBytes(for quant: Quant) -> UInt64 {
        switch quant {
        case .int4: 3_350_000_000
        default: 5_400_000_000
        }
    }

    public static let peakActivationBytes: UInt64 = 2_600_000_000
}

/// The `locate` mode: the prompt is a phrase, the answer is the boxes as JSON (the default).
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
/// The engine contract has no box-typed capability yet, so this serves `imageAnalysis`:
/// - mode `locate` (or no mode): `prompt` is the phrase; the response text is `LocateAnswer` JSON.
///   `metaData["count"]` (int) caps the number of boxes (default 16). An empty list is a success.
/// - mode `answer`: the prompt is a question; the response is the model's text.
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
                    modes: [.locate, .answer])
            ])
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
