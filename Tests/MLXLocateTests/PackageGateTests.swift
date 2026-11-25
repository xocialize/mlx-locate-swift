// Offline gates for the engine package: manifest, licence, MAT, CAN, INF, FIT. No kernel runs, no
// weights read — a green suite here does not mean the model works (that is `locate-cli`).
import Foundation
import Locate
import MLXServeConformance
import MLXServeConformanceNN
import MLXToolKit
import XCTest

@testable import MLXLocate

extension LocatePackage: InferenceModeInspectable {
    public func inferenceModeFlags() -> [InferenceModeConformance.ModuleTrainingFlag] {
        InferenceModeConformance.flags(of: inferenceModeGraphs)
    }
}

final class PackageGateTests: XCTestCase {

    func testSurfacesAreAnalysisWithBothModesAndTypedLocate() throws {
        let surfaces = LocatePackage.manifest.surfaces
        // imageAnalysis stays first: the engine derives the package id from the first surface's name.
        XCTAssertEqual(surfaces.map(\.capability), [.imageAnalysis, .imageLocate])
        XCTAssertEqual(surfaces.first?.name, "qwen3-vl-locate")
        XCTAssertEqual(Set(try XCTUnwrap(surfaces.first).supportedModes), [.locate, .answer])
        XCTAssertEqual(Set(surfaces.map(\.name)).count, surfaces.count)
    }

    /// C11: the locate declaration is on the surface, coherent, and what the README says.
    func testLocateControlsDeclaration() throws {
        let surface = try XCTUnwrap(LocatePackage.manifest.surfaces.first { $0.capability == .imageLocate })
        let controls = try XCTUnwrap(surface.imageLocateControls)
        XCTAssertEqual(controls, ImageLocateControls(maxCount: 16, scores: true, seeded: false))
        XCTAssertEqual(controls, LocatePackage.locateControls)
        XCTAssertEqual(controls.declarationProblems, [])
        XCTAssertTrue(surface.parameters.contains { $0.name == "count" })
        XCTAssertFalse(surface.parameters.contains { $0.name == "seed" })
    }

    /// The generation budget holds `maxCount` boxes of an ordinary phrase (41 tokens a box measured
    /// with a four-word label), never drops below v0.2.0's 256, and is bounded for any phrase.
    func testTokenBudgetCoversTheDeclaredCount() {
        let n = LocatePackage.locateControls.maxCount
        XCTAssertGreaterThanOrEqual(Locator.tokenBudget(boxes: n, phrase: "the character's hands"), 16 + n * 41)
        XCTAssertEqual(Locator.tokenBudget(boxes: 1, phrase: "a"), Locator.defaultMaxTokens)
        XCTAssertEqual(Locator.tokenBudget(boxes: n, phrase: String(repeating: "word ", count: 400)),
                       Locator.maxTokenBudget)
    }

    /// The core's boxes through `served(count:)` satisfy the surface's own response check: in
    /// descending score whatever order the model wrote them in, cut to the count, all scored.
    func testServedBoxesPassTheResponseCheck() {
        let raw = [
            LocatedBox(x: 0.1, y: 0.1, width: 0.2, height: 0.2, label: "a", score: 0.4),
            LocatedBox(x: 0.5, y: 0.5, width: 0.5, height: 0.5, label: "a", score: 0.9),
            LocatedBox(x: 0.3, y: 0.0, width: 0.1, height: 0.1, label: "a", score: 0.6),
        ]
        let controls = LocatePackage.locateControls
        let image = Image(format: .png, data: Data())
        for count in [nil, 1, 2, 3, 16] as [Int?] {
            let request = ImageLocateRequest(image: image, prompt: "a", count: count)
            let boxes = LocatePackage.contractBoxes(raw).served(count: request.effectiveCount)
            XCTAssertEqual(controls.responseProblems(ImageLocateResponse(boxes: boxes), for: request), [])
            XCTAssertEqual(boxes.count, min(raw.count, request.effectiveCount))
            XCTAssertEqual(boxes.first?.score, 0.9)
        }
        let none = ImageLocateRequest(image: image, prompt: "a")
        XCTAssertEqual(controls.responseProblems(ImageLocateResponse(boxes: []), for: none), [])
    }

    /// A directly driven package refuses what the engine's pre-flight refuses, before it needs weights.
    func testTypedLocateRefusals() async {
        let package = LocatePackage(configuration: LocateConfiguration())
        let image = Image(format: .png, data: Data())
        let bad: [ImageLocateRequest] = [
            ImageLocateRequest(image: image, prompt: "  "),
            ImageLocateRequest(image: image, prompt: "the hair", count: 0),
            ImageLocateRequest(image: image, prompt: "the hair", count: 17),
            ImageLocateRequest(image: image, prompt: "the hair", mode: .answer),
        ]
        for request in bad {
            do {
                _ = try await package.run(request)
                XCTFail("must be refused: \(request)")
            } catch let e as PackageError {
                guard case .unsupportedRequestFeature = e else { XCTFail("\(e)"); continue }
            } catch { XCTFail("\(error)") }
        }
        // In limits, it gets as far as needing the model.
        do {
            _ = try await package.run(ImageLocateRequest(image: image, prompt: "the hair", count: 16))
            XCTFail("not loaded")
        } catch let e as PackageError {
            XCTAssertEqual(e, .notLoaded)
        } catch { XCTFail("\(error)") }
    }

    func testCANGatePreCancelledTypedRun() async {
        let package = LocatePackage(configuration: LocateConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: ImageLocateRequest(image: Image(format: .png, data: Data()), prompt: "the hair"))
        XCTAssertTrue(report.passed, report.summary)
    }

    /// C7/C8: Apache-2.0 on both layers (Qwen3-VL-4B-Instruct's LICENSE; the port's own).
    func testLicenceIsPermissiveOnBothLayers() {
        let l = LocatePackage.manifest.license
        XCTAssertEqual(l.weightLicense, .apache2)
        XCTAssertEqual(l.portCodeLicense, .apache2)
        XCTAssertEqual(LocatePackage.manifest.provenance.sourceRepo, "Qwen/Qwen3-VL-4B-Instruct")
    }

    /// Declarations never drop below what was measured (2026-11-15, phys_footprint bytes).
    func testFootprintsCoverTheMeasurement() {
        let f = Dictionary(uniqueKeysWithValues: LocatePackage.manifest.requirements.footprints.map { ($0.quant, $0) })
        XCTAssertEqual(Set(f.keys), [.int8, .int4])
        XCTAssertGreaterThanOrEqual(f[.int8]!.residentBytes, 5_238_903_152)
        XCTAssertGreaterThanOrEqual(f[.int4]!.residentBytes, 3_197_569_976)
        XCTAssertGreaterThanOrEqual(f[.int8]!.peakActivationBytes, 2_353_087_424)
        XCTAssertGreaterThanOrEqual(f[.int4]!.peakActivationBytes, 2_416_804_800)
        // 16 boxes at 1024² through the engine (2026-11-25): the process peak fits resident + activation.
        XCTAssertGreaterThanOrEqual(f[.int8]!.residentBytes + f[.int8]!.peakActivationBytes, 8_052_741_544)
        XCTAssertGreaterThanOrEqual(f[.int4]!.residentBytes + f[.int4]!.peakActivationBytes, 6_122_426_352)
    }

    func testFITGate() {
        for quant in LocateConfiguration.servableQuants {
            let report = FootprintConformance.check(
                manifest: LocatePackage.manifest, configuration: LocateConfiguration(quant: quant))
            XCTAssertTrue(report.passed, "\(quant):\n\(report.summary)")
        }
    }

    /// The side cap is what bounds the activation, so no configuration may raise it past the measurement.
    func testSideIsClampedToTheMeasuredEnvelope() {
        XCTAssertEqual(LocateConfiguration(maxSide: 4096).effectiveMaxSide, LocateFootprint.measuredMaxSide)
        XCTAssertEqual(LocateConfiguration(maxSide: 768).effectiveMaxSide, 768)
    }

    func testMATGate() {
        for quant in LocateConfiguration.servableQuants {
            let report = MaterializationConformance.check(freshConfiguration: LocateConfiguration(quant: quant))
            XCTAssertTrue(report.passed, "\(quant):\n\(report.summary)")
        }
    }

    func testWeightSourcesArePinnedAndQuantKeyed() throws {
        let s8 = try XCTUnwrap(LocateConfiguration(quant: .int8).weightSources.first)
        let s4 = try XCTUnwrap(LocateConfiguration(quant: .int4).weightSources.first)
        XCTAssertEqual(s8.repo, "mlx-community/Qwen3-VL-4B-Instruct-8bit")
        XCTAssertEqual(s4.repo, "mlx-community/Qwen3-VL-4B-Instruct-4bit")
        XCTAssertNotNil(s8.revision)
        XCTAssertNotNil(s4.revision)
        for glob in ["model.safetensors", "config.json", "tokenizer.json", "preprocessor_config.json", "chat_template.jinja"] {
            XCTAssertTrue(s8.matching?.contains(glob) ?? false, glob)
        }
        XCTAssertEqual(LocateConfiguration().missingWeightSources(storeRoot: nil).count, 1)
    }

    func testConfigurationDecodesFromAnEmptyPayload() throws {
        let c = try JSONDecoder().decode(LocateConfiguration.self, from: Data("{}".utf8))
        XCTAssertEqual(c.quant, .int8)
        XCTAssertEqual(c.maxSide, 1024)
    }

    func testCANGatePreCancelledRun() async {
        let package = LocatePackage(configuration: LocateConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: ImageAnalysisRequest(image: Image(format: .png, data: Data()), prompt: "the hair"))
        XCTAssertTrue(report.passed, report.summary)
    }

    /// Cadence: per generated token, in the generate phase (`Locator`'s `onToken`).
    func testCANCadenceDeclaration() {
        let report = CancellationConformance.checkCadence(
            manifest: LocatePackage.manifest,
            posture: .cadence([.init(phase: .generate, unit: .token, reportsRunProgress: false)]))
        XCTAssertTrue(report.passed, report.summary)
    }

    /// INF-1 must fail on an unloaded package (proves the seam reads real state). The loaded-graph
    /// green lives in the live lane (`locate-cli engine`).
    func testINFFailsOnUnloadedPackage() async {
        let package = LocatePackage(configuration: LocateConfiguration())
        let report = await InferenceModeConformance.check(
            flags: package.inferenceModeFlags(), posture: .moduleGraph)
        XCTAssertFalse(report.passed)
    }

    func testRejectsUnknownModeAndOtherCapabilities() async {
        let package = LocatePackage(configuration: LocateConfiguration())
        do {
            _ = try await package.run(MattingRequest(image: Image(format: .png, data: Data())))
            XCTFail("matting must be refused")
        } catch let e as PackageError {
            XCTAssertEqual(e, .unsupportedCapability(.matting))
        } catch { XCTFail("\(error)") }
    }
}
