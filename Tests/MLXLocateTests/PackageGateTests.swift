// Offline gates for the engine package: manifest, licence, MAT, CAN, INF, FIT. No kernel runs, no
// weights read — a green suite here does not mean the model works (that is `locate-cli`).
import Foundation
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

    func testOneImageAnalysisSurfaceWithBothModes() throws {
        let surfaces = LocatePackage.manifest.surfaces
        XCTAssertEqual(surfaces.map(\.capability), [.imageAnalysis])
        XCTAssertEqual(Set(try XCTUnwrap(surfaces.first).supportedModes), [.locate, .answer])
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
