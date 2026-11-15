// locate-cli — gates for mlx-locate-swift.
//
//   locate-cli query     --model DIR --image P --phrase "…" [--count N] [--side 1024]
//   locate-cli yardstick --model DIR --images a.png,b.png [--phrases "a|b"] [--side 1024] [--template single] [--out rows.jsonl]
//   locate-cli mem-bench --model DIR --image P [--side 1024]     (one process per number)
//   locate-cli engine    --model DIR --image P --phrase "…" [--quant int8|int4]   (through MLXServeEngine)
//
// Every line printed is one JSON object. Timings need a Release build.
import CoreGraphics
import Foundation
import ImageIO
import Locate
import MLX
import MLXLocate
import MLXServeCore
import MLXToolKit

func vmInfo() -> task_vm_info_data_t {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    _ = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return info
}

func loadCG(_ path: String) -> CGImage {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        FileHandle.standardError.write(Data("cannot read \(path)\n".utf8)); exit(2)
    }
    return cg
}

nonisolated(unsafe) var outHandle: FileHandle?
func emit(_ d: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: d, options: [.sortedKeys])
    print(String(decoding: data, as: UTF8.self))
    outHandle?.write(data); outHandle?.write(Data("\n".utf8))
}

func boxesJSON(_ boxes: [LocatedBox], _ w: Int, _ h: Int) -> [[String: Any]] {
    boxes.map { ["x": $0.x, "y": $0.y, "width": $0.width, "height": $0.height, "label": $0.label,
                 "score": $0.score, "px": $0.pixels(width: w, height: h)] }
}

let argv = Array(CommandLine.arguments.dropFirst())
guard let command = argv.first else {
    FileHandle.standardError.write(Data("usage: locate-cli query|yardstick|mem-bench|engine --model DIR …\n".utf8))
    exit(2)
}
var opts = [String: String]()
var i = 1
while i < argv.count {
    if argv[i].hasPrefix("--") { opts[String(argv[i].dropFirst(2))] = i + 1 < argv.count ? argv[i + 1] : ""; i += 2 } else { i += 1 }
}
let modelDir = URL(fileURLWithPath: opts["model"] ?? "")
let side = Int(opts["side"] ?? "1024") ?? 1024
if let o = opts["out"] {
    FileManager.default.createFile(atPath: o, contents: nil)
    outHandle = FileHandle(forWritingAtPath: o)
}

switch command {
case "query", "yardstick":
    let t0 = Date()
    let locator = try await Locator.load(directory: modelDir)
    emit(["event": "load", "model": modelDir.lastPathComponent, "sec": Date().timeIntervalSince(t0),
          "mlx_active_gb": Double(Memory.activeMemory) / 1e9, "phys_gb": Double(vmInfo().phys_footprint) / 1e9])
    Memory.peakMemory = 0
    let images = command == "query" ? [opts["image"]!] : opts["images"]!.split(separator: ",").map(String.init)
    let phrases = command == "query" ? [opts["phrase"]!]
        : (opts["phrases"] ?? "the character's hair|the character's face|the character's head")
            .split(separator: "|").map(String.init)
    let count = Int(opts["count"] ?? "16") ?? 16
    let repeats = Int(opts["repeat"] ?? (command == "query" ? "1" : "2")) ?? 2
    for path in images {
        let cg = loadCG(path)
        for phrase in phrases {
            for rep in 0..<repeats {
                let t = Date()
                let template = opts["template"] == "single" ? Locator.singleObjectTemplate : Locator.promptTemplate
                let r = try await locator.locate(image: cg, phrase: phrase, maxBoxes: count, maxSide: side,
                                                 template: template)
                emit(["event": "box", "image": URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent,
                      "phrase": phrase, "rep": rep, "size": [cg.width, cg.height], "seen": [r.seenWidth, r.seenHeight],
                      "boxes": boxesJSON(r.boxes, cg.width, cg.height), "text": r.text,
                      "sec": Date().timeIntervalSince(t), "prompt_tokens": r.promptTokens,
                      "gen_tokens": r.generatedTokens, "mlx_peak_gb": Double(Memory.peakMemory) / 1e9])
            }
        }
    }
    emit(["event": "done", "mlx_peak_gb": Double(Memory.peakMemory) / 1e9,
          "phys_gb": Double(vmInfo().phys_footprint) / 1e9,
          "phys_peak_gb": Double(vmInfo().ledger_phys_footprint_peak) / 1e9])

case "mem-bench":
    let base = vmInfo().phys_footprint
    let locator = try await Locator.load(directory: modelDir)
    let floor = vmInfo().phys_footprint      // post-load, pre-run: the resident floor
    let cg = loadCG(opts["image"]!)
    let t = Date()
    let r = try await locator.locate(image: cg, phrase: opts["phrase"] ?? "the character's hair", maxSide: side)
    let sec = Date().timeIntervalSince(t)
    let peak = UInt64(vmInfo().ledger_phys_footprint_peak)
    let after = vmInfo().phys_footprint
    emit(["event": "mem", "model": modelDir.lastPathComponent, "side": side, "seen": [r.seenWidth, r.seenHeight],
          "prompt_tokens": r.promptTokens, "gen_tokens": r.generatedTokens, "sec_cold_first_query": sec,
          "base_bytes": base, "floor_bytes": floor, "peak_bytes": peak, "after_bytes": after,
          "activation_bytes": peak > floor ? peak - floor : 0,
          "mlx_peak_bytes": Memory.peakMemory, "mlx_active_bytes": Memory.activeMemory])
    print(String(format: "[MEM-BYTES] floorBytes=%llu peakBytes=%llu activationBytes=%llu (floor %.3f GB · peak %.3f GB · activation %.3f GB)",
                 floor, peak, peak > floor ? peak - floor : 0, Double(floor) / 1e9, Double(peak) / 1e9,
                 Double(peak > floor ? peak - floor : 0) / 1e9))

case "engine":
    let quant: Quant = opts["quant"] == "int4" ? .int4 : .int8
    let cg = loadCG(opts["image"]!)
    let data = try Data(contentsOf: URL(fileURLWithPath: opts["image"]!))
    let image = MLXToolKit.Image(format: .png, data: data, width: cg.width, height: cg.height)
    let engine = MLXServeEngine()
    await engine.useModelStore(ModelStore(root: nil))
    let id = try await engine.register(
        LocatePackage.registration, configuration: LocateConfiguration(quant: quant, modelDirectory: modelDir))
    var meta: MetaData = [:]
    if let c = opts["count"], let n = Int(c) { meta["count"] = .int(n) }
    for rep in 0..<2 {
        let t = Date()
        let resp = try await engine.run(
            ImageAnalysisRequest(image: image, prompt: opts["phrase"]!, mode: .locate, metaData: meta), package: id)
        guard let text = (resp as? ImageAnalysisResponse)?.text else { exit(3) }
        emit(["event": "engine", "rep": rep, "sec": Date().timeIntervalSince(t), "text": text])
    }
    if let q = opts["question"] {
        let resp = try await engine.run(ImageAnalysisRequest(image: image, prompt: q, mode: .answer), package: id)
        emit(["event": "engine-answer", "text": (resp as? ImageAnalysisResponse)?.text ?? ""])
    }

default:
    FileHandle.standardError.write(Data("unknown command \(command)\n".utf8))
    exit(2)
}
