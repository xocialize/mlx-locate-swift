import Foundation

/// One located region, in the `image.locate` shape: `x`, `y`, `width`, `height` are fractions (0...1)
/// of the upright picture, origin top-left.
public struct LocatedBox: Codable, Sendable, Equatable {
    public var x: Double
    public var y: Double
    public var width: Double
    public var height: Double
    public var label: String
    /// The model's confidence in the box: the geometric mean probability of the tokens that spell its
    /// four coordinates (greedy decoding). Useful for ranking and thresholding; it is not calibrated.
    public var score: Double

    public init(x: Double, y: Double, width: Double, height: Double, label: String, score: Double) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.score = score
    }

    /// Corners in pixels of a `width × height` picture: `[x0, y0, x1, y1]`.
    public func pixels(width w: Int, height h: Int) -> [Double] {
        [x * Double(w), y * Double(h), (x + width) * Double(w), (y + height) * Double(h)]
    }
}

/// Reads Qwen-VL grounding output: a JSON-ish list of `{"bbox_2d": [x1, y1, x2, y2], "label": "…"}`,
/// coordinates relative to the picture on a 0...1000 grid (Qwen3-VL's convention, checked against
/// mlx-3d's SAM 3 boxes: the y axis only lines up as 0...1000 on a non-square picture).
///
/// Parsing is deliberately tolerant (markdown fences, trailing prose, a truncated last entry): it
/// matches each `{...}` object that holds a `bbox_2d` and ignores the rest.
public enum GroundingParser {

    /// One parsed entry, with the UTF-16 ranges of its four numbers in `text` (for scoring).
    public struct Entry: Equatable, Sendable {
        public var box: [Double]          // x1, y1, x2, y2 on 0...1000
        public var label: String?
        public var numberRanges: [NSRange]
    }

    private static let objectRE = try! NSRegularExpression(pattern: #"\{[^{}]*\}"#)
    private static let boxRE = try! NSRegularExpression(
        pattern: #""bbox_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"#)
    private static let labelRE = try! NSRegularExpression(pattern: #""label"\s*:\s*"((?:[^"\\]|\\.)*)""#)

    public static func entries(in text: String) -> [Entry] {
        let ns = text as NSString
        var out: [Entry] = []
        for obj in objectRE.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            guard let m = boxRE.firstMatch(in: text, range: obj.range) else { continue }
            let ranges = (1...4).map { m.range(at: $0) }
            let box = ranges.map { Double(ns.substring(with: $0)) ?? .nan }
            let label = labelRE.firstMatch(in: text, range: obj.range).map { ns.substring(with: $0.range(at: 1)) }
            out.append(Entry(box: box, label: label, numberRanges: ranges))
        }
        return out
    }

    /// Entries → boxes: clamped to the picture, degenerate and duplicate boxes dropped, at most
    /// `maxBoxes` in the model's own order. `tokenEnds[i]` is the UTF-16 length of the decoded text
    /// after generated token `i`, and `logprobs[i]` that token's log-probability; with both, a box's
    /// score is the geometric mean probability of the tokens overlapping its four numbers, else 1.
    public static func boxes(
        from text: String, fallbackLabel: String, maxBoxes: Int,
        tokenEnds: [Int] = [], logprobs: [Float] = []
    ) -> [LocatedBox] {
        var out: [LocatedBox] = []
        for e in entries(in: text) {
            guard e.box.allSatisfy({ $0.isFinite }) else { continue }
            let c = e.box.map { min(max($0, 0), 1000) / 1000 }
            let (x0, y0, x1, y1) = (min(c[0], c[2]), min(c[1], c[3]), max(c[0], c[2]), max(c[1], c[3]))
            guard x1 - x0 > 1e-3, y1 - y0 > 1e-3 else { continue }
            let label = (e.label?.isEmpty == false ? e.label! : fallbackLabel)
            let b = LocatedBox(x: x0, y: y0, width: x1 - x0, height: y1 - y0, label: label,
                               score: score(e.numberRanges, tokenEnds: tokenEnds, logprobs: logprobs))
            if out.contains(where: { abs($0.x - b.x) < 1e-6 && abs($0.y - b.y) < 1e-6
                && abs($0.width - b.width) < 1e-6 && abs($0.height - b.height) < 1e-6 }) { continue }
            out.append(b)
            if out.count >= maxBoxes { break }
        }
        return out
    }

    static func score(_ ranges: [NSRange], tokenEnds: [Int], logprobs: [Float]) -> Double {
        guard !tokenEnds.isEmpty, tokenEnds.count == logprobs.count else { return 1 }
        var sum = 0.0, n = 0
        var start = 0
        for (i, end) in tokenEnds.enumerated() {
            defer { start = end }
            guard end > start else { continue }
            let tok = NSRange(location: start, length: end - start)
            if ranges.contains(where: { NSIntersectionRange($0, tok).length > 0 }) {
                sum += Double(logprobs[i]); n += 1
            }
        }
        return n == 0 ? 1 : exp(sum / Double(n))
    }
}
