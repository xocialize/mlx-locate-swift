// Offline: the Qwen-VL grounding output parser and the box score.
import Foundation
import XCTest

@testable import Locate

final class GroundingParserTests: XCTestCase {

    /// The exact text Qwen3-VL-4B wrote for ex01's hair (yardstick run, 2026-11-15).
    let ex01Hair = "```json\n[\n\t{\"bbox_2d\": [332, 68, 625, 387], \"label\": \"the character's hair\"}\n]\n```"

    func testParsesFencedListOnTheThousandGrid() throws {
        let boxes = GroundingParser.boxes(from: ex01Hair, fallbackLabel: "x", maxBoxes: 16)
        let b = try XCTUnwrap(boxes.first)
        XCTAssertEqual(boxes.count, 1)
        XCTAssertEqual(b.x, 0.332, accuracy: 1e-9)
        XCTAssertEqual(b.y, 0.068, accuracy: 1e-9)
        XCTAssertEqual(b.width, 0.293, accuracy: 1e-9)
        XCTAssertEqual(b.height, 0.319, accuracy: 1e-9)
        XCTAssertEqual(b.label, "the character's hair")
        XCTAssertEqual(b.score, 1)   // no token data → 1
        // ex01 is 1000×1167: in pixels that is SAM 3's hair box to within a few pixels on x.
        let px = b.pixels(width: 1000, height: 1167)
        XCTAssertEqual(px[0], 332, accuracy: 1e-6)
        XCTAssertEqual(px[3], 387 * 1.167, accuracy: 1e-6)
    }

    func testEmptyListIsNoBoxes() {
        XCTAssertTrue(GroundingParser.boxes(from: "```json\n[]\n```", fallbackLabel: "x", maxBoxes: 16).isEmpty)
        XCTAssertTrue(GroundingParser.boxes(from: "There is no face in this picture.", fallbackLabel: "x",
                                            maxBoxes: 16).isEmpty)
    }

    func testClampsSwapsDropsDegenerateAndDuplicatesAndCaps() {
        let text = """
        [{"bbox_2d": [900, 50, 100, 1200], "label": "a"},
         {"bbox_2d": [10, 10, 10, 500], "label": "flat"},
         {"bbox_2d": [100, 50, 900, 1000], "label": "dup"},
         {"bbox_2d": [0, 0, 500, 500]},
         {"bbox_2d": [1, 1, 2, 2], "label": "over the cap"}]
        """
        let boxes = GroundingParser.boxes(from: text, fallbackLabel: "phrase", maxBoxes: 2)
        XCTAssertEqual(boxes.count, 2)
        XCTAssertEqual(boxes[0].x, 0.1, accuracy: 1e-9)
        XCTAssertEqual(boxes[0].y, 0.05, accuracy: 1e-9)
        XCTAssertEqual(boxes[0].height, 0.95, accuracy: 1e-9)   // 1200 clamped to 1000
        XCTAssertEqual(boxes[1].label, "phrase")                  // missing label → the phrase
    }

    func testTruncatedTailKeepsCompleteEntries() {
        let text = "[{\"bbox_2d\": [10, 20, 30, 40], \"label\": \"a\"}, {\"bbox_2d\": [50, 60, 7"
        XCTAssertEqual(GroundingParser.boxes(from: text, fallbackLabel: "x", maxBoxes: 16).count, 1)
    }

    /// The score is the geometric mean probability of the tokens overlapping the four numbers only.
    func testScoreUsesOnlyCoordinateTokens() {
        let text = "[{\"bbox_2d\": [1, 2, 3, 4]}]"
        let ns = text as NSString
        // One token per character; coordinate digits get logp log(0.5), everything else log(1e-6).
        var ends: [Int] = [], lps: [Float] = []
        let digits = Set("1234")
        for k in 0..<ns.length {
            ends.append(k + 1)
            let ch = Character(ns.substring(with: NSRange(location: k, length: 1)))
            lps.append(digits.contains(ch) ? Float(log(0.5)) : Float(log(1e-6)))
        }
        let b = GroundingParser.boxes(from: text, fallbackLabel: "x", maxBoxes: 1, tokenEnds: ends, logprobs: lps)
        XCTAssertEqual(b.first?.score ?? 0, 0.5, accuracy: 1e-6)
    }
}
