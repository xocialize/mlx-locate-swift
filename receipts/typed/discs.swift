import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
let w = 1024, h = 1024  // discs25.png; discs20.png was 1024 x 768, 4 rows, cy = 100 + r * 185
let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                    space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
ctx.setFillColor(red: 0.96, green: 0.96, blue: 0.94, alpha: 1); ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
let cols: [(CGFloat, CGFloat, CGFloat)] = [(0.8,0.15,0.15),(0.15,0.35,0.8),(0.15,0.63,0.27),(0.9,0.63,0.1),(0.55,0.23,0.67)]
var seed: UInt64 = 7
func rnd(_ n: Int) -> Int { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Int((seed >> 33) % UInt64(n)) }
var k = 0
for r in 0..<5 { for c in 0..<5 {
    let cx = 110 + c * 200 + rnd(31) - 15, cy = 110 + r * 200 + rnd(31) - 15, rad = 45 + rnd(26)
    let col = cols[k % 5]; k += 1
    ctx.setFillColor(red: col.0, green: col.1, blue: col.2, alpha: 1)
    ctx.fillEllipse(in: CGRect(x: cx - rad, y: cy - rad, width: 2 * rad, height: 2 * rad))
} }
let url = URL(fileURLWithPath: CommandLine.arguments[1])
let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
CGImageDestinationAddImage(dst, ctx.makeImage()!, nil); CGImageDestinationFinalize(dst)
