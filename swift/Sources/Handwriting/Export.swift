import AppKit
import CoreGraphics
import Foundation

/// Saving a page as vectors or pixels. Both are drawn with a transparent
/// background, so handwriting drops straight onto whatever is underneath.
enum Export {

    static func hex(_ colour: NSColor) -> String {
        let rgb = colour.usingColorSpace(.sRGB) ?? .black
        return String(format: "#%02X%02X%02X",
                      Int((rgb.redComponent * 255).rounded()),
                      Int((rgb.greenComponent * 255).rounded()),
                      Int((rgb.blueComponent * 255).rounded()))
    }

    static func svg(_ page: Drawing.Page, penWidth: CGFloat, colour: NSColor) -> String {
        var lines = [
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
            String(format: "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%.0f\" "
                   + "height=\"%.0f\" viewBox=\"0 0 %.2f %.2f\">",
                   page.size.width, page.size.height, page.size.width, page.size.height),
            String(format: "<g fill=\"none\" stroke=\"%@\" stroke-width=\"%.2f\" "
                   + "stroke-linecap=\"round\" stroke-linejoin=\"round\">", hex(colour), penWidth),
        ]
        for stroke in page.strokes {
            lines.append("<path d=\"\(Drawing.svgPath(for: stroke))\"/>")
        }
        lines.append("</g></svg>")
        return lines.joined(separator: "\n")
    }

    static func png(_ page: Drawing.Page, penWidth: CGFloat, colour: NSColor,
                    scale: CGFloat = 2) -> Data? {
        let width = max(1, Int((page.size.width * scale).rounded()))
        let height = max(1, Int((page.size.height * scale).rounded()))
        guard let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }

        // page coordinates run downwards, the context upwards
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: scale, y: -scale)
        draw(page, in: context, penWidth: penWidth, colour: colour)

        guard let image = context.makeImage() else { return nil }
        let representation = NSBitmapImageRep(cgImage: image)
        return representation.representation(using: .png, properties: [:])
    }

    static func draw(_ page: Drawing.Page, in context: CGContext, penWidth: CGFloat,
                     colour: NSColor) {
        context.setStrokeColor((colour.usingColorSpace(.sRGB) ?? .black).cgColor)
        context.setLineWidth(penWidth)
        context.setLineCap(.round)
        context.setLineJoin(.round)
        for stroke in page.strokes {
            context.addPath(Drawing.path(for: stroke))
            context.strokePath()
        }
    }
}
