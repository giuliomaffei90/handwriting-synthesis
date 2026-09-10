import CoreGraphics
import Foundation

/// Turning pen movements into something you can look at.
///
/// The per-line treatment is the one from the original demo - scale, smooth
/// away recording noise, correct the slant, flip the y axis - and the page is
/// then cropped to the writing rather than a fixed box.
enum Drawing {

    static let lineHeight: CGFloat = 60
    static let scale: CGFloat = 1.5
    static let margin: CGFloat = 24

    /// Savitzky-Golay, 7 wide, cubic: for a fixed window the smoothing
    /// coefficients are constant, so the whole filter is one convolution.
    private static let savitzkyGolay: [CGFloat] = [-2, 3, 6, 7, 6, 3, -2].map { $0 / 21 }

    private static func smoothed(_ values: [CGFloat]) -> [CGFloat] {
        guard let first = values.first, let last = values.last else { return values }
        var padded = [CGFloat](repeating: first, count: 3) + values
        padded.append(contentsOf: [CGFloat](repeating: last, count: 3))
        return (0..<values.count).map { i in
            zip(savitzkyGolay, padded[i..<(i + 7)]).reduce(0) { $0 + $1.0 * $1.1 }
        }
    }

    /// Corrects the global slant and offset of a hand.
    private static func aligned(_ points: [CGPoint]) -> [CGPoint] {
        let n = CGFloat(points.count)
        guard n > 1 else { return points }
        let sumX = points.reduce(0) { $0 + $1.x }
        let sumY = points.reduce(0) { $0 + $1.y }
        let sumXX = points.reduce(0) { $0 + $1.x * $1.x }
        let sumXY = points.reduce(0) { $0 + $1.x * $1.y }
        let determinant = n * sumXX - sumX * sumX
        guard abs(determinant) > 1e-9 else { return points }
        let offset = (sumXX * sumY - sumX * sumXY) / determinant
        let slope = (n * sumXY - sumX * sumY) / determinant

        let theta = atan(slope)
        let c = cos(theta), s = sin(theta)
        return points.map {
            CGPoint(x: $0.x * c + $0.y * s - offset, y: -$0.x * s + $0.y * c - offset)
        }
    }

    /// One line of pen movements becomes one array of points per pen stroke.
    static func strokes(from offsets: [SIMD3<Float>], scale: CGFloat = scale,
                        denoise: Bool = true, align: Bool = true) -> [[CGPoint]] {
        guard !offsets.isEmpty else { return [] }

        var x: CGFloat = 0, y: CGFloat = 0
        var coordinates: [CGPoint] = []
        var lifts: [Bool] = []
        coordinates.reserveCapacity(offsets.count)
        for point in offsets {
            x += CGFloat(point.x) * scale
            y += CGFloat(point.y) * scale
            coordinates.append(CGPoint(x: x, y: y))
            lifts.append(point.z == 1)
        }

        if denoise && coordinates.count > 1 {
            var start = 0
            var smoothedPoints: [CGPoint] = []
            for end in 0..<coordinates.count where lifts[end] || end == coordinates.count - 1 {
                let piece = Array(coordinates[start...end])
                let xs = smoothed(piece.map(\.x)), ys = smoothed(piece.map(\.y))
                smoothedPoints.append(contentsOf: zip(xs, ys).map(CGPoint.init))
                start = end + 1
            }
            coordinates = smoothedPoints
        }
        if align && coordinates.count > 1 { coordinates = aligned(coordinates) }
        coordinates = coordinates.map { CGPoint(x: $0.x, y: -$0.y) }

        var result: [[CGPoint]] = []
        var current: [CGPoint] = []
        for (index, point) in coordinates.enumerated() {
            current.append(point)
            if lifts[index] {
                result.append(current)
                current = []
            }
        }
        if !current.isEmpty { result.append(current) }
        return result.filter { !$0.isEmpty }
    }

    struct Page {
        var strokes: [[CGPoint]]
        var size: CGSize
        var isEmpty: Bool { strokes.isEmpty }
    }

    enum Alignment { case left, centre, right }

    /// Stack the lines into a page, cropped to the writing.
    static func page(of lines: [[SIMD3<Float>]], alignment: Alignment = .left,
                     lineHeight: CGFloat = lineHeight, margin: CGFloat = margin) -> Page {
        let perLine = lines.map { strokes(from: $0) }
        let widths: [CGFloat] = perLine.map { line in
            let xs = line.flatMap { $0.map(\.x) }
            guard let low = xs.min(), let high = xs.max() else { return 0 }
            return high - low
        }
        let pageWidth = widths.max() ?? 0

        var placed: [[CGPoint]] = []
        for (index, line) in perLine.enumerated() where !line.isEmpty {
            let xs = line.flatMap { $0.map(\.x) }
            var dx = -(xs.min() ?? 0)
            switch alignment {
            case .left: break
            case .centre: dx += (pageWidth - widths[index]) / 2
            case .right: dx += pageWidth - widths[index]
            }
            let dy = CGFloat(index) * lineHeight
            placed.append(contentsOf: line.map { stroke in
                stroke.map { CGPoint(x: $0.x + dx, y: $0.y + dy) }
            })
        }

        let all = placed.flatMap { $0 }
        guard let minX = all.map(\.x).min(), let minY = all.map(\.y).min(),
              let maxX = all.map(\.x).max(), let maxY = all.map(\.y).max()
        else { return Page(strokes: [], size: CGSize(width: 1, height: 1)) }

        let shifted = placed.map { stroke in
            stroke.map { CGPoint(x: $0.x - minX + margin, y: $0.y - minY + margin) }
        }
        return Page(strokes: shifted,
                    size: CGSize(width: maxX - minX + 2 * margin,
                                 height: maxY - minY + 2 * margin))
    }

    // MARK: - the pen path
    //
    // The network puts down points about a unit apart, so joining them with
    // straight lines shows facets once enlarged. Every interior point becomes
    // the control point of a quadratic curve running between its two
    // neighbouring segment midpoints, which both the screen and the exported
    // file follow.

    static func path(for stroke: [CGPoint]) -> CGMutablePath {
        let path = CGMutablePath()
        guard let start = stroke.first else { return path }
        path.move(to: start)
        if stroke.count < 3 {
            for point in stroke.dropFirst() { path.addLine(to: point) }
            return path
        }
        for index in 1..<(stroke.count - 1) {
            let next = stroke[index + 1]
            let midpoint = CGPoint(x: (stroke[index].x + next.x) / 2,
                                   y: (stroke[index].y + next.y) / 2)
            path.addQuadCurve(to: midpoint, control: stroke[index])
        }
        path.addLine(to: stroke[stroke.count - 1])
        return path
    }

    static func svgPath(for stroke: [CGPoint]) -> String {
        guard let start = stroke.first else { return "" }
        var d = String(format: "M%.2f,%.2f", start.x, start.y)
        if stroke.count < 3 {
            for point in stroke.dropFirst() { d += String(format: " L%.2f,%.2f", point.x, point.y) }
            return d
        }
        for index in 1..<(stroke.count - 1) {
            let next = stroke[index + 1]
            d += String(format: " Q%.2f,%.2f %.2f,%.2f", stroke[index].x, stroke[index].y,
                        (stroke[index].x + next.x) / 2, (stroke[index].y + next.y) / 2)
        }
        let last = stroke[stroke.count - 1]
        return d + String(format: " L%.2f,%.2f", last.x, last.y)
    }
}
