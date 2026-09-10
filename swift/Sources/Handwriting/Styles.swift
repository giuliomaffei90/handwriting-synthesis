import Foundation

/// The handwriting samples the model is primed with. A style is not trained:
/// it is one line someone wrote, shown to the model before it starts.
struct Styles {

    struct Style {
        var id: Int
        var text: String
        var strokes: [SIMD3<Float>]

        /// Timesteps this hand spends per character - the pace a line is budgeted against.
        var pace: Float { Float(strokes.count) / Float(max(text.count, 1)) }
    }

    private(set) var all: [Style] = []

    init(url: URL = Resources.url("styles")) throws {
        let container = try Container(contentsOf: url)
        var ids: [Int] = []
        for name in container.strings.keys where name.hasSuffix(".text") {
            if let id = Int(name.dropFirst("style".count).dropLast(".text".count)) { ids.append(id) }
        }
        for id in ids.sorted() {
            guard let text = container.strings["style\(id).text"] else { continue }
            let tensor = try container.tensor("style\(id).strokes")
            var strokes: [SIMD3<Float>] = []
            strokes.reserveCapacity(tensor.rows)
            for t in 0..<tensor.rows {
                let base = t * 3
                strokes.append(SIMD3(tensor.values[base], tensor.values[base + 1],
                                     tensor.values[base + 2]))
            }
            all.append(Style(id: id, text: text, strokes: strokes))
        }
    }

    subscript(id: Int) -> Style? { all.first { $0.id == id } }
}
