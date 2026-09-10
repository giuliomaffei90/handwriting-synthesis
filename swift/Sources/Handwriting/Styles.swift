import Foundation

/// The handwriting samples the model is primed with. A style is not trained:
/// it is one line someone wrote, shown to the model before it starts.
struct Styles {

    struct Style {
        var id: Int
        var text: String
        var strokes: [SIMD3<Float>]
        /// The same sentence written in this hand, generated ahead of time so
        /// the picker can show it the instant a style is chosen.
        var preview: [SIMD3<Float>]

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
        func points(_ name: String) -> [SIMD3<Float>] {
            guard let tensor = container.tensors[name] else { return [] }
            return (0..<tensor.rows).map { t in
                SIMD3(tensor.values[t * 3], tensor.values[t * 3 + 1], tensor.values[t * 3 + 2])
            }
        }

        for id in ids.sorted() {
            guard let text = container.strings["style\(id).text"] else { continue }
            let strokes = points("style\(id).strokes")
            guard !strokes.isEmpty else { continue }
            let preview = points("style\(id).preview")
            // fall back to the priming sample if the app was built without previews
            all.append(Style(id: id, text: text, strokes: strokes,
                             preview: preview.isEmpty ? strokes : preview))
        }
    }

    subscript(id: Int) -> Style? { all.first { $0.id == id } }
}
