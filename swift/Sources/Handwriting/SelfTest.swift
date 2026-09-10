import Foundation

/// Replays the trace captured from the original TensorFlow graph and checks
/// this Swift port against it, the same way `tests/test_engine.py` checks the
/// numpy one. Run the app with `--selftest`.
enum SelfTest {

    static func run() throws {
        let engine = try Engine()
        print("model: \(engine.lstmSize)-unit LSTMs, \(engine.attentionMixtures) window "
              + "components, \(engine.outputMixtures) output components, "
              + "\(engine.alphabetSize) characters")

        let reference = try Container(contentsOf: Resources.url("reference"))
        let x = try reference.tensor("x")
        let lengths = try reference.tensor("x_len")
        let codes = try reference.tensor("c")
        let codeLengths = try reference.tensor("c_len")
        let expected = try reference.tensor("params")

        let rows = x.shape[0], steps = x.shape[1], width = codes.shape[1]
        var worstParameters: Float = 0
        var worstState: Float = 0

        for row in 0..<rows {
            let live = Int(lengths.values[row])
            var strokes: [SIMD3<Float>] = []
            for t in 0..<live {
                let base = (row * steps + t) * 3
                strokes.append(SIMD3(x.values[base], x.values[base + 1], x.values[base + 2]))
            }
            let context = Engine.Context(
                codes: (0..<width).map { Int(codes.values[row * width + $0]) },
                length: Int(codeLengths.values[row]))

            var state = engine.zeroState(width: width)
            let produced = engine.teacherForce(strokes: strokes, context: context, state: &state)

            let stride = expected.shape[2]
            for t in 0..<live {
                for k in 0..<stride {
                    let want = expected.values[(row * steps + t) * stride + k]
                    worstParameters = max(worstParameters, abs(produced[t][k] - want))
                }
            }

            let fields: [(String, [Float])] = [
                ("h1", state.h1), ("c1", state.c1), ("h2", state.h2), ("c2", state.c2),
                ("h3", state.h3), ("c3", state.c3), ("alpha", state.alpha),
                ("beta", state.beta), ("kappa", state.kappa), ("w", state.window),
                ("phi", state.phi),
            ]
            for (name, values) in fields {
                let want = try reference.tensor("state_" + name)
                let stride = want.shape[1]
                var worst: Float = 0
                for k in 0..<values.count {
                    worst = max(worst, abs(values[k] - want.values[row * stride + k]))
                }
                worstState = max(worstState, worst)
                if row == 0 {
                    print(String(format: "  state.%-6@ max abs err %.2e", name as NSString, worst))
                }
            }
        }

        print(String(format: "mixture parameters: max abs err %.2e over %d recurrent steps",
                     worstParameters, Int(lengths.values.max() ?? 0)))
        guard worstParameters < 1e-3, worstState < 1e-3 else {
            throw Failure.diverged(parameters: worstParameters, state: worstState)
        }

        try checkText()
        try checkStyles()
        print("SELFTEST OK")
    }

    /// Every style has to carry the sentence the picker shows, or a stale
    /// resource file quietly puts the priming samples back.
    private static func checkStyles() throws {
        let container = try Container(contentsOf: Resources.url("styles"))
        let styles = try Styles()
        var missing: [Int] = []
        for style in styles.all where container.tensors["style\(style.id).preview"] == nil {
            missing.append(style.id)
        }
        guard missing.isEmpty else {
            throw Failure.styles(missing)
        }
        print("styles: \(styles.all.count), each with its own preview line")
    }

    private static func checkText() throws {
        // an accent becomes the letter and an apostrophe, the way it is typed on
        // a machine without accents; other marks are simply flattened
        let (text, dropped) = Alphabet.sanitize(
            "Perch\u{e9} Qui: 3 \u{201C}test\u{201D} \u{2013} ok\u{00A0}\u{2029}")
        guard text == "Perche' qui: 3 \"test\" - ok " else { throw Failure.text("sanitize gave \(text)") }
        guard dropped == ["\u{2029}"] else { throw Failure.text("dropped \(dropped)") }
        guard Alphabet.sanitize("caff\u{e8} citt\u{e0} virt\u{f9}").text == "caffe' citta' virtu'"
        else { throw Failure.text("accents") }
        guard Alphabet.sanitize("\u{c8} vero").text == "E' vero" else { throw Failure.text("capital accent") }
        guard Alphabet.sanitize("gar\u{e7}on ma\u{f1}ana f\u{fc}r").text == "garcon manana fur"
        else { throw Failure.text("other marks") }
        guard Alphabet.wrap("a\nb") == ["a", "b"] else { throw Failure.text("wrap") }
        guard Alphabet.wrap(String(repeating: "word ", count: 100)).allSatisfy({ $0.count <= 75 })
        else { throw Failure.text("wrap width") }
        print("text preparation ok")
    }

    enum Failure: Error, CustomStringConvertible {
        case diverged(parameters: Float, state: Float)
        case text(String)
        case styles([Int])

        var description: String {
            switch self {
            case let .diverged(p, s):
                return "diverged from TensorFlow: parameters \(p), state \(s)"
            case let .text(what): return "text preparation: \(what)"
            case let .styles(ids):
                return "styles \(ids) have no preview line - repack the resources"
            }
        }
    }
}
