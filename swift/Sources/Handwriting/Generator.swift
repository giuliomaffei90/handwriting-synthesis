import Foundation

/// Writes lines of text, in pieces, retrying the ones the model gets wrong.
///
/// A piece is accepted when the window runs past its last character inside the
/// step budget without the reading having fallen back. Styles that rarely lift
/// the pen otherwise get stuck rewriting a word, and any style asked for more
/// text than it was trained on can lose the thread outright; both read as
/// invented gibberish. A piece that fails is written again from the same primed
/// state, and one that never comes out right falls back to the attempt that
/// read furthest, cut to where its reading last moved on.
struct Generator {

    /// The corpus the model learnt from is made of short lines: half are 29
    /// characters or fewer, only one in fifty reaches 45, and not one reaches
    /// 63. Asking for more than it has ever seen is where most of the gibberish
    /// comes from, so a line is written in pieces this long and joined back
    /// together on one baseline, which reads as a single line.
    static let pieceWidth = 38

    let engine: Engine
    let styles: Styles

    init() throws {
        engine = try Engine()
        styles = try Styles()
    }

    struct Request {
        var lines: [String]
        var style: Int?
        var bias: Float
        var seed: UInt64
    }

    /// Returns one array of pen movements per line; empty lines give nothing.
    func write(_ request: Request, progress: ((Double) -> Bool)? = nil) -> [[SIMD3<Float>]] {
        var rng = RandomSource(seed: request.seed)
        let style = request.style.flatMap { styles[$0] }
        let prefix = style.map { $0.text + " " } ?? ""
        let pace = style?.pace ?? Engine.defaultPace

        // every piece of every line, so progress can count them all
        var work: [(line: Int, text: String)] = []
        for (index, line) in request.lines.enumerated() {
            for piece in Alphabet.wrap(line, width: Generator.pieceWidth)
            where !piece.trimmingCharacters(in: .whitespaces).isEmpty {
                work.append((index, piece))
            }
        }

        var parts = [Int: [[SIMD3<Float>]]]()
        for (done, job) in work.enumerated() {
            let strokes = writePiece(job.text, prefix: prefix, style: style, pace: pace,
                                     bias: request.bias, rng: &rng) { fraction in
                progress?((Double(done) + fraction) / Double(work.count)) ?? true
            }
            parts[job.line, default: []].append(strokes)
        }

        var output = [[SIMD3<Float>]](repeating: [], count: request.lines.count)
        for (index, pieces) in parts {
            output[index] = Drawing.join(pieces)
        }
        return output
    }

    /// One piece: primed once, then written until it comes out right.
    private func writePiece(_ text: String, prefix: String, style: Styles.Style?, pace: Float,
                            bias: Float, rng: inout RandomSource,
                            progress: @escaping (Double) -> Bool) -> [SIMD3<Float>] {
        let codes = Alphabet.encode(prefix + text)
        let context = Engine.Context(codes: codes, length: codes.count)

        var primed = engine.zeroState(width: context.width)
        if let style {
            engine.teacherForce(strokes: style.strokes, context: context, state: &primed)
        }

        let limit = Int((Engine.stepBudget * pace * Float(text.count)).rounded(.up))
        var best: [SIMD3<Float>] = []
        var bestRead = -Float.greatestFiniteMagnitude

        for _ in 0..<Engine.attempts {
            // primed sampling takes its first input from the state, plain
            // sampling starts with the pen lifted at the origin
            let first = style == nil
                ? SIMD3<Float>(0, 0, 1)
                : engine.sample(primed, bias: bias, using: &rng)

            let run = engine.freeRun(state: primed, context: context, bias: bias, limit: limit,
                                     first: first, using: &rng, progress: progress)
            if run.finished { return run.points }
            if run.readAt > bestRead {
                bestRead = run.readAt
                best = run.points
            }
        }
        return best
    }
}
