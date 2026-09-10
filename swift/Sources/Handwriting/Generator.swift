import Foundation

/// Writes lines of text, retrying the ones the model fails to finish.
///
/// A line is accepted when the window runs past its last character inside the
/// step budget. Styles that rarely lift the pen can get stuck rewriting a word
/// instead, which reads as invented gibberish, so those lines are written again
/// from the same primed state; one that never comes out right falls back to the
/// attempt that read furthest.
struct Generator {

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

        let written = request.lines.enumerated().filter { !$0.element.trimmingCharacters(
            in: .whitespaces).isEmpty }
        var output = [[SIMD3<Float>]](repeating: [], count: request.lines.count)

        for (order, item) in written.enumerated() {
            let (index, line) = item
            let codes = Alphabet.encode(prefix + line)
            let context = Engine.Context(codes: codes, length: codes.count)

            var primed = engine.zeroState(width: context.width)
            if let style {
                engine.teacherForce(strokes: style.strokes, context: context, state: &primed)
            }

            let limit = Int((Engine.stepBudget * pace * Float(line.count)).rounded(.up))
            var best: [SIMD3<Float>] = []
            var bestRead = -Float.greatestFiniteMagnitude

            for _ in 0..<Engine.attempts {
                // primed sampling takes its first input from the state, plain
                // sampling starts with the pen lifted at the origin
                let first = style == nil
                    ? SIMD3<Float>(0, 0, 1)
                    : engine.sample(primed, bias: request.bias, using: &rng)

                let run = engine.freeRun(
                    state: primed, context: context, bias: request.bias, limit: limit,
                    first: first, using: &rng,
                    progress: progress.map { report in
                        { fraction in
                            report((Double(order) + fraction) / Double(written.count))
                        }
                    })

                if run.finished {
                    best = run.points
                    break
                }
                if run.readAt > bestRead {
                    bestRead = run.readAt
                    best = run.points
                }
            }
            output[index] = best
        }
        return output
    }
}
