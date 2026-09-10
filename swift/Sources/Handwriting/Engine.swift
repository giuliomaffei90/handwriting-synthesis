import Accelerate
import Foundation

@inline(__always) private func sigmoid(_ x: Float) -> Float { 0.5 * (tanhf(0.5 * x) + 1) }
@inline(__always) private func softplus(_ x: Float) -> Float { log1pf(expf(-abs(x))) + max(x, 0) }

/// The handwriting network: three LSTMs, a Gaussian window over the characters,
/// and a mixture density output that gets sampled into pen movements.
///
/// A port of `hw/engine.py`, which is itself checked against the original
/// TensorFlow graph; `SelfTest` replays that same trace through this code.
final class Engine {

    /// How far the window must move for the pen to count as making progress.
    static let stallProgress: Float = 0.5
    /// Timestep budget as a multiple of the chosen style's own writing pace.
    static let stepBudget: Float = 1.45
    /// How far the reading may fall back before the line is written off. When
    /// the model loses the thread it returns to characters it has already
    /// written and what comes out of the pen from there on is gibberish. Lines
    /// that came out right never fell back by more than 2.9 characters.
    static let lostBacktrack: Float = 3
    /// Pace assumed when writing without a style, in timesteps per character.
    static let defaultPace: Float = 30
    static let attempts = 5

    struct State {
        var h1: [Float], c1: [Float]
        var h2: [Float], c2: [Float]
        var h3: [Float], c3: [Float]
        var alpha: [Float], beta: [Float], kappa: [Float]
        var window: [Float]
        var phi: [Float]
    }

    /// The characters of one line, as the window sees them.
    struct Context {
        var codes: [Int]
        var length: Int
        var width: Int { codes.count }
    }

    private let weights: [String: Container.Tensor]
    let lstmSize: Int
    let attentionMixtures: Int
    let outputMixtures: Int
    let alphabetSize: Int

    init(url: URL = Resources.url("model")) throws {
        let container = try Container(contentsOf: url)
        weights = container.tensors
        lstmSize = try container.tensor("lstm1_bias").count / 4
        attentionMixtures = try container.tensor("attention_bias").count / 3
        outputMixtures = (try container.tensor("gmm_bias").count - 1) / 6
        alphabetSize = try container.tensor("lstm1_kernel").rows - lstmSize - 3
    }

    // MARK: - the cell

    /// y = x . kernel + bias, with the kernel stored row major as [inputs, outputs].
    private func affine(_ x: [Float], _ kernel: String, _ bias: String) -> [Float] {
        let k = weights[kernel]!, b = weights[bias]!
        var y = b.values
        cblas_sgemv(CblasRowMajor, CblasTrans,
                    Int32(k.rows), Int32(k.columns), 1,
                    k.values, Int32(k.columns), x, 1, 1, &y, 1)
        return y
    }

    private func lstm(_ layer: Int, _ input: [Float], _ h: inout [Float], _ c: inout [Float]) {
        var joined = input
        joined.append(contentsOf: h)
        let z = affine(joined, "lstm\(layer)_kernel", "lstm\(layer)_bias")
        let n = lstmSize
        for k in 0..<n {
            let input = sigmoid(z[k])
            let candidate = tanhf(z[n + k])
            let forget = sigmoid(z[2 * n + k] + 1)      // forget_bias = 1
            let output = sigmoid(z[3 * n + k])
            c[k] = forget * c[k] + input * candidate
            h[k] = output * tanhf(c[k])
        }
    }

    func zeroState(width: Int) -> State {
        let zeros = { (n: Int) in [Float](repeating: 0, count: n) }
        return State(h1: zeros(lstmSize), c1: zeros(lstmSize),
                     h2: zeros(lstmSize), c2: zeros(lstmSize),
                     h3: zeros(lstmSize), c3: zeros(lstmSize),
                     alpha: zeros(attentionMixtures), beta: zeros(attentionMixtures),
                     kappa: zeros(attentionMixtures),
                     window: zeros(alphabetSize), phi: zeros(width))
    }

    func step(input x: [Float], state: inout State, context: Context) {
        var first = state.window
        first.append(contentsOf: x)
        lstm(1, first, &state.h1, &state.c1)

        var attention = state.window
        attention.append(contentsOf: x)
        attention.append(contentsOf: state.h1)
        let raw = affine(attention, "attention_kernel", "attention_bias")

        let m = attentionMixtures
        for k in 0..<m {
            state.alpha[k] = softplus(raw[k])
            state.beta[k] = max(softplus(raw[m + k]), 0.01)
            state.kappa[k] += softplus(raw[2 * m + k]) / 25
        }
        for u in 0..<context.width {
            var sum: Float = 0
            let position = Float(u)
            for k in 0..<m {
                let offset = state.kappa[k] - position
                sum += state.alpha[k] * expf(-offset * offset / state.beta[k])
            }
            state.phi[u] = sum
        }
        // the window is the attention-weighted sum of one-hot characters, which
        // is just phi accumulated into the bin of each character it can see
        for a in 0..<alphabetSize { state.window[a] = 0 }
        for u in 0..<context.length { state.window[context.codes[u]] += state.phi[u] }

        var second = x
        second.append(contentsOf: state.h1)
        second.append(contentsOf: state.window)
        lstm(2, second, &state.h2, &state.c2)

        var third = x
        third.append(contentsOf: state.h2)
        third.append(contentsOf: state.window)
        lstm(3, third, &state.h3, &state.c3)
    }

    // MARK: - output

    func mixtureParameters(_ state: State) -> [Float] {
        affine(state.h3, "gmm_kernel", "gmm_bias")
    }

    /// One sampled pen movement: how far to move, and whether the pen lifts.
    func sample(_ state: State, bias: Float, using rng: inout RandomSource) -> SIMD3<Float> {
        let p = mixtureParameters(state)
        let n = outputMixtures

        var weightsʹ = [Float](repeating: 0, count: n)
        var largest = -Float.greatestFiniteMagnitude
        for k in 0..<n { largest = max(largest, p[k] * (1 + bias)) }
        var total: Float = 0
        for k in 0..<n {
            let value = expf(p[k] * (1 + bias) - largest)
            weightsʹ[k] = value
            total += value
        }
        var kept: Float = 0
        for k in 0..<n {
            weightsʹ[k] /= total
            if weightsʹ[k] < 0.01 { weightsʹ[k] = 0 } else { kept += weightsʹ[k] }
        }

        var target = rng.uniform() * kept
        var chosen = n - 1
        for k in 0..<n {
            target -= weightsʹ[k]
            if target < 0 { chosen = k; break }
        }

        let sigma1 = max(expf(p[n + chosen] - bias), 1e-4)
        let sigma2 = max(expf(p[2 * n + chosen] - bias), 1e-4)
        let rho = min(max(tanhf(p[3 * n + chosen]), -1 + 1e-8), 1 - 1e-8)
        let mu1 = p[4 * n + chosen], mu2 = p[5 * n + chosen]

        let z1 = rng.normal(), z2 = rng.normal()
        let dx = mu1 + sigma1 * z1
        let dy = mu2 + sigma2 * (rho * z1 + sqrtf(1 - rho * rho) * z2)

        var lift = min(max(sigmoid(p[6 * n]), 1e-8), 1 - 1e-8)
        if lift < 0.01 { lift = 0 }
        return SIMD3(dx, dy, rng.uniform() < lift ? 1 : 0)
    }

    // MARK: - runs

    /// Run the model over strokes it is given: the priming pass, and the path
    /// the self test replays.
    @discardableResult
    func teacherForce(strokes: [SIMD3<Float>], context: Context,
                      state: inout State) -> [[Float]] {
        var parameters: [[Float]] = []
        parameters.reserveCapacity(strokes.count)
        for point in strokes {
            step(input: [point.x, point.y, point.z], state: &state, context: context)
            parameters.append(mixtureParameters(state))
        }
        return parameters
    }

    struct Run {
        var points: [SIMD3<Float>]
        var finished: Bool
        var readAt: Float       // how far into the text the window got
    }

    /// Feed the model's own samples back in until the line is written.
    func freeRun(state: State, context: Context, bias: Float, limit: Int,
                 first: SIMD3<Float>, using rng: inout RandomSource,
                 progress: ((Double) -> Bool)? = nil) -> Run {
        var state = state
        var x = first
        var points: [SIMD3<Float>] = []
        points.reserveCapacity(limit)

        var readAt = -Float.greatestFiniteMagnitude
        var readMax = -Float.greatestFiniteMagnitude
        var lastProgress = 0

        for t in 0..<limit {
            step(input: [x.x, x.y, x.z], state: &state, context: context)
            x = sample(state, bias: bias, using: &rng)
            points.append(x)

            var weighted: Float = 0, total: Float = 0
            for k in 0..<attentionMixtures {
                weighted += state.alpha[k] * state.kappa[k]
                total += state.alpha[k]
            }
            let position = weighted / max(total, 1e-8)
            if position > readAt + Engine.stallProgress {
                readAt = position
                lastProgress = t
            }
            readMax = max(readMax, position)
            // gone back to text it has already written, with more still to go
            if position < readMax - Engine.lostBacktrack,
               position < Float(context.length - 2) {
                return Run(points: Array(points.prefix(lastProgress + 1)),
                           finished: false, readAt: readAt)
            }

            var peak = 0
            for u in 1..<context.width where state.phi[u] > state.phi[peak] { peak = u }
            let pastEnd = peak >= context.length
            let atEnd = peak >= context.length - 1 && x.z == 1
            if pastEnd || atEnd { return Run(points: points, finished: true, readAt: readAt) }

            if t % 25 == 0, let progress, !progress(Double(t + 1) / Double(limit)) {
                return Run(points: [], finished: true, readAt: readAt)   // cancelled
            }
        }
        return Run(points: Array(points.prefix(lastProgress + 1)), finished: false, readAt: readAt)
    }
}
