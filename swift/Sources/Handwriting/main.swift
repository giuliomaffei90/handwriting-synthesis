import AppKit
import Foundation

let arguments = CommandLine.arguments

if arguments.contains("--selftest") {
    do { try SelfTest.run() } catch {
        FileHandle.standardError.write(Data("SELFTEST FAILED: \(error)\n".utf8))
        exit(1)
    }
    exit(0)
}

/// Writing from the command line, which is how the app is checked without
/// clicking through it: --write "text" [--style 9] [--bias 1.0] [--seed 1] --out page
if let index = arguments.firstIndex(of: "--write"), index + 1 < arguments.count {
    func option(_ name: String) -> String? {
        guard let i = arguments.firstIndex(of: name), i + 1 < arguments.count else { return nil }
        return arguments[i + 1]
    }
    let text = arguments[index + 1]
    let style = option("--style").flatMap(Int.init)
    let bias = option("--bias").flatMap(Float.init) ?? 1
    let seed = option("--seed").flatMap(UInt64.init) ?? 1
    let out = option("--out") ?? "page"

    do {
        let started = Date()
        let generator = try Generator()
        let lines = Alphabet.wrap(Alphabet.sanitize(text).text)
        let written = generator.write(.init(lines: lines, style: style, bias: bias, seed: seed))
        let page = Drawing.page(of: written)
        let elapsed = Date().timeIntervalSince(started)

        try Export.svg(page, penWidth: 2, colour: .black).write(
            toFile: out + ".svg", atomically: true, encoding: .utf8)
        if let png = Export.png(page, penWidth: 2, colour: .black) {
            try png.write(to: URL(fileURLWithPath: out + ".png"))
        }
        print(String(format: "%d lines, %d strokes, %d points in %.1fs -> %@.svg/.png",
                     lines.count, page.strokes.count,
                     written.reduce(0) { $0 + $1.count }, elapsed, out))
    } catch {
        FileHandle.standardError.write(Data("failed: \(error)\n".utf8))
        exit(1)
    }
    exit(0)
}

HandwritingApp.main()
