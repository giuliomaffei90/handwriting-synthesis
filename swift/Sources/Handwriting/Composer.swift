import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// What the window is showing and doing.
@MainActor
final class Composer: ObservableObject {

    @Published var text = "Everything is going to be alright."
    @Published var styleID = 9
    @Published var neatness = 1.0
    @Published var penWidth = 2.0
    @Published var ink = Color.black
    @Published private(set) var page: Drawing.Page?
    @Published private(set) var samplePage: Drawing.Page?
    @Published private(set) var progress: Double?
    @Published private(set) var notice = ""
    @Published private(set) var status = ""
    @Published private(set) var failure: String?

    private var generator: Generator?
    private var job: Task<Void, Never>?

    var isWriting: Bool { job != nil }
    var styleIDs: [Int] { generator?.styles.all.map(\.id) ?? [] }
    var inkColour: NSColor { NSColor(ink) }

    func load() {
        do {
            generator = try Generator()
            if let first = styleIDs.first, !styleIDs.contains(styleID) { styleID = first }
            showSample()
        } catch {
            failure = "The model could not be loaded: \(error)"
        }
    }

    /// The real handwriting sample the chosen style primes the model with.
    func showSample() {
        guard let style = generator?.styles[styleID] else { return }
        samplePage = Drawing.page(of: [style.strokes], margin: 6)
    }

    func write() {
        if let job {
            job.cancel()
            self.job = nil
            status = "Stopped."
            progress = nil
            return
        }
        guard let generator else { return }

        let (clean, dropped) = Alphabet.sanitize(text)
        let lines = Alphabet.wrap(clean)
        guard lines.contains(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty }) else {
            status = "Type some text first."
            return
        }
        notice = dropped.isEmpty ? ""
            : "Skipped characters the model was never taught: "
                + dropped.map(String.init).joined(separator: " ")

        let request = Generator.Request(lines: lines, style: styleID, bias: Float(neatness),
                                        seed: UInt64.random(in: 0..<UInt64.max))
        progress = 0
        status = "Writing..."

        job = Task { [weak self] in
            let written = await Task.detached(priority: .userInitiated) { () -> [[SIMD3<Float>]] in
                generator.write(request) { fraction in
                    if Task.isCancelled { return false }
                    Task { @MainActor [weak self] in self?.progress = fraction }
                    return true
                }
            }.value

            guard let self, !Task.isCancelled else { return }
            let page = Drawing.page(of: written)
            self.page = page.isEmpty ? nil : page
            self.status = page.isEmpty ? "Nothing came out - try again."
                : "\(page.strokes.count) strokes, \(written.reduce(0) { $0 + $1.count }) points"
            self.progress = nil
            self.job = nil
        }
    }

    // MARK: - saving

    func savePNG() { save(name: "handwriting.png", type: .png) }
    func saveSVG() { save(name: "handwriting.svg", type: .svg) }

    private func save(name: String, type: UTType) {
        guard let page else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = name
        panel.allowedContentTypes = [type]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            if type == .svg {
                let svg = Export.svg(page, penWidth: penWidth, colour: inkColour)
                try svg.write(to: url, atomically: true, encoding: .utf8)
            } else if let data = Export.png(page, penWidth: penWidth, colour: inkColour) {
                try data.write(to: url)
            }
            status = "Saved \(url.lastPathComponent)"
        } catch {
            failure = "Could not save: \(error.localizedDescription)"
        }
    }
}
