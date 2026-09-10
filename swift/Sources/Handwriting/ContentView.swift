import SwiftUI

struct ContentView: View {
    @StateObject private var composer = Composer()

    var body: some View {
        VStack(spacing: 12) {
            controls
            hint
            sample

            // what you type and what it becomes, side by side like two pages
            HStack(spacing: 14) {
                editor
                preview
            }
            .frame(maxHeight: .infinity)

            actions
        }
        .padding(18)
        .frame(minWidth: 760, minHeight: 600)
        .onAppear { composer.load() }
        .alert("Something went wrong", isPresented: .constant(composer.failure != nil)) {
            Button("OK") { }
        } message: {
            Text(composer.failure ?? "")
        }
    }

    private var controls: some View {
        HStack(spacing: 22) {
            Picker("Style", selection: $composer.styleID) {
                ForEach(composer.styles, id: \.id) { Text($0.name).tag($0.id) }
            }
            .frame(width: 190)
            .onChange(of: composer.styleID) { _ in composer.showSample() }

            LabelledSlider(title: "Neatness", value: $composer.neatness,
                           range: 0.3...2, format: "%.2f")
            LabelledSlider(title: "Pen", value: $composer.penWidth,
                           range: 0.7...6, format: "%.1f")
            ColorPicker("", selection: $composer.ink, supportsOpacity: false)
                .labelsHidden()
            Spacer(minLength: 0)
        }
    }

    private var hint: some View {
        Text("Neatness steadies the hand: low wanders and scrawls, high writes carefully. "
             + "Images are saved with a transparent background.")
            .font(.caption)
            .foregroundStyle(.tertiary)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// The same sentence, written in the chosen style.
    private var sample: some View {
        PageView(page: composer.samplePage, ink: .secondary, penWidth: 2.4, maximumZoom: 1,
                 padding: 4)
            .frame(height: 66)
            .frame(maxWidth: .infinity)
            .background(RoundedRectangle(cornerRadius: 6).fill(.quaternary.opacity(0.35)))
            .overlay(alignment: .leading) {
                Text("style")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .padding(.leading, 8)
            }
    }

    private var editor: some View {
        VStack(alignment: .leading, spacing: 6) {
            TextEditor(text: $composer.text)
                .font(.system(size: 14))
                .scrollContentBackground(.hidden)
                .padding(10)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(RoundedRectangle(cornerRadius: 10).fill(.quaternary.opacity(0.5)))

            if !composer.notice.isEmpty {
                Label(composer.notice, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// A sheet of paper: the writing starts at the top left, as a letter would.
    private var preview: some View {
        PageView(page: composer.page, ink: composer.ink, penWidth: composer.penWidth,
                 maximumZoom: 2, padding: 14, anchor: .topLeading)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(RoundedRectangle(cornerRadius: 10).fill(.white))
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.quaternary))
            .overlay {
                if composer.page == nil && !composer.isWriting {
                    Text("Your handwriting appears here")
                        .font(.callout)
                        .foregroundStyle(Color(white: 0.68))
                }
            }
    }

    private var actions: some View {
        HStack(spacing: 12) {
            Button(composer.isWriting ? "Stop" : "Write") { composer.write() }
                .keyboardShortcut(.return, modifiers: .command)
                .buttonStyle(.borderedProminent)

            if let progress = composer.progress {
                ProgressView(value: progress).frame(width: 130)
            }
            Text(composer.status)
                .font(.caption)
                .foregroundStyle(.secondary)

            Spacer()
            Button("Save PNG") { composer.savePNG() }.disabled(composer.page == nil)
            Button("Save SVG") { composer.saveSVG() }.disabled(composer.page == nil)
        }
    }
}

/// A slider that shows what it is set to.
private struct LabelledSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let format: String

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text("\(title) \(String(format: format, value))")
                .font(.caption)
                .foregroundStyle(.secondary)
            Slider(value: $value, in: range).frame(width: 130)
        }
    }
}

/// Draws a page, scaled to fit and never blown up past `maximumZoom`.
struct PageView: View {
    let page: Drawing.Page?
    let ink: Color
    let penWidth: Double
    let maximumZoom: Double
    var padding: CGFloat = 8
    /// Where the page sits when it is smaller than the view.
    var anchor: UnitPoint = .center

    var body: some View {
        Canvas { context, size in
            guard let page, !page.isEmpty else { return }
            let zoom = min((size.width - 2 * padding) / page.size.width,
                           (size.height - 2 * padding) / page.size.height,
                           maximumZoom)
            let spareX = size.width - 2 * padding - page.size.width * zoom
            let spareY = size.height - 2 * padding - page.size.height * zoom
            context.translateBy(x: padding + spareX * anchor.x, y: padding + spareY * anchor.y)
            context.scaleBy(x: zoom, y: zoom)
            let stroke = StrokeStyle(lineWidth: penWidth, lineCap: .round, lineJoin: .round)
            for line in page.strokes {
                context.stroke(Path(Drawing.path(for: line)), with: .color(ink), style: stroke)
            }
        }
    }
}
