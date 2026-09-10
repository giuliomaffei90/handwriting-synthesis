import Foundation

/// The 73 characters the network was taught, and getting arbitrary text onto them.
enum Alphabet {

    static let characters: [Character] = Array(
        "\0 !\"#'(),-.0123456789:;?ABCDEFGHIJKLMNOPRSTUVWYabcdefghijklmnopqrstuvwxyz")

    static let index: [Character: Int] = {
        var map: [Character: Int] = [:]
        for (i, c) in characters.enumerated() { map[c] = i }
        return map
    }()

    /// Characters the model never saw, mapped to the closest thing it did see.
    static let substitutions: [Character: String] = [
        "\u{2019}": "'", "\u{2018}": "'", "\u{201C}": "\"", "\u{201D}": "\"",
        "\u{2013}": "-", "\u{2014}": "-", "\u{2026}": "...", "\u{00B7}": ".",
        "\u{00A0}": " ", "\t": " ",
        "Q": "q", "X": "x", "Z": "z",   // missing uppercase in the training set
        "&": "and", "/": "-", "*": ".", "_": "-", "\u{20AC}": "EUR", "$": "S",
        "%": ".", "=": "-", "+": "t", "@": "a", "\u{00DF}": "ss",
    ]

    static let maximumLineLength = 75

    /// Grave and acute, the accents an Italian keyboard actually types.
    static let accents: Set<Unicode.Scalar> = ["\u{0300}", "\u{0301}"]

    /// Map text onto the alphabet, reporting whatever had to be dropped.
    ///
    /// An accented letter becomes the plain letter and an apostrophe, the way
    /// it is typed on a machine that has no accents: e' for e-grave. Other
    /// marks - cedillas, tildes, diaereses - are simply dropped, since an
    /// apostrophe would be wrong there.
    static func sanitize(_ text: String) -> (text: String, dropped: [Character]) {
        var out = ""
        var dropped: Set<Character> = []
        for character in text {
            if index[character] != nil || character == "\n" {
                out.append(character)
                continue
            }
            var replacement = substitutions[character]
            if replacement == nil {
                let scalars = Array(String(character).decomposedStringWithCanonicalMapping
                    .unicodeScalars)
                let marks = scalars.filter { $0.properties.generalCategory == .nonspacingMark }
                let base = String(String.UnicodeScalarView(
                    scalars.filter { $0.properties.generalCategory != .nonspacingMark }))
                if !base.isEmpty && base.allSatisfy({ index[$0] != nil }) {
                    replacement = marks.contains(where: { accents.contains($0) })
                        ? base + "'" : base
                } else {
                    // last resort for the likes of a ligature or a full-width digit
                    let folded = String(character)
                        .precomposedStringWithCompatibilityMapping
                        .folding(options: .diacriticInsensitive,
                                 locale: .init(identifier: "en_US"))
                    if !folded.isEmpty && folded.allSatisfy({ index[$0] != nil }) {
                        replacement = folded
                    }
                }
            }
            if let replacement, replacement.allSatisfy({ index[$0] != nil || $0 == "\n" }) {
                out.append(contentsOf: replacement)
            } else {
                dropped.insert(character)
            }
        }
        return (out, dropped.sorted())
    }

    /// Split into lines of at most `width` characters, breaking on spaces.
    static func wrap(_ text: String, width: Int = maximumLineLength) -> [String] {
        var lines: [String] = []
        for paragraph in text.components(separatedBy: "\n") {
            var current = ""
            for word in paragraph.components(separatedBy: " ") {
                var word = word
                while word.count > width {
                    if !current.isEmpty { lines.append(current); current = "" }
                    lines.append(String(word.prefix(width)))
                    word = String(word.dropFirst(width))
                }
                let candidate = current.isEmpty ? word : current + " " + word
                if candidate.count > width {
                    lines.append(current)
                    current = word
                } else {
                    current = candidate
                }
            }
            lines.append(current)
        }
        return lines
    }

    /// Character indices, with the terminator the model expects appended.
    static func encode(_ text: String) -> [Int] {
        text.map { index[$0] ?? 0 } + [0]
    }
}

/// Small reproducible generator - splitmix64 with Box-Muller on top.
struct RandomSource {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    mutating func uniform() -> Float { Float(next() >> 40) * 0x1p-24 }

    /// Standard normal.
    mutating func normal() -> Float {
        let u = max(uniform(), Float.leastNormalMagnitude)
        return sqrtf(-2 * logf(u)) * cosf(2 * .pi * uniform())
    }
}
