import Foundation

/// Reader for the little container that `tools/make_swift_resources.py` writes:
/// the model weights, the styles and the TensorFlow trace used to check the port.
///
///     "HWB1" | count | entries...
///     entry: nameLen | name | type | ndim | dims | payload
///     type 0 = float32 little-endian row major, type 1 = utf8 text
struct Container {

    struct Tensor {
        var shape: [Int]
        var values: [Float]

        var count: Int { values.count }
        var rows: Int { shape.first ?? 0 }
        var columns: Int { shape.count > 1 ? shape[1] : 1 }

        /// Row `index` of a 2-D tensor.
        func row(_ index: Int) -> ArraySlice<Float> {
            let width = columns
            return values[(index * width)..<((index + 1) * width)]
        }
    }

    private(set) var tensors: [String: Tensor] = [:]
    private(set) var strings: [String: String] = [:]

    init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        var offset = 0

        func read<T>(_ type: T.Type) throws -> T {
            let size = MemoryLayout<T>.size
            guard offset + size <= data.count else { throw Failure.truncated }
            defer { offset += size }
            return data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: T.self) }
        }
        func read(bytes count: Int) throws -> Data {
            guard offset + count <= data.count else { throw Failure.truncated }
            defer { offset += count }
            return data.subdata(in: offset..<(offset + count))
        }

        guard try read(bytes: 4) == Data("HWB1".utf8) else { throw Failure.badMagic }
        let count = Int(try read(UInt32.self))

        for _ in 0..<count {
            let nameLength = Int(try read(UInt32.self))
            let name = String(decoding: try read(bytes: nameLength), as: UTF8.self)
            let kind = try read(UInt8.self)
            let rank = Int(try read(UInt32.self))
            var shape: [Int] = []
            for _ in 0..<rank { shape.append(Int(try read(UInt32.self))) }

            if kind == 1 {
                strings[name] = String(decoding: try read(bytes: shape[0]), as: UTF8.self)
            } else {
                let elements = shape.reduce(1, *)
                let raw = try read(bytes: elements * 4)
                var values = [Float](repeating: 0, count: elements)
                _ = values.withUnsafeMutableBytes { raw.copyBytes(to: $0) }
                tensors[name] = Tensor(shape: shape, values: values)
            }
        }
    }

    enum Failure: Error { case badMagic, truncated, missing(String) }

    func tensor(_ name: String) throws -> Tensor {
        guard let t = tensors[name] else { throw Failure.missing(name) }
        return t
    }
}

/// Where the packed resources live: inside the app bundle once installed, or
/// next to the package while developing.
enum Resources {
    static func url(_ name: String) -> URL {
        if let bundled = Bundle.main.url(forResource: name, withExtension: "bin") {
            return bundled
        }
        var here = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
        for _ in 0..<5 {
            let candidate = here.appendingPathComponent("Resources/\(name).bin")
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
            here = here.deletingLastPathComponent()
        }
        return URL(fileURLWithPath: "swift/Resources/\(name).bin")
    }
}
