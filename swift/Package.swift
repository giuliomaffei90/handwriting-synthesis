// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Handwriting",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(name: "Handwriting", path: "Sources/Handwriting")
    ]
)
