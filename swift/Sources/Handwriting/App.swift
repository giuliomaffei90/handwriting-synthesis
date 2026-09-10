import SwiftUI

struct HandwritingApp: App {
    var body: some Scene {
        WindowGroup("Handwriting") {
            ContentView()
        }
        .defaultSize(width: 1100, height: 760)
        .windowResizability(.contentMinSize)
        .commands { CommandGroup(replacing: .newItem) { } }
    }
}
