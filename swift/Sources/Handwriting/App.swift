import SwiftUI

struct HandwritingApp: App {
    var body: some Scene {
        WindowGroup("Handwriting") {
            ContentView()
        }
        .defaultSize(width: 820, height: 660)
        .windowResizability(.contentMinSize)
        .commands { CommandGroup(replacing: .newItem) { } }
    }
}
