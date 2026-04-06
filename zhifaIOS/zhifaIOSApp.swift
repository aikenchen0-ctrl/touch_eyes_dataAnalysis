//  中文注释：应用入口与场景配置。

import SwiftUI

@main
struct zhifaIOSApp: App {
    var body: some Scene {
        WindowGroup {
            ZStack {
                Color.black.ignoresSafeArea()
                ContentView()
            }
        }
    }
}
