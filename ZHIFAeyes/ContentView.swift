//  中文注释：SwiftUI 入口壳，承载 UIKit 主界面。

import SwiftUI
import UIKit

struct ContentView: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> UIViewController {
        RootViewController()
    }

    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
        // 中文注释：状态更新预留。
    }
}

#Preview {
    ContentView()
}
