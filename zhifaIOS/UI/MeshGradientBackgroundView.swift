import UIKit
import SwiftUI
import Combine

final class MeshGradientBackgroundView: UIView {
    @available(iOS 18.0, *)
    private var meshDriver: MeshGradientDriver?
    private var hostingController: UIHostingController<AnyView>?
    private let fallbackLayer = CAGradientLayer()

    override init(frame: CGRect) {
        super.init(frame: frame)
        setupView()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupView()
    }

    private func setupView() {
        backgroundColor = .black
        isUserInteractionEnabled = false

        if #available(iOS 18.0, *) {
            let driver = MeshGradientDriver()
            meshDriver = driver
            let rootView = AnyView(MeshGradientScene(driver: driver))
            let hosting = UIHostingController(rootView: rootView)
            hosting.view.backgroundColor = .black
            hosting.view.translatesAutoresizingMaskIntoConstraints = false
            addSubview(hosting.view)
            NSLayoutConstraint.activate([
                hosting.view.leadingAnchor.constraint(equalTo: leadingAnchor),
                hosting.view.trailingAnchor.constraint(equalTo: trailingAnchor),
                hosting.view.topAnchor.constraint(equalTo: topAnchor),
                hosting.view.bottomAnchor.constraint(equalTo: bottomAnchor)
            ])
            hostingController = hosting
        } else {
            // 中文注释：系统不支持 MeshGradient 时使用静态渐变兜底
            fallbackLayer.colors = [
                UIColor.black.cgColor,
                UIColor.black.cgColor,
                UIColor.black.cgColor
            ]
            fallbackLayer.startPoint = CGPoint(x: 0.0, y: 0.0)
            fallbackLayer.endPoint = CGPoint(x: 1.0, y: 1.0)
            layer.addSublayer(fallbackLayer)
        }
    }

    func updateTouch(point: CGPoint, pressure: CGFloat) {
        guard #available(iOS 18.0, *), let driver = meshDriver else { return }
        let width = max(bounds.width, 1)
        let height = max(bounds.height, 1)
        let normalized = CGPoint(
            x: min(max(point.x / width, 0), 1),
            y: min(max(point.y / height, 0), 1)
        )
        driver.update(point: normalized, pressure: min(max(pressure, 0), 1))
    }

    func resetPressure() {
        guard #available(iOS 18.0, *), let driver = meshDriver else { return }
        driver.update(point: CGPoint(x: 0.5, y: 0.5), pressure: 0)
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        fallbackLayer.frame = bounds
    }
}

@available(iOS 18.0, *)
private final class MeshGradientDriver: ObservableObject {
    @Published var touchPoint: CGPoint = CGPoint(x: 0.5, y: 0.5)
    @Published var pressure: CGFloat = 0

    func update(point: CGPoint, pressure: CGFloat) {
        touchPoint = point
        self.pressure = pressure
    }
}

@available(iOS 18.0, *)
private struct MeshGradientScene: View {
    @ObservedObject var driver: MeshGradientDriver

    var body: some View {
        TimelineView(.animation) { timeline in
            let t = Float(timeline.date.timeIntervalSinceReferenceDate)
            let baseShiftA = 0.03 * sin(t * 0.4)
            let baseShiftB = 0.03 * cos(t * 0.35)
            let baseShiftC = 0.02 * sin(t * 0.2 + 1.2)
            let press = Float(driver.pressure)
            let pressStrength = 0.18 * press
            let px = Float(driver.touchPoint.x)
            let py = Float(driver.touchPoint.y)

            let basePoints: [SIMD2<Float>] = [
                SIMD2(0.0, 0.0),
                SIMD2(0.5 + baseShiftA, 0.0),
                SIMD2(1.0, 0.0),
                SIMD2(0.0, 0.5 + baseShiftB),
                SIMD2(0.5 + baseShiftC, 0.5 - baseShiftA),
                SIMD2(1.0, 0.5 + baseShiftA),
                SIMD2(0.0, 1.0),
                SIMD2(0.5 - baseShiftB, 1.0),
                SIMD2(1.0, 1.0)
            ]
            let strengths: [Float] = [0.12, 0.14, 0.1, 0.14, 0.2, 0.14, 0.1, 0.14, 0.12]
            let warpedPoints = zip(basePoints, strengths).map { item in
                Self.warp(base: item.0, targetX: px, targetY: py, strength: pressStrength, factor: item.1)
            }

            MeshGradient(
                width: 3,
                height: 3,
                points: warpedPoints,
                colors: [
                    Color.black,
                    Color.black,
                    Color.black,
                    Color.black,
                    Color.black,
                    Color.black,
                    Color.black,
                    Color.black,
                    Color.black
                ]
            )
            .ignoresSafeArea()
        }
    }

    private static func clamp01(_ value: Float) -> Float {
        min(max(value, 0), 1)
    }

    private static func warp(base: SIMD2<Float>, targetX: Float, targetY: Float, strength: Float, factor: Float) -> SIMD2<Float> {
        let dx = targetX - base.x
        let dy = targetY - base.y
        return SIMD2(
            clamp01(base.x + dx * factor * strength),
            clamp01(base.y + dy * factor * strength)
        )
    }
}
