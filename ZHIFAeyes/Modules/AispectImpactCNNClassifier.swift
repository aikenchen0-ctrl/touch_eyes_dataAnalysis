import Foundation

struct AispectImpactFrame {
    let x: Double
    let y: Double
    let z: Double
    let xNorm: Double
    let yNorm: Double
    let rotationRateX: Double
    let rotationRateY: Double
    let rotationRateZ: Double
    let gravityX: Double
    let gravityY: Double
    let gravityZ: Double

    init(
        x: Double,
        y: Double,
        z: Double,
        xNorm: Double = 0,
        yNorm: Double = 0,
        rotationRateX: Double = 0,
        rotationRateY: Double = 0,
        rotationRateZ: Double = 0,
        gravityX: Double = 0,
        gravityY: Double = 0,
        gravityZ: Double = 0
    ) {
        self.x = x
        self.y = y
        self.z = z
        self.xNorm = xNorm
        self.yNorm = yNorm
        self.rotationRateX = rotationRateX
        self.rotationRateY = rotationRateY
        self.rotationRateZ = rotationRateZ
        self.gravityX = gravityX
        self.gravityY = gravityY
        self.gravityZ = gravityZ
    }
}

struct AispectImpactPrediction {
    let heavyProbability: Double
    let lightProbability: Double

    var isHeavy: Bool {
        heavyProbability >= lightProbability
    }
}

final class AispectImpactCNNClassifier {
    static let shared = AispectImpactCNNClassifier()

    // 中文注释：旧模型资源保留在工程中，但新标准模型尚未训练，端侧不加载旧模型。
    var isReady: Bool {
        false
    }

    func predict(frames: [AispectImpactFrame]) -> AispectImpactPrediction? {
        nil
    }
}
