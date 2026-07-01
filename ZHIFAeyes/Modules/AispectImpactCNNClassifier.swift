import Foundation

struct AispectImpactFrame {
    let delta: Double
    let x: Double
    let y: Double
    let z: Double
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

    private struct TensorResource: Decodable {
        let shape: [Int]
        let values: [Double]
    }

    private struct ScalerResource: Decodable {
        let center: [Double]
        let scale: [Double]
    }

    private struct Model {
        let feature0Weight: [Double]
        let feature0Bias: [Double]
        let bn0Weight: [Double]
        let bn0Bias: [Double]
        let bn0Mean: [Double]
        let bn0Var: [Double]

        let feature4Weight: [Double]
        let feature4Bias: [Double]
        let bn1Weight: [Double]
        let bn1Bias: [Double]
        let bn1Mean: [Double]
        let bn1Var: [Double]

        let feature9Weight: [Double]
        let feature9Bias: [Double]
        let bn2Weight: [Double]
        let bn2Bias: [Double]
        let bn2Mean: [Double]
        let bn2Var: [Double]

        let classifier0Weight: [Double]
        let classifier0Bias: [Double]
        let classifier3Weight: [Double]
        let classifier3Bias: [Double]

        let center: [Double]
        let scale: [Double]
    }

    private let epsilon: Double = 1e-5
    private let bundle: Bundle
    private lazy var model: Model? = loadModel()

    init(bundle: Bundle = .main) {
        self.bundle = bundle
    }

    var isReady: Bool {
        model != nil
    }

    func predict(frames: [AispectImpactFrame]) -> AispectImpactPrediction? {
        guard frames.count == 21 else { return nil }
        guard let model else { return nil }

        let normalized = normalize(frames: frames, center: model.center, scale: model.scale)
        var x = transposeToChannels(frames: normalized)
        x = conv1d(input: x, weight: model.feature0Weight, bias: model.feature0Bias, outChannels: 32, inChannels: 4, length: 21)
        x = relu(batchNorm(input: x, gamma: model.bn0Weight, beta: model.bn0Bias, mean: model.bn0Mean, variance: model.bn0Var))

        x = conv1d(input: x, weight: model.feature4Weight, bias: model.feature4Bias, outChannels: 64, inChannels: 32, length: 21)
        x = relu(batchNorm(input: x, gamma: model.bn1Weight, beta: model.bn1Bias, mean: model.bn1Mean, variance: model.bn1Var))
        x = maxPool1d2(input: x)

        x = conv1d(input: x, weight: model.feature9Weight, bias: model.feature9Bias, outChannels: 128, inChannels: 64, length: 10)
        x = relu(batchNorm(input: x, gamma: model.bn2Weight, beta: model.bn2Bias, mean: model.bn2Mean, variance: model.bn2Var))
        x = maxPool1d2(input: x)

        let flat = flatten(input: x)
        let hidden = relu(linear(input: flat, weight: model.classifier0Weight, bias: model.classifier0Bias, outFeatures: 64, inFeatures: 640))
        let logits = linear(input: hidden, weight: model.classifier3Weight, bias: model.classifier3Bias, outFeatures: 2, inFeatures: 64)
        let probabilities = softmax(logits)

        guard probabilities.count == 2 else { return nil }
        return AispectImpactPrediction(heavyProbability: probabilities[0], lightProbability: probabilities[1])
    }

    private func loadModel() -> Model? {
        guard let weightsURL = bundle.url(forResource: "AispectImpactCNNWeights", withExtension: "json"),
              let scalerURL = bundle.url(forResource: "AispectImpactCNNScaler", withExtension: "json") else {
            return nil
        }

        let decoder = JSONDecoder()

        do {
            let weightsData = try Data(contentsOf: weightsURL)
            let scalerData = try Data(contentsOf: scalerURL)
            let weights = try decoder.decode([String: TensorResource].self, from: weightsData)
            let scaler = try decoder.decode(ScalerResource.self, from: scalerData)

            guard let feature0Weight = weights["features.0.weight"]?.values,
                  let feature0Bias = weights["features.0.bias"]?.values,
                  let bn0Weight = weights["features.1.weight"]?.values,
                  let bn0Bias = weights["features.1.bias"]?.values,
                  let bn0Mean = weights["features.1.running_mean"]?.values,
                  let bn0Var = weights["features.1.running_var"]?.values,
                  let feature4Weight = weights["features.4.weight"]?.values,
                  let feature4Bias = weights["features.4.bias"]?.values,
                  let bn1Weight = weights["features.5.weight"]?.values,
                  let bn1Bias = weights["features.5.bias"]?.values,
                  let bn1Mean = weights["features.5.running_mean"]?.values,
                  let bn1Var = weights["features.5.running_var"]?.values,
                  let feature9Weight = weights["features.9.weight"]?.values,
                  let feature9Bias = weights["features.9.bias"]?.values,
                  let bn2Weight = weights["features.10.weight"]?.values,
                  let bn2Bias = weights["features.10.bias"]?.values,
                  let bn2Mean = weights["features.10.running_mean"]?.values,
                  let bn2Var = weights["features.10.running_var"]?.values,
                  let classifier0Weight = weights["classifier.0.weight"]?.values,
                  let classifier0Bias = weights["classifier.0.bias"]?.values,
                  let classifier3Weight = weights["classifier.3.weight"]?.values,
                  let classifier3Bias = weights["classifier.3.bias"]?.values,
                  scaler.center.count == 4,
                  scaler.scale.count == 4 else {
                return nil
            }

            return Model(
                feature0Weight: feature0Weight,
                feature0Bias: feature0Bias,
                bn0Weight: bn0Weight,
                bn0Bias: bn0Bias,
                bn0Mean: bn0Mean,
                bn0Var: bn0Var,
                feature4Weight: feature4Weight,
                feature4Bias: feature4Bias,
                bn1Weight: bn1Weight,
                bn1Bias: bn1Bias,
                bn1Mean: bn1Mean,
                bn1Var: bn1Var,
                feature9Weight: feature9Weight,
                feature9Bias: feature9Bias,
                bn2Weight: bn2Weight,
                bn2Bias: bn2Bias,
                bn2Mean: bn2Mean,
                bn2Var: bn2Var,
                classifier0Weight: classifier0Weight,
                classifier0Bias: classifier0Bias,
                classifier3Weight: classifier3Weight,
                classifier3Bias: classifier3Bias,
                center: scaler.center,
                scale: scaler.scale.map { abs($0) > 1e-9 ? $0 : 1.0 }
            )
        } catch {
            return nil
        }
    }

    private func normalize(frames: [AispectImpactFrame], center: [Double], scale: [Double]) -> [[Double]] {
        frames.map { frame in
            [
                (frame.delta - center[0]) / scale[0],
                (frame.x - center[1]) / scale[1],
                (frame.y - center[2]) / scale[2],
                (frame.z - center[3]) / scale[3]
            ]
        }
    }

    private func transposeToChannels(frames: [[Double]]) -> [[Double]] {
        var channels = Array(repeating: Array(repeating: 0.0, count: 21), count: 4)
        for index in 0..<21 {
            for channel in 0..<4 {
                channels[channel][index] = frames[index][channel]
            }
        }
        return channels
    }

    private func conv1d(input: [[Double]], weight: [Double], bias: [Double], outChannels: Int, inChannels: Int, length: Int) -> [[Double]] {
        let kernelSize = 3
        let padding = 1
        var output = Array(repeating: Array(repeating: 0.0, count: length), count: outChannels)
        for outChannel in 0..<outChannels {
            for position in 0..<length {
                var sum = bias[outChannel]
                for inChannel in 0..<inChannels {
                    for kernel in 0..<kernelSize {
                        let source = position + kernel - padding
                        guard source >= 0, source < length else { continue }
                        let weightIndex = ((outChannel * inChannels + inChannel) * kernelSize) + kernel
                        sum += input[inChannel][source] * weight[weightIndex]
                    }
                }
                output[outChannel][position] = sum
            }
        }
        return output
    }

    private func batchNorm(input: [[Double]], gamma: [Double], beta: [Double], mean: [Double], variance: [Double]) -> [[Double]] {
        var output = input
        for channel in 0..<input.count {
            let denominator = (variance[channel] + epsilon).squareRoot()
            for index in 0..<input[channel].count {
                let normalized = (input[channel][index] - mean[channel]) / denominator
                output[channel][index] = normalized * gamma[channel] + beta[channel]
            }
        }
        return output
    }

    private func relu(_ input: [[Double]]) -> [[Double]] {
        input.map { row in
            row.map { max(0.0, $0) }
        }
    }

    private func relu(_ input: [Double]) -> [Double] {
        input.map { max(0.0, $0) }
    }

    private func maxPool1d2(input: [[Double]]) -> [[Double]] {
        input.map { row in
            guard row.count >= 2 else { return [] }
            var pooled: [Double] = []
            var index = 0
            while index + 1 < row.count {
                pooled.append(max(row[index], row[index + 1]))
                index += 2
            }
            return pooled
        }
    }

    private func flatten(input: [[Double]]) -> [Double] {
        input.flatMap { $0 }
    }

    private func linear(input: [Double], weight: [Double], bias: [Double], outFeatures: Int, inFeatures: Int) -> [Double] {
        var output = Array(repeating: 0.0, count: outFeatures)
        for outFeature in 0..<outFeatures {
            var sum = bias[outFeature]
            let base = outFeature * inFeatures
            for inFeature in 0..<inFeatures {
                sum += input[inFeature] * weight[base + inFeature]
            }
            output[outFeature] = sum
        }
        return output
    }

    private func softmax(_ input: [Double]) -> [Double] {
        guard let maxValue = input.max() else { return [] }
        let exponentials = input.map { Foundation.exp($0 - maxValue) }
        let total = exponentials.reduce(0, +)
        guard total > 0 else { return Array(repeating: 0.5, count: input.count) }
        return exponentials.map { $0 / total }
    }
}
