package com.zhifaios.eyes.aispect;

import android.content.Context;
import android.content.res.AssetManager;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class AispectImpactCNNClassifier {
    private static final String ASSET_DIR = "aispect_impact_models";
    private static final String CATALOG_ASSET = ASSET_DIR + "/AispectImpactCNNModelCatalog.json";
    private static final String MODEL_G4 = "g4_pre5_post15";
    private static final String MODEL_G13 = "g13_time_f14_pre4_post9";
    private static final String MODEL_ANDROID_TOP_WINDOW = "android_patch21_no_confidence_pre5_post9";
    private static final String MODEL_FINGER_FOUR_CLASS_TOP = "oneplus_four_class_after_recollect_hard_nearfull_weighted_old13_multichannel";
    private static final double EPSILON = 1e-5;
    private static final int CONTACT_MAP_SIZE = 9;
    private static final double CONTACT_GRID_RADIUS_PX = 48.0;
    private static final double CONTACT_DEFAULT_SIGMA_PX = 12.0;

    public static final class ModelInfo {
        public final String id;
        public final String displayName;
        public final String weightsResourceName;
        public final String scalerResourceName;
        public final int inputChannels;
        public final int frameCount;
        public final int[] frameIndices;
        public final String[] featureNames;
        public final int classCount;
        public final String[] labelOrder;

        ModelInfo(
                String id,
                String displayName,
                String weightsResourceName,
                String scalerResourceName,
                int inputChannels,
                int frameCount,
                int[] frameIndices,
                String[] featureNames,
                int classCount,
                String[] labelOrder
        ) {
            this.id = id;
            this.displayName = displayName;
            this.weightsResourceName = weightsResourceName;
            this.scalerResourceName = scalerResourceName;
            this.inputChannels = inputChannels;
            this.frameCount = frameCount;
            this.frameIndices = frameIndices.clone();
            this.featureNames = featureNames.clone();
            this.classCount = classCount;
            this.labelOrder = labelOrder.clone();
        }
    }

    private final Context context;
    private final Map<String, Model> modelCache = new HashMap<>();
    private final Map<String, Boolean> modelLoadAttempts = new HashMap<>();
    private List<ModelInfo> catalogModels;

    public AispectImpactCNNClassifier(Context context) {
        this.context = context.getApplicationContext();
    }

    public List<ModelInfo> availableModels() {
        return Collections.unmodifiableList(loadCatalogModels());
    }

    public ModelInfo defaultModelInfo() {
        ModelInfo fingerFourClass = modelInfoForId(MODEL_FINGER_FOUR_CLASS_TOP);
        if (fingerFourClass != null) {
            return fingerFourClass;
        }
        ModelInfo androidTopWindow = modelInfoForId(MODEL_ANDROID_TOP_WINDOW);
        if (androidTopWindow != null) {
            return androidTopWindow;
        }
        ModelInfo enhanced = modelInfoForId(MODEL_G13);
        if (enhanced != null) {
            return enhanced;
        }
        List<ModelInfo> models = loadCatalogModels();
        if (!models.isEmpty()) {
            return models.get(0);
        }
        return fallbackModelInfo(MODEL_G4);
    }

    public ModelInfo modelInfoForId(String modelId) {
        if (modelId == null) {
            return null;
        }
        for (ModelInfo info : loadCatalogModels()) {
            if (modelId.equals(info.id)) {
                return info;
            }
        }
        if (MODEL_G4.equals(modelId) || MODEL_G13.equals(modelId)) {
            return fallbackModelInfo(modelId);
        }
        return null;
    }

    public AispectModels.ImpactPrediction predictBest(
            AispectSignalWindowBuilder.Window window,
            AispectDeviceCapabilityProfiler.Snapshot capability
    ) {
        return predictBest(window, capability, null, null);
    }

    public AispectModels.ImpactPrediction predictBest(
            AispectSignalWindowBuilder.Window window,
            AispectDeviceCapabilityProfiler.Snapshot capability,
            AispectContactPatch contactPatch,
            AispectTouchNormalizationProfile normalizationProfile
    ) {
        return predictBest(window, capability, contactPatch, normalizationProfile, null);
    }

    public AispectModels.ImpactPrediction predictBest(
            AispectSignalWindowBuilder.Window window,
            AispectDeviceCapabilityProfiler.Snapshot capability,
            AispectContactPatch contactPatch,
            AispectTouchNormalizationProfile normalizationProfile,
            AispectContactEncodingProfile contactEncodingProfile
    ) {
        if (capability != null && capability.level == AispectDeviceCapabilityProfiler.Level.A) {
            AispectModels.ImpactPrediction enhanced = predict(window, MODEL_G13, contactPatch, normalizationProfile, contactEncodingProfile);
            if (enhanced != null) {
                return enhanced;
            }
        }
        AispectModels.ImpactPrediction base = predict(window, MODEL_G4, contactPatch, normalizationProfile, contactEncodingProfile);
        if (base != null) {
            return base;
        }
        return predict(window, MODEL_G13, contactPatch, normalizationProfile, contactEncodingProfile);
    }

    public AispectModels.ImpactPrediction predict(AispectSignalWindowBuilder.Window window, String modelId) {
        return predict(window, modelId, null, null);
    }

    public AispectModels.ImpactPrediction predict(
            AispectSignalWindowBuilder.Window window,
            String modelId,
            AispectContactPatch contactPatch,
            AispectTouchNormalizationProfile normalizationProfile
    ) {
        return predict(window, modelId, contactPatch, normalizationProfile, null);
    }

    public AispectModels.ImpactPrediction predict(
            AispectSignalWindowBuilder.Window window,
            String modelId,
            AispectContactPatch contactPatch,
            AispectTouchNormalizationProfile normalizationProfile,
            AispectContactEncodingProfile contactEncodingProfile
    ) {
        Model model = modelForId(modelId);
        if (model == null || window == null) {
            return null;
        }
        double[][] features = makeFeatures(window, model, contactPatch, normalizationProfile, contactEncodingProfile);
        if (features == null) {
            return null;
        }
        double[][] x = transposeToChannels(features, model.inputChannels, model.frameIndices.length);
        x = conv1d(x, model.feature0Weight, model.feature0Bias, model.feature0OutChannels, model.inputChannels, model.frameIndices.length);
        x = relu(batchNorm(x, model.bn0Weight, model.bn0Bias, model.bn0Mean, model.bn0Var));
        x = conv1d(x, model.feature4Weight, model.feature4Bias, model.feature4OutChannels, model.feature0OutChannels, x[0].length);
        x = relu(batchNorm(x, model.bn1Weight, model.bn1Bias, model.bn1Mean, model.bn1Var));
        x = maxPool1d2(x);
        x = conv1d(x, model.feature9Weight, model.feature9Bias, model.feature9OutChannels, model.feature4OutChannels, x[0].length);
        x = relu(batchNorm(x, model.bn2Weight, model.bn2Bias, model.bn2Mean, model.bn2Var));
        x = maxPool1d2(x);

        double[] flat = flatten(x);
        if (flat.length != model.classifier0InFeatures) {
            return null;
        }
        double[] hidden = relu(linear(flat, model.classifier0Weight, model.classifier0Bias, 64, model.classifier0InFeatures));
        double[] logits = linear(hidden, model.classifier3Weight, model.classifier3Bias, model.classCount, 64);
        double[] probabilities = softmax(logits);
        if (probabilities.length != model.classCount) {
            return null;
        }
        return new AispectModels.ImpactPrediction(probabilities, model.labelOrder, model.id);
    }

    public boolean hasAnyModel() {
        for (ModelInfo info : loadCatalogModels()) {
            if (modelForId(info.id) != null) {
                return true;
            }
        }
        return modelForId(MODEL_G4) != null || modelForId(MODEL_G13) != null;
    }

    private Model modelForId(String modelId) {
        if (modelId == null) {
            return null;
        }
        if (!modelLoadAttempts.containsKey(modelId)) {
            modelLoadAttempts.put(modelId, true);
            ModelInfo info = modelInfoForId(modelId);
            if (info == null) {
                info = fallbackModelInfo(modelId);
            }
            Model model = loadModel(info);
            if (model != null) {
                modelCache.put(modelId, model);
            }
        }
        return modelCache.get(modelId);
    }

    private Model loadModel(ModelInfo info) {
        if (info == null) {
            return null;
        }
        try {
            JSONObject weights = new JSONObject(readAsset(resourceAssetPath(info.weightsResourceName)));
            JSONObject scaler = new JSONObject(readAsset(resourceAssetPath(info.scalerResourceName)));
            Tensor feature0 = tensor(weights, "features.0.weight");
            Tensor feature4 = tensor(weights, "features.4.weight");
            Tensor feature9 = tensor(weights, "features.9.weight");
            Tensor classifier0 = tensor(weights, "classifier.0.weight");
            int inputChannels = feature0.shape[1];
            double[] center = numberArray(scaler.getJSONArray("center"));
            double[] scale = numberArray(scaler.getJSONArray("scale"));
            String[] featureNames = featureNames(scaler, inputChannels);
            int[] frameIndices = frameIndices(scaler, info.id);
            Tensor classifier3 = tensor(weights, "classifier.3.weight");
            int classCount = classCount(scaler, classifier3);
            String[] labelOrder = labelOrder(scaler, classCount);
            if (center.length != inputChannels || scale.length != inputChannels || featureNames.length != inputChannels) {
                return null;
            }
            return new Model(
                    info.id,
                    inputChannels,
                    frameIndices,
                    featureNames,
                    center,
                    sanitizeScale(scale),
                    feature0.values,
                    tensor(weights, "features.0.bias").values,
                    feature0.shape[0],
                    tensor(weights, "features.1.weight").values,
                    tensor(weights, "features.1.bias").values,
                    tensor(weights, "features.1.running_mean").values,
                    tensor(weights, "features.1.running_var").values,
                    feature4.values,
                    tensor(weights, "features.4.bias").values,
                    feature4.shape[0],
                    tensor(weights, "features.5.weight").values,
                    tensor(weights, "features.5.bias").values,
                    tensor(weights, "features.5.running_mean").values,
                    tensor(weights, "features.5.running_var").values,
                    feature9.values,
                    tensor(weights, "features.9.bias").values,
                    feature9.shape[0],
                    tensor(weights, "features.10.weight").values,
                    tensor(weights, "features.10.bias").values,
                    tensor(weights, "features.10.running_mean").values,
                    tensor(weights, "features.10.running_var").values,
                    classifier0.values,
                    tensor(weights, "classifier.0.bias").values,
                    classifier0.shape[1],
                    classifier3.values,
                    tensor(weights, "classifier.3.bias").values,
                    classCount,
                    labelOrder
            );
        } catch (IOException | JSONException | RuntimeException error) {
            return null;
        }
    }

    private List<ModelInfo> loadCatalogModels() {
        if (catalogModels != null) {
            return catalogModels;
        }
        ArrayList<ModelInfo> models = new ArrayList<>();
        try {
            JSONArray array = new JSONArray(readAsset(CATALOG_ASSET));
            for (int i = 0; i < array.length(); i++) {
                JSONObject object = array.getJSONObject(i);
                String id = object.getString("id");
                String displayName = object.optString("displayName", id);
                String weightsResourceName = object.optString("weightsResourceName", "AispectImpactCNNWeights_" + id);
                String scalerResourceName = object.optString("scalerResourceName", "AispectImpactCNNScaler_" + id);
                ModelInfo info = catalogModelInfo(id, displayName, weightsResourceName, scalerResourceName);
                if (info != null) {
                    models.add(info);
                }
            }
        } catch (IOException | JSONException | RuntimeException error) {
            ModelInfo fallback = fallbackModelInfo(MODEL_G13);
            if (fallback != null) {
                models.add(fallback);
            }
        }
        catalogModels = models;
        return catalogModels;
    }

    private ModelInfo catalogModelInfo(
            String id,
            String displayName,
            String weightsResourceName,
            String scalerResourceName
    ) {
        try {
            JSONObject scaler = new JSONObject(readAsset(resourceAssetPath(scalerResourceName)));
            int inputChannels = scaler.getJSONArray("center").length();
            String[] featureNames = featureNames(scaler, inputChannels);
            int[] frameIndices = frameIndices(scaler, id);
            int frameCount = scaler.optInt("frameCount", frameIndices.length);
            int classCount = scaler.optInt("classCount", 2);
            String[] labelOrder = labelOrder(scaler, classCount);
            return new ModelInfo(id, displayName, weightsResourceName, scalerResourceName, inputChannels, frameCount, frameIndices, featureNames, classCount, labelOrder);
        } catch (IOException | JSONException | RuntimeException error) {
            return null;
        }
    }

    private static ModelInfo fallbackModelInfo(String modelId) {
        String weights = "AispectImpactCNNWeights_" + modelId;
        String scaler = "AispectImpactCNNScaler_" + modelId;
        int[] indices = new int[0];
        String[] features = new String[0];
        int channels = 0;
        int frameCount = 0;
        String[] labelOrder = new String[]{"heavy", "light"};
        if (MODEL_G4.equals(modelId)) {
            indices = new int[21];
            for (int i = 0; i < indices.length; i++) {
                indices[i] = i - 5;
            }
            features = new String[]{"delta", "x", "y", "z"};
            channels = 4;
            frameCount = 21;
        }
        return new ModelInfo(modelId, modelId, weights, scaler, channels, frameCount, indices, features, labelOrder.length, labelOrder);
    }

    private double[][] makeFeatures(
            AispectSignalWindowBuilder.Window window,
            Model model,
            AispectContactPatch contactPatch,
            AispectTouchNormalizationProfile normalizationProfile,
            AispectContactEncodingProfile contactEncodingProfile
    ) {
        // 手机端必须按训练导出的 featureNames 顺序取值，任何通道顺序漂移都会让 CNN 权重失效。
        Map<Integer, AispectModels.ImpactFrame> byIndex = new HashMap<>();
        for (int i = 0; i < window.frames.length; i++) {
            byIndex.put(window.firstFrameIndex + i, window.frames[i]);
        }
        ContactFeatureCache contactFeatures = new ContactFeatureCache(contactPatch, normalizationProfile, contactEncodingProfile);
        double[][] output = new double[model.frameIndices.length][model.inputChannels];
        for (int frame = 0; frame < model.frameIndices.length; frame++) {
            AispectModels.ImpactFrame source = byIndex.get(model.frameIndices[frame]);
            if (source == null) {
                // 缺失时间帧与训练脚本一致补零，保证短窗口样本仍能推理，不在手机端静默丢样本。
                source = AispectModels.ImpactFrame.zero();
            }
            for (int channel = 0; channel < model.inputChannels; channel++) {
                double raw = featureValue(source, model.featureNames[channel], contactPatch, normalizationProfile, contactFeatures);
                output[frame][channel] = (raw - model.center[channel]) / model.scale[channel];
            }
        }
        return output;
    }

    private static double featureValue(
            AispectModels.ImpactFrame frame,
            String name,
            AispectContactPatch contactPatch,
            AispectTouchNormalizationProfile normalizationProfile,
            ContactFeatureCache contactFeatures
    ) {
        if ("delta".equals(name)) {
            return frame.delta;
        }
        if ("x".equals(name) || "userAcceleration.x".equals(name)) {
            return frame.x;
        }
        if ("y".equals(name) || "userAcceleration.y".equals(name)) {
            return frame.y;
        }
        if ("z".equals(name) || "userAcceleration.z".equals(name)) {
            return frame.z;
        }
        if ("rotationRate.x".equals(name)) {
            return frame.rotationRateX;
        }
        if ("rotationRate.y".equals(name)) {
            return frame.rotationRateY;
        }
        if ("rotationRate.z".equals(name)) {
            return frame.rotationRateZ;
        }
        if ("gravity.x".equals(name)) {
            return frame.gravityX;
        }
        if ("gravity.y".equals(name)) {
            return frame.gravityY;
        }
        if ("gravity.z".equals(name)) {
            return frame.gravityZ;
        }
        if ("x_norm".equals(name)) {
            return frame.xNorm;
        }
        if ("y_norm".equals(name)) {
            return frame.yNorm;
        }
        if ("time_since_touch_down".equals(name)) {
            return frame.timeSinceTouchDown;
        }
        if ("time_since_anchor".equals(name)) {
            return frame.timeSinceAnchor;
        }
        if (name != null && name.startsWith("contactPatch.")) {
            return contactPatchValue(contactPatch, normalizationProfile, name);
        }
        if (name != null && name.startsWith("touchNormalization.")) {
            return touchNormalizationValue(normalizationProfile, name);
        }
        if (name != null && contactFeatures != null) {
            return contactFeatures.value(name);
        }
        return 0;
    }

    private static final class ContactFeatureCache {
        // 接触特征缓存负责把同一份原始接触区域映射成高斯、多通道和协方差三种训练表达。
        private final AispectContactPatch patch;
        private final AispectContactEncodingProfile encodingProfile;
        private double[] gaussianMap;
        private double gaussianSum = Double.NaN;
        private double gaussianEnergy = Double.NaN;
        private double gaussianSpread = Double.NaN;

        ContactFeatureCache(
                AispectContactPatch contactPatch,
                AispectTouchNormalizationProfile normalizationProfile,
                AispectContactEncodingProfile contactEncodingProfile
        ) {
            if (contactPatch != null) {
                patch = contactPatch;
            } else if (normalizationProfile != null) {
                patch = normalizationProfile.contactPatch;
            } else {
                patch = null;
            }
            encodingProfile = contactEncodingProfile;
        }

        double value(String name) {
            if (name.startsWith("gaussian.map.")) {
                int index = parseMapIndex(name);
                double[] map = mapValues();
                if (index >= 0 && index < map.length) {
                    return map[index];
                }
                return 0;
            }
            if ("gaussian.center".equals(name)) {
                double[] map = mapValues();
                return map[(CONTACT_MAP_SIZE * CONTACT_MAP_SIZE) / 2];
            }
            if ("gaussian.sum".equals(name)) {
                return gaussianSum();
            }
            if ("gaussian.energy".equals(name)) {
                return gaussianEnergy();
            }
            if ("gaussian.spread".equals(name)) {
                return gaussianSpread();
            }
            if ("gaussian.sigma_x_px".equals(name)) {
                return sigmaXPx();
            }
            if ("gaussian.sigma_y_px".equals(name)) {
                return sigmaYPx();
            }
            if ("gaussian.valid_mask".equals(name)) {
                return validMask();
            }
            if ("multi.center_x_norm".equals(name) || "covariance.x_norm".equals(name)) {
                if (encodingProfile != null && encodingProfile.covarianceMatrix != null) {
                    return encodingProfile.covarianceMatrix.centerXNorm;
                }
                return patch == null ? 0 : patch.xNorm;
            }
            if ("multi.center_y_norm".equals(name) || "covariance.y_norm".equals(name)) {
                if (encodingProfile != null && encodingProfile.covarianceMatrix != null) {
                    return encodingProfile.covarianceMatrix.centerYNorm;
                }
                return patch == null ? 0 : patch.yNorm;
            }
            if ("multi.major_px".equals(name)) {
                return rawMajorPx();
            }
            if ("multi.minor_px".equals(name)) {
                return rawMinorPx();
            }
            if ("multi.orientation_rad".equals(name)) {
                return patch == null ? 0 : patch.orientationRad;
            }
            if ("multi.valid_mask".equals(name) || "covariance.valid_mask".equals(name)) {
                return validMask();
            }
            if ("covariance.cov_11".equals(name)) {
                if (encodingProfile != null && encodingProfile.covarianceMatrix != null) {
                    return encodingProfile.covarianceMatrix.covariance11;
                }
                return patch == null ? 0 : patch.covariance11;
            }
            if ("covariance.cov_12".equals(name)) {
                if (encodingProfile != null && encodingProfile.covarianceMatrix != null) {
                    return encodingProfile.covarianceMatrix.covariance12;
                }
                return patch == null ? 0 : patch.covariance12;
            }
            if ("covariance.cov_22".equals(name)) {
                if (encodingProfile != null && encodingProfile.covarianceMatrix != null) {
                    return encodingProfile.covarianceMatrix.covariance22;
                }
                return patch == null ? 0 : patch.covariance22;
            }
            return 0;
        }

        private double[] mapValues() {
            if (gaussianMap != null) {
                return gaussianMap;
            }
            if (encodingProfile != null && encodingProfile.gaussianMap != null && encodingProfile.gaussianMap.heatmap != null) {
                // 新样本优先读取采集时已落盘的热图，保证训练端和手机端使用同一套编码。
                gaussianMap = encodingProfile.gaussianMap.heatmap.clone();
                if (gaussianMap.length == CONTACT_MAP_SIZE * CONTACT_MAP_SIZE) {
                    return gaussianMap;
                }
            }
            // 历史样本或缺失编码时现场重建热图，并用默认 sigma 配合 validMask 表示未知接触区域。
            gaussianMap = buildGaussianMap(sigmaXPx(), sigmaYPx(), patch == null ? 0 : patch.orientationRad);
            return gaussianMap;
        }

        private double gaussianSum() {
            if (Double.isNaN(gaussianSum)) {
                double total = 0;
                for (double value : mapValues()) {
                    total += value;
                }
                gaussianSum = total;
            }
            return gaussianSum;
        }

        private double gaussianEnergy() {
            if (Double.isNaN(gaussianEnergy)) {
                double total = 0;
                for (double value : mapValues()) {
                    total += value * value;
                }
                gaussianEnergy = total;
            }
            return gaussianEnergy;
        }

        private double gaussianSpread() {
            if (Double.isNaN(gaussianSpread)) {
                double total = gaussianSum();
                if (total <= 0) {
                    gaussianSpread = 0;
                } else {
                    int center = CONTACT_MAP_SIZE / 2;
                    int index = 0;
                    double spread = 0;
                    for (int y = 0; y < CONTACT_MAP_SIZE; y++) {
                        for (int x = 0; x < CONTACT_MAP_SIZE; x++) {
                            double dx = x - center;
                            double dy = y - center;
                            spread += mapValues()[index] * Math.sqrt(dx * dx + dy * dy);
                            index += 1;
                        }
                    }
                    gaussianSpread = spread / total;
                }
            }
            return gaussianSpread;
        }

        private double sigmaXPx() {
            double raw = rawMajorPx();
            if (validMask() > 0 && raw > 0) {
                return Math.max(1.0, raw * 0.5);
            }
            // 缺半径时不能填零，零会被模型误解成极小接触；默认值只作占位，可信度由 validMask 表达。
            return CONTACT_DEFAULT_SIGMA_PX;
        }

        private double sigmaYPx() {
            double raw = rawMinorPx();
            if (validMask() > 0 && raw > 0) {
                return Math.max(1.0, raw * 0.5);
            }
            // 缺半径时不能填零，零会被模型误解成极小接触；默认值只作占位，可信度由 validMask 表达。
            return CONTACT_DEFAULT_SIGMA_PX;
        }

        private double rawMajorPx() {
            if (encodingProfile != null && encodingProfile.multiChannel != null) {
                return encodingProfile.multiChannel.rawMajorPx;
            }
            if (patch != null && patch.rawMajorPx > 0) {
                return patch.rawMajorPx;
            }
            if (patch != null && patch.sigmaXNorm > 0) {
                return patch.sigmaXNorm * 2.0;
            }
            return CONTACT_DEFAULT_SIGMA_PX * 2.0;
        }

        private double rawMinorPx() {
            if (encodingProfile != null && encodingProfile.multiChannel != null) {
                return encodingProfile.multiChannel.rawMinorPx;
            }
            if (patch != null && patch.rawMinorPx > 0) {
                return patch.rawMinorPx;
            }
            if (patch != null && patch.sigmaYNorm > 0) {
                return patch.sigmaYNorm * 2.0;
            }
            return CONTACT_DEFAULT_SIGMA_PX * 2.0;
        }

        private double validMask() {
            if (encodingProfile != null && encodingProfile.multiChannel != null) {
                return encodingProfile.multiChannel.validMask;
            }
            return patch != null && patch.valid ? 1.0 : 0.0;
        }
    }

    private static double[] buildGaussianMap(double sigmaXPx, double sigmaYPx, double orientationRad) {
        // 固定 9x9 热图把不同来源的半径和椭圆统一成接触概率图，便于和离线 CNN 训练对齐。
        double[] output = new double[CONTACT_MAP_SIZE * CONTACT_MAP_SIZE];
        double cos = Math.cos(orientationRad);
        double sin = Math.sin(orientationRad);
        int center = CONTACT_MAP_SIZE / 2;
        int index = 0;
        for (int y = 0; y < CONTACT_MAP_SIZE; y++) {
            for (int x = 0; x < CONTACT_MAP_SIZE; x++) {
                double dx = ((x - center) / (double) center) * CONTACT_GRID_RADIUS_PX;
                double dy = ((y - center) / (double) center) * CONTACT_GRID_RADIUS_PX;
                double rotatedX = cos * dx + sin * dy;
                double rotatedY = -sin * dx + cos * dy;
                double exponent = -0.5 * (
                        (rotatedX * rotatedX) / Math.max(1.0, sigmaXPx * sigmaXPx)
                                + (rotatedY * rotatedY) / Math.max(1.0, sigmaYPx * sigmaYPx)
                );
                output[index] = Math.exp(exponent);
                index += 1;
            }
        }
        return output;
    }

    private static int parseMapIndex(String name) {
        try {
            return Integer.parseInt(name.substring("gaussian.map.".length()));
        } catch (NumberFormatException error) {
            return -1;
        }
    }

    private static double contactPatchValue(
            AispectContactPatch contactPatch,
            AispectTouchNormalizationProfile normalizationProfile,
            String name
    ) {
        // 增强 22 通道优先读取整次触摸的接触区域，归一化画像只作为缺参兜底来源。
        AispectContactPatch patch = contactPatch;
        if (patch == null && normalizationProfile != null) {
            patch = normalizationProfile.contactPatch;
        }
        if (patch == null) {
            return 0;
        }
        if ("contactPatch.cov_11".equals(name)) {
            return patch.covariance11;
        }
        if ("contactPatch.cov_12".equals(name)) {
            return patch.covariance12;
        }
        if ("contactPatch.cov_22".equals(name)) {
            return patch.covariance22;
        }
        if ("contactPatch.sigmaX".equals(name)) {
            return patch.sigmaXNorm;
        }
        if ("contactPatch.sigmaY".equals(name)) {
            return patch.sigmaYNorm;
        }
        if ("contactPatch.rawMajorPx".equals(name)) {
            return patch.rawMajorPx;
        }
        if ("contactPatch.rawMinorPx".equals(name)) {
            return patch.rawMinorPx;
        }
        return 0;
    }

    private static double touchNormalizationValue(AispectTouchNormalizationProfile normalizationProfile, String name) {
        // touchNormalization 输出的是当前选中画像的数值和质量信号，其中 confidence 不是力度标签。
        if (normalizationProfile == null || normalizationProfile.selectedResult == null) {
            return 0;
        }
        AispectTouchNormalizationProfile.Result selected = normalizationProfile.selectedResult;
        if ("touchNormalization.majorValue".equals(name)) {
            return selected.majorValue;
        }
        if ("touchNormalization.minorValue".equals(name)) {
            return selected.minorValue;
        }
        if ("touchNormalization.areaValue".equals(name)) {
            return selected.areaValue;
        }
        if ("touchNormalization.confidence".equals(name)) {
            return selected.confidence;
        }
        if ("touchNormalization.rawMajorPx".equals(name)) {
            return selected.rawMajorPx;
        }
        if ("touchNormalization.rawMinorPx".equals(name)) {
            return selected.rawMinorPx;
        }
        if ("touchNormalization.screenAreaValue".equals(name) && normalizationProfile.screenScale != null) {
            return normalizationProfile.screenScale.areaValue;
        }
        if ("touchNormalization.covarianceAreaValue".equals(name) && normalizationProfile.covarianceMatrix != null) {
            return normalizationProfile.covarianceMatrix.areaValue;
        }
        if ("touchNormalization.baselineAreaValue".equals(name) && normalizationProfile.sessionBaseline != null) {
            return normalizationProfile.sessionBaseline.areaValue;
        }
        return 0;
    }

    private static double[][] transposeToChannels(double[][] frames, int channels, int length) {
        double[][] output = new double[channels][length];
        for (int index = 0; index < length; index++) {
            for (int channel = 0; channel < channels; channel++) {
                output[channel][index] = frames[index][channel];
            }
        }
        return output;
    }

    private static double[][] conv1d(double[][] input, double[] weight, double[] bias, int outChannels, int inChannels, int length) {
        int kernelSize = 3;
        int padding = 1;
        double[][] output = new double[outChannels][length];
        for (int outChannel = 0; outChannel < outChannels; outChannel++) {
            for (int position = 0; position < length; position++) {
                double sum = bias[outChannel];
                for (int inChannel = 0; inChannel < inChannels; inChannel++) {
                    for (int kernel = 0; kernel < kernelSize; kernel++) {
                        int source = position + kernel - padding;
                        if (source < 0 || source >= length) {
                            continue;
                        }
                        int weightIndex = ((outChannel * inChannels + inChannel) * kernelSize) + kernel;
                        sum += input[inChannel][source] * weight[weightIndex];
                    }
                }
                output[outChannel][position] = sum;
            }
        }
        return output;
    }

    private static double[][] batchNorm(double[][] input, double[] gamma, double[] beta, double[] mean, double[] variance) {
        double[][] output = new double[input.length][input[0].length];
        for (int channel = 0; channel < input.length; channel++) {
            double denominator = Math.sqrt(variance[channel] + EPSILON);
            for (int index = 0; index < input[channel].length; index++) {
                double normalized = (input[channel][index] - mean[channel]) / denominator;
                output[channel][index] = normalized * gamma[channel] + beta[channel];
            }
        }
        return output;
    }

    private static double[][] relu(double[][] input) {
        double[][] output = new double[input.length][input[0].length];
        for (int row = 0; row < input.length; row++) {
            for (int column = 0; column < input[row].length; column++) {
                output[row][column] = Math.max(0, input[row][column]);
            }
        }
        return output;
    }

    private static double[] relu(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(0, input[i]);
        }
        return output;
    }

    private static double[][] maxPool1d2(double[][] input) {
        int length = input[0].length / 2;
        double[][] output = new double[input.length][length];
        for (int row = 0; row < input.length; row++) {
            for (int i = 0; i < length; i++) {
                output[row][i] = Math.max(input[row][i * 2], input[row][i * 2 + 1]);
            }
        }
        return output;
    }

    private static double[] flatten(double[][] input) {
        double[] output = new double[input.length * input[0].length];
        int index = 0;
        for (double[] row : input) {
            for (double value : row) {
                output[index] = value;
                index += 1;
            }
        }
        return output;
    }

    private static double[] linear(double[] input, double[] weight, double[] bias, int outFeatures, int inFeatures) {
        double[] output = new double[outFeatures];
        for (int outFeature = 0; outFeature < outFeatures; outFeature++) {
            double sum = bias[outFeature];
            int base = outFeature * inFeatures;
            for (int inFeature = 0; inFeature < inFeatures; inFeature++) {
                sum += input[inFeature] * weight[base + inFeature];
            }
            output[outFeature] = sum;
        }
        return output;
    }

    private static double[] softmax(double[] input) {
        double max = input[0];
        for (double value : input) {
            max = Math.max(max, value);
        }
        double total = 0;
        double[] exp = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            exp[i] = Math.exp(input[i] - max);
            total += exp[i];
        }
        if (total <= 0) {
            double[] fallback = new double[input.length];
            for (int i = 0; i < input.length; i++) {
                fallback[i] = 1.0 / input.length;
            }
            return fallback;
        }
        for (int i = 0; i < exp.length; i++) {
            exp[i] = exp[i] / total;
        }
        return exp;
    }

    private String readAsset(String name) throws IOException {
        AssetManager assets = context.getAssets();
        try (InputStream input = assets.open(name); ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[16 * 1024];
            int read;
            while ((read = input.read(buffer)) >= 0) {
                output.write(buffer, 0, read);
            }
            return output.toString(StandardCharsets.UTF_8.name());
        }
    }

    private static String resourceAssetPath(String resourceName) {
        if (resourceName == null || resourceName.isEmpty()) {
            return "";
        }
        String name = resourceName.endsWith(".json") ? resourceName : resourceName + ".json";
        if (name.startsWith(ASSET_DIR + "/")) {
            return name;
        }
        return ASSET_DIR + "/" + name;
    }

    private static Tensor tensor(JSONObject root, String name) throws JSONException {
        JSONObject object = root.getJSONObject(name);
        return new Tensor(intArray(object.getJSONArray("shape")), numberArray(object.getJSONArray("values")));
    }

    private static int[] frameIndices(JSONObject scaler, String modelId) throws JSONException {
        if (scaler.has("frameIndices")) {
            return intArray(scaler.getJSONArray("frameIndices"));
        }
        if (MODEL_G4.equals(modelId)) {
            int[] indices = new int[21];
            for (int i = 0; i < indices.length; i++) {
                indices[i] = i - 5;
            }
            return indices;
        }
        if (scaler.has("frameCount")) {
            int count = scaler.getInt("frameCount");
            int[] indices = new int[count];
            for (int i = 0; i < count; i++) {
                indices[i] = i;
            }
            return indices;
        }
        return new int[0];
    }

    private static String[] featureNames(JSONObject scaler, int inputChannels) throws JSONException {
        if (scaler.has("featureNames")) {
            JSONArray array = scaler.getJSONArray("featureNames");
            String[] values = new String[array.length()];
            for (int i = 0; i < array.length(); i++) {
                values[i] = array.getString(i);
            }
            return values;
        }
        if (inputChannels == 4) {
            return new String[]{"delta", "x", "y", "z"};
        }
        String[] fallback = new String[inputChannels];
        for (int i = 0; i < inputChannels; i++) {
            fallback[i] = "";
        }
        return fallback;
    }

    private static int classCount(JSONObject scaler, Tensor classifier3) {
        int outputCount = 2;
        if (classifier3 != null && classifier3.shape.length > 0 && classifier3.shape[0] > 0) {
            outputCount = classifier3.shape[0];
        }
        return scaler.optInt("classCount", outputCount);
    }

    private static String[] labelOrder(JSONObject scaler, int classCount) throws JSONException {
        if (scaler.has("labelOrder")) {
            JSONArray array = scaler.getJSONArray("labelOrder");
            String[] labels = new String[array.length()];
            for (int i = 0; i < array.length(); i++) {
                labels[i] = array.getString(i);
            }
            if (labels.length == classCount) {
                return labels;
            }
        }
        if (classCount == 2) {
            return new String[]{"heavy", "light"};
        }
        String[] fallback = new String[classCount];
        for (int i = 0; i < classCount; i++) {
            fallback[i] = "class_" + i;
        }
        return fallback;
    }

    private static double[] sanitizeScale(double[] scale) {
        double[] output = new double[scale.length];
        for (int i = 0; i < scale.length; i++) {
            output[i] = Math.abs(scale[i]) > 1e-9 ? scale[i] : 1.0;
        }
        return output;
    }

    private static double[] numberArray(JSONArray array) throws JSONException {
        double[] values = new double[array.length()];
        for (int i = 0; i < array.length(); i++) {
            values[i] = array.getDouble(i);
        }
        return values;
    }

    private static int[] intArray(JSONArray array) throws JSONException {
        int[] values = new int[array.length()];
        for (int i = 0; i < array.length(); i++) {
            values[i] = array.getInt(i);
        }
        return values;
    }

    private static final class Tensor {
        final int[] shape;
        final double[] values;

        Tensor(int[] shape, double[] values) {
            this.shape = shape;
            this.values = values;
        }
    }

    private static final class Model {
        final String id;
        final int inputChannels;
        final int[] frameIndices;
        final String[] featureNames;
        final double[] center;
        final double[] scale;
        final double[] feature0Weight;
        final double[] feature0Bias;
        final int feature0OutChannels;
        final double[] bn0Weight;
        final double[] bn0Bias;
        final double[] bn0Mean;
        final double[] bn0Var;
        final double[] feature4Weight;
        final double[] feature4Bias;
        final int feature4OutChannels;
        final double[] bn1Weight;
        final double[] bn1Bias;
        final double[] bn1Mean;
        final double[] bn1Var;
        final double[] feature9Weight;
        final double[] feature9Bias;
        final int feature9OutChannels;
        final double[] bn2Weight;
        final double[] bn2Bias;
        final double[] bn2Mean;
        final double[] bn2Var;
        final double[] classifier0Weight;
        final double[] classifier0Bias;
        final int classifier0InFeatures;
        final double[] classifier3Weight;
        final double[] classifier3Bias;
        final int classCount;
        final String[] labelOrder;

        Model(
                String id,
                int inputChannels,
                int[] frameIndices,
                String[] featureNames,
                double[] center,
                double[] scale,
                double[] feature0Weight,
                double[] feature0Bias,
                int feature0OutChannels,
                double[] bn0Weight,
                double[] bn0Bias,
                double[] bn0Mean,
                double[] bn0Var,
                double[] feature4Weight,
                double[] feature4Bias,
                int feature4OutChannels,
                double[] bn1Weight,
                double[] bn1Bias,
                double[] bn1Mean,
                double[] bn1Var,
                double[] feature9Weight,
                double[] feature9Bias,
                int feature9OutChannels,
                double[] bn2Weight,
                double[] bn2Bias,
                double[] bn2Mean,
                double[] bn2Var,
                double[] classifier0Weight,
                double[] classifier0Bias,
                int classifier0InFeatures,
                double[] classifier3Weight,
                double[] classifier3Bias,
                int classCount,
                String[] labelOrder
        ) {
            this.id = id;
            this.inputChannels = inputChannels;
            this.frameIndices = frameIndices;
            this.featureNames = featureNames;
            this.center = center;
            this.scale = scale;
            this.feature0Weight = feature0Weight;
            this.feature0Bias = feature0Bias;
            this.feature0OutChannels = feature0OutChannels;
            this.bn0Weight = bn0Weight;
            this.bn0Bias = bn0Bias;
            this.bn0Mean = bn0Mean;
            this.bn0Var = bn0Var;
            this.feature4Weight = feature4Weight;
            this.feature4Bias = feature4Bias;
            this.feature4OutChannels = feature4OutChannels;
            this.bn1Weight = bn1Weight;
            this.bn1Bias = bn1Bias;
            this.bn1Mean = bn1Mean;
            this.bn1Var = bn1Var;
            this.feature9Weight = feature9Weight;
            this.feature9Bias = feature9Bias;
            this.feature9OutChannels = feature9OutChannels;
            this.bn2Weight = bn2Weight;
            this.bn2Bias = bn2Bias;
            this.bn2Mean = bn2Mean;
            this.bn2Var = bn2Var;
            this.classifier0Weight = classifier0Weight;
            this.classifier0Bias = classifier0Bias;
            this.classifier0InFeatures = classifier0InFeatures;
            this.classifier3Weight = classifier3Weight;
            this.classifier3Bias = classifier3Bias;
            this.classCount = classCount;
            this.labelOrder = labelOrder;
        }
    }
}
