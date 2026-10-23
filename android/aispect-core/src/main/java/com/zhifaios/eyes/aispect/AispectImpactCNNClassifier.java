package com.zhifaios.eyes.aispect;

import com.zhifaios.eyes.aispect.legacy.AispectLegacyFeatureBuilder;

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
import java.util.Arrays;
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
    private static final String MODEL_LEMON_TAO_BEST = "lemon_tao_full4_enhanced_patch21_no_confidence";
    private static final String MODEL_FINGER_FOUR_CLASS_TOP = "oneplus_four_class_after_recollect_hard_nearfull_weighted_old13_multichannel";
    private static final double EPSILON = 1e-5;

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
        public final String featureContract;
        public final String runtimeArchitecture;

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
                String[] labelOrder,
                String featureContract,
                String runtimeArchitecture
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
            this.featureContract = featureContract == null ? "" : featureContract;
            this.runtimeArchitecture = runtimeArchitecture == null ? "" : runtimeArchitecture;
        }
    }

    private final Context context;
    private final Map<String, Model> modelCache = new HashMap<>();
    private final Map<String, Boolean> modelLoadAttempts = new HashMap<>();
    private final Map<String, AispectTimeGridGroupNormRuntime> timeGridRuntimeCache = new HashMap<>();
    private final Map<String, Boolean> timeGridRuntimeLoadAttempts = new HashMap<>();
    private final Map<String, AispectStatsMlpRuntime> statsMlpRuntimeCache = new HashMap<>();
    private final Map<String, Boolean> statsMlpRuntimeLoadAttempts = new HashMap<>();
    private AispectCausalSizeBaseline causalSizeBaseline;
    private String causalSizeDeviceKey = "";
    private List<ModelInfo> catalogModels;

    public AispectImpactCNNClassifier(Context context) {
        this.context = context.getApplicationContext();
    }

    public void setDeviceIdentity(String deviceKey) {
        String normalized = deviceKey == null ? "" : deviceKey.trim();
        if (normalized.equals(causalSizeDeviceKey)
                && causalSizeBaseline != null) {
            return;
        }
        causalSizeDeviceKey = normalized;
        causalSizeBaseline = new AispectCausalSizeBaseline(
                context,
                normalized
        );
    }

    public static boolean usesCausalPressFeatureContract(String featureContract) {
        return AispectCausalPressFeatureBuilder.isCausalFeatureContract(featureContract);
    }

    public static boolean usesLegacyFeatureContract(String featureContract) {
        return !usesCausalPressFeatureContract(featureContract);
    }

    public List<ModelInfo> availableModels() {
        return Collections.unmodifiableList(loadCatalogModels());
    }

    public ModelInfo defaultModelInfo() {
        ModelInfo onePlusBest = modelInfoForId(MODEL_FINGER_FOUR_CLASS_TOP);
        if (onePlusBest != null) {
            return onePlusBest;
        }
        ModelInfo lemonTaoBest = modelInfoForId(MODEL_LEMON_TAO_BEST);
        if (lemonTaoBest != null) {
            return lemonTaoBest;
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
        return predict(window, modelId, contactPatch, normalizationProfile, null, null);
    }

    public AispectModels.ImpactPrediction predict(
            AispectSignalWindowBuilder.Window window,
            String modelId,
            List<AispectTouchFrame> touchFrames
    ) {
        return predict(window, modelId, null, null, null, touchFrames);
    }

    public AispectModels.ImpactPrediction predict(
            AispectSignalWindowBuilder.Window window,
            String modelId,
            AispectContactPatch contactPatch,
            AispectTouchNormalizationProfile normalizationProfile,
            AispectContactEncodingProfile contactEncodingProfile
    ) {
        return predict(window, modelId, contactPatch, normalizationProfile, contactEncodingProfile, null);
    }

    private AispectModels.ImpactPrediction predict(
            AispectSignalWindowBuilder.Window window,
            String modelId,
            AispectContactPatch contactPatch,
            AispectTouchNormalizationProfile normalizationProfile,
            AispectContactEncodingProfile contactEncodingProfile,
            List<AispectTouchFrame> touchFrames
    ) {
        ModelInfo requestedInfo = modelInfoForId(modelId);
        AispectStatsMlpRuntime statsMlpRuntime =
                requestedInfo != null
                        && AispectStatsMlpRuntime.isSupportedArchitecture(requestedInfo.runtimeArchitecture)
                        ? statsMlpRuntimeForId(modelId)
                        : null;
        if (statsMlpRuntime != null) {
            if (window == null || !window.windowComplete || touchFrames == null) {
                return null;
            }
            double[][] raw = AispectCausalPressFeatureBuilder.buildTimeGrid(
                    window.frames,
                    touchFrames,
                    Math.round(window.downAnchorTimestampSeconds * 1_000_000_000.0),
                    statsMlpRuntime.featureContract(),
                    AispectStatsMlpRuntime.V4_FEATURE_CONTRACT.equals(
                            statsMlpRuntime.featureContract()
                    )
                            ? causalSizeBaseline()
                            : null
            );
            double[] probabilities = statsMlpRuntime.predict(raw);
            return probabilities == null
                    ? null
                    : new AispectModels.ImpactPrediction(
                            probabilities,
                            statsMlpRuntime.labelOrder(),
                            modelId
                    );
        }
        AispectTimeGridGroupNormRuntime timeGridRuntime = timeGridRuntimeForId(modelId);
        if (timeGridRuntime != null) {
            if (window == null || !window.windowComplete || touchFrames == null) {
                return null;
            }
            double[][] raw = AispectCausalPressFeatureBuilder.buildTimeGrid(
                    window.frames,
                    touchFrames,
                    Math.round(window.downAnchorTimestampSeconds * 1_000_000_000.0),
                    timeGridRuntime.featureContract()
            );
            double[] probabilities = timeGridRuntime.predict(raw);
            return probabilities == null
                    ? null
                    : new AispectModels.ImpactPrediction(probabilities, timeGridRuntime.labelOrder(), modelId);
        }
        Model model = modelForId(modelId);
        if (model == null || window == null || !window.windowComplete) {
            return null;
        }
        double[][] features = makeFeatures(window, model, contactPatch, normalizationProfile, contactEncodingProfile, touchFrames);
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
            boolean loaded = AispectStatsMlpRuntime.isSupportedArchitecture(info.runtimeArchitecture)
                    ? statsMlpRuntimeForId(info.id) != null
                    : AispectTimeGridGroupNormRuntime.isSupportedArchitecture(info.runtimeArchitecture)
                    ? timeGridRuntimeForId(info.id) != null
                    : modelForId(info.id) != null;
            if (loaded) {
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
            String featureContract = scaler.optString("featureContract", "");
            int[] frameIndices = frameIndices(scaler, info.id);
            Tensor classifier3 = tensor(weights, "classifier.3.weight");
            int classCount = classCount(scaler, classifier3);
            String[] labelOrder = labelOrder(scaler, classCount);
            if (center.length != inputChannels || scale.length != inputChannels || featureNames.length != inputChannels) {
                return null;
            }
            if (usesCausalPressFeatureContract(featureContract)
                    && (!Arrays.equals(featureNames, AispectCausalPressFeatureBuilder.featureNames(featureContract))
                    || inputChannels != featureNames.length)) {
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
                    labelOrder,
                    featureContract
            );
        } catch (IOException | JSONException | RuntimeException error) {
            return null;
        }
    }

    private AispectTimeGridGroupNormRuntime timeGridRuntimeForId(String modelId) {
        if (modelId == null) {
            return null;
        }
        if (!timeGridRuntimeLoadAttempts.containsKey(modelId)) {
            timeGridRuntimeLoadAttempts.put(modelId, true);
            ModelInfo info = modelInfoForId(modelId);
            if (info != null) {
                try {
                    JSONObject weights = new JSONObject(readAsset(resourceAssetPath(info.weightsResourceName)));
                    JSONObject scaler = new JSONObject(readAsset(resourceAssetPath(info.scalerResourceName)));
                    AispectTimeGridGroupNormRuntime runtime = AispectTimeGridGroupNormRuntime.load(info.id, weights, scaler);
                    if (runtime != null) {
                        timeGridRuntimeCache.put(modelId, runtime);
                    }
                } catch (IOException | JSONException | RuntimeException error) {
                    return null;
                }
            }
        }
        return timeGridRuntimeCache.get(modelId);
    }

    private AispectStatsMlpRuntime statsMlpRuntimeForId(String modelId) {
        if (modelId == null) {
            return null;
        }
        if (!statsMlpRuntimeLoadAttempts.containsKey(modelId)) {
            statsMlpRuntimeLoadAttempts.put(modelId, true);
            ModelInfo info = modelInfoForId(modelId);
            if (info != null) {
                try {
                    JSONObject weights = new JSONObject(readAsset(resourceAssetPath(info.weightsResourceName)));
                    JSONObject scaler = new JSONObject(readAsset(resourceAssetPath(info.scalerResourceName)));
                    AispectStatsMlpRuntime runtime = AispectStatsMlpRuntime.load(info.id, weights, scaler);
                    if (runtime != null) {
                        statsMlpRuntimeCache.put(modelId, runtime);
                    }
                } catch (IOException | JSONException | RuntimeException error) {
                    return null;
                }
            }
        }
        return statsMlpRuntimeCache.get(modelId);
    }

    private AispectCausalSizeBaseline causalSizeBaseline() {
        if (causalSizeBaseline == null) {
            causalSizeBaseline = new AispectCausalSizeBaseline(
                    context,
                    causalSizeDeviceKey
            );
        }
        return causalSizeBaseline;
    }

    /**
     * 中文注释：只允许采集控制器在抬手后的完整窗口落定后调用。
     * 预测发生在按下期，此时当前触摸尚未结束，不能写入历史分母。
     */
    public void observeCompletedCausalSizeTouch(
            List<AispectTouchFrame> touchFrames
    ) {
        causalSizeBaseline().observeCompletedTouch(touchFrames);
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
            boolean isTimeGridRuntime = AispectTimeGridGroupNormRuntime.isSupportedArchitecture(
                    scaler.optString("runtimeArchitecture", "")
            );
            boolean isStatsMlpRuntime = AispectStatsMlpRuntime.isSupportedArchitecture(
                    scaler.optString("runtimeArchitecture", "")
            );
            String runtimeArchitecture = scaler.optString("runtimeArchitecture", "");
            int inputChannels;
            if (isStatsMlpRuntime) {
                inputChannels = scaler.getJSONArray("featureNames").length();
            } else if (isTimeGridRuntime) {
                inputChannels = scaler.getJSONArray("channels").length();
            } else {
                inputChannels = scaler.getJSONArray("center").length();
            }
            String[] featureNames = featureNames(scaler, inputChannels);
            int[] frameIndices = (isTimeGridRuntime || isStatsMlpRuntime)
                    ? sequentialFrameIndices(scaler.getInt("frameCount"))
                    : frameIndices(scaler, id);
            int frameCount = scaler.optInt("frameCount", frameIndices.length);
            int classCount = scaler.optInt("classCount", 2);
            String[] labelOrder = labelOrder(scaler, classCount);
            String featureContract = scaler.optString("featureContract", "");
            return new ModelInfo(
                    id,
                    displayName,
                    weightsResourceName,
                    scalerResourceName,
                    inputChannels,
                    frameCount,
                    frameIndices,
                    featureNames,
                    classCount,
                    labelOrder,
                    featureContract,
                    runtimeArchitecture
            );
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
        return new ModelInfo(
                modelId,
                modelId,
                weights,
                scaler,
                channels,
                frameCount,
                indices,
                features,
                labelOrder.length,
                labelOrder,
                "",
                ""
        );
    }

    private double[][] makeFeatures(
            AispectSignalWindowBuilder.Window window,
            Model model,
            AispectContactPatch contactPatch,
            AispectTouchNormalizationProfile normalizationProfile,
            AispectContactEncodingProfile contactEncodingProfile,
            List<AispectTouchFrame> touchFrames
    ) {
        if (usesCausalPressFeatureContract(model.featureContract)) {
            double[][] raw;
            if (AispectCausalPressFeatureBuilder.isTimeGridFeatureContract(model.featureContract)) {
                raw = AispectCausalPressFeatureBuilder.buildTimeGrid(
                        window.frames,
                        touchFrames,
                        Math.round(window.downAnchorTimestampSeconds * 1_000_000_000.0),
                        model.featureContract
                );
            } else {
                raw = AispectCausalPressFeatureBuilder.build(
                        window.frames,
                        window.firstFrameIndex,
                        model.frameIndices,
                        touchFrames,
                        model.featureContract
                );
            }
            return normalizeCausalFeatures(raw, model);
        }
        return AispectLegacyFeatureBuilder.build(
                window,
                model.frameIndices,
                model.inputChannels,
                model.featureNames,
                model.center,
                model.scale,
                contactPatch,
                normalizationProfile,
                contactEncodingProfile
        );
    }

    private static double[][] normalizeCausalFeatures(double[][] raw, Model model) {
        if (raw == null || raw.length != model.frameIndices.length) {
            return null;
        }
        double[][] output = new double[raw.length][model.inputChannels];
        for (int frame = 0; frame < raw.length; frame++) {
            if (raw[frame] == null || raw[frame].length != model.inputChannels) {
                return null;
            }
            for (int channel = 0; channel < model.inputChannels; channel++) {
                output[frame][channel] = (raw[frame][channel] - model.center[channel]) / model.scale[channel];
            }
        }
        return output;
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

    private static int[] sequentialFrameIndices(int count) {
        int[] indices = new int[Math.max(0, count)];
        for (int index = 0; index < indices.length; index++) {
            indices[index] = index;
        }
        return indices;
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
        final String featureContract;

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
                String[] labelOrder,
                String featureContract
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
            this.featureContract = featureContract == null ? "" : featureContract;
        }
    }
}
