package com.zhifaios.eyes.aispect;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

public final class AispectFirstTouchClassifier {
    private static final String MODEL_ASSET = "aispect_impact_models/AispectFirstTouchModel_first_touch_early_v1.json";
    private static final int FEATURE_COUNT = 10;
    private static final double DEFAULT_PATCH_SIGMA = 0.012;

    private final String modelId;
    private final double[] center;
    private final double[] scale;
    private final double[][] hiddenWeight;
    private final double[] hiddenBias;
    private final double[][] outputWeight;
    private final double[] outputBias;
    private final String[] labelOrder;

    public AispectFirstTouchClassifier(Context context) {
        this(loadModel(context));
    }

    AispectFirstTouchClassifier(
            String modelId,
            double[] center,
            double[] scale,
            double[][] hiddenWeight,
            double[] hiddenBias,
            double[][] outputWeight,
            double[] outputBias,
            String[] labelOrder
    ) {
        this.modelId = modelId == null ? "" : modelId;
        this.center = copy(center);
        this.scale = copy(scale);
        this.hiddenWeight = copy(hiddenWeight);
        this.hiddenBias = copy(hiddenBias);
        this.outputWeight = copy(outputWeight);
        this.outputBias = copy(outputBias);
        this.labelOrder = labelOrder == null ? new String[0] : labelOrder.clone();
    }

    private AispectFirstTouchClassifier(Model model) {
        this(
                model == null ? "" : model.modelId,
                model == null ? null : model.center,
                model == null ? null : model.scale,
                model == null ? null : model.hiddenWeight,
                model == null ? null : model.hiddenBias,
                model == null ? null : model.outputWeight,
                model == null ? null : model.outputBias,
                model == null ? null : model.labelOrder
        );
    }

    public AispectModels.ImpactPrediction predict(AispectTouchFrame frame) {
        if (frame == null) {
            return null;
        }
        AispectContactPatch patch = AispectContactPatch.fromFrame(frame);
        return predictFeatures(features(
                frame.xNorm,
                frame.yNorm,
                frame.pressure,
                frame.size,
                patch.sigmaXNorm,
                patch.sigmaYNorm,
                patch.orientationRad
        ));
    }

    public String modelId() {
        return modelId;
    }

    AispectModels.ImpactPrediction predictFeatures(double[] values) {
        if (!isUsable() || values == null || values.length != FEATURE_COUNT) {
            return null;
        }
        double[] normalized = new double[FEATURE_COUNT];
        for (int index = 0; index < FEATURE_COUNT; index++) {
            normalized[index] = (finite(values[index], 0.0) - center[index]) / Math.max(1e-9, scale[index]);
        }
        double[] hidden = new double[hiddenBias.length];
        for (int row = 0; row < hidden.length; row++) {
            double value = hiddenBias[row];
            for (int column = 0; column < FEATURE_COUNT; column++) {
                value += hiddenWeight[row][column] * normalized[column];
            }
            hidden[row] = Math.max(0.0, value);
        }
        double[] logits = new double[outputBias.length];
        for (int row = 0; row < logits.length; row++) {
            double value = outputBias[row];
            for (int column = 0; column < hidden.length; column++) {
                value += outputWeight[row][column] * hidden[column];
            }
            logits[row] = value;
        }
        return new AispectModels.ImpactPrediction(softmax(logits), labelOrder, modelId);
    }

    static double[] features(
            double xNorm,
            double yNorm,
            double pressure,
            double size,
            double sigmaX,
            double sigmaY,
            double orientation
    ) {
        double safeSigmaX = Math.max(1e-9, finite(sigmaX, DEFAULT_PATCH_SIGMA));
        double safeSigmaY = Math.max(1e-9, finite(sigmaY, DEFAULT_PATCH_SIGMA));
        double safeOrientation = finite(orientation, 0.0);
        return new double[]{
                clampUnit(finite(xNorm, 0.0)),
                clampUnit(finite(yNorm, 0.0)),
                Math.max(0.0, finite(pressure, 0.0)),
                Math.max(0.0, finite(size, 0.0)),
                safeSigmaX,
                safeSigmaY,
                Math.max(-6.0, Math.min(6.0, Math.log(safeSigmaX / safeSigmaY))),
                Math.max(-20.0, Math.min(0.0, Math.log(Math.max(1e-12, safeSigmaX * safeSigmaY)))),
                Math.sin(safeOrientation),
                Math.cos(safeOrientation),
        };
    }

    private boolean isUsable() {
        if (modelId.isEmpty() || center.length != FEATURE_COUNT || scale.length != FEATURE_COUNT) {
            return false;
        }
        if (hiddenWeight.length == 0 || hiddenBias.length != hiddenWeight.length) {
            return false;
        }
        for (double[] row : hiddenWeight) {
            if (row.length != FEATURE_COUNT) {
                return false;
            }
        }
        if (outputWeight.length != labelOrder.length || outputBias.length != labelOrder.length) {
            return false;
        }
        for (double[] row : outputWeight) {
            if (row.length != hiddenWeight.length) {
                return false;
            }
        }
        return labelOrder.length > 0;
    }

    private static Model loadModel(Context context) {
        if (context == null) {
            return null;
        }
        try {
            JSONObject json = new JSONObject(readAsset(context, MODEL_ASSET));
            return new Model(
                    json.getString("modelId"),
                    numberArray(json.getJSONArray("center")),
                    numberArray(json.getJSONArray("scale")),
                    numberMatrix(json.getJSONArray("hiddenWeight")),
                    numberArray(json.getJSONArray("hiddenBias")),
                    numberMatrix(json.getJSONArray("outputWeight")),
                    numberArray(json.getJSONArray("outputBias")),
                    stringArray(json.getJSONArray("labelOrder"))
            );
        } catch (IOException | JSONException | RuntimeException error) {
            return null;
        }
    }

    private static String readAsset(Context context, String path) throws IOException {
        try (InputStream input = context.getAssets().open(path);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = input.read(buffer)) >= 0) {
                output.write(buffer, 0, count);
            }
            return output.toString(StandardCharsets.UTF_8.name());
        }
    }

    private static double[] numberArray(JSONArray array) throws JSONException {
        double[] output = new double[array.length()];
        for (int index = 0; index < output.length; index++) {
            output[index] = array.getDouble(index);
        }
        return output;
    }

    private static double[][] numberMatrix(JSONArray array) throws JSONException {
        double[][] output = new double[array.length()][];
        for (int index = 0; index < output.length; index++) {
            output[index] = numberArray(array.getJSONArray(index));
        }
        return output;
    }

    private static String[] stringArray(JSONArray array) throws JSONException {
        String[] output = new String[array.length()];
        for (int index = 0; index < output.length; index++) {
            output[index] = array.getString(index);
        }
        return output;
    }

    private static double[] softmax(double[] logits) {
        double maximum = Double.NEGATIVE_INFINITY;
        for (double value : logits) {
            maximum = Math.max(maximum, value);
        }
        double total = 0.0;
        double[] output = new double[logits.length];
        for (int index = 0; index < logits.length; index++) {
            output[index] = Math.exp(logits[index] - maximum);
            total += output[index];
        }
        if (total <= 0 || !Double.isFinite(total)) {
            return new double[logits.length];
        }
        for (int index = 0; index < output.length; index++) {
            output[index] /= total;
        }
        return output;
    }

    private static double finite(double value, double fallback) {
        return Double.isFinite(value) ? value : fallback;
    }

    private static double clampUnit(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }

    private static double[] copy(double[] values) {
        return values == null ? new double[0] : values.clone();
    }

    private static double[][] copy(double[][] values) {
        if (values == null) {
            return new double[0][];
        }
        double[][] output = new double[values.length][];
        for (int index = 0; index < values.length; index++) {
            output[index] = copy(values[index]);
        }
        return output;
    }

    private static final class Model {
        final String modelId;
        final double[] center;
        final double[] scale;
        final double[][] hiddenWeight;
        final double[] hiddenBias;
        final double[][] outputWeight;
        final double[] outputBias;
        final String[] labelOrder;

        Model(
                String modelId,
                double[] center,
                double[] scale,
                double[][] hiddenWeight,
                double[] hiddenBias,
                double[][] outputWeight,
                double[] outputBias,
                String[] labelOrder
        ) {
            this.modelId = modelId;
            this.center = center;
            this.scale = scale;
            this.hiddenWeight = hiddenWeight;
            this.hiddenBias = hiddenBias;
            this.outputWeight = outputWeight;
            this.outputBias = outputBias;
            this.labelOrder = labelOrder;
        }
    }
}
