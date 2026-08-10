package com.zhifaios.eyes.aispect.legacy;

import com.zhifaios.eyes.aispect.AispectContactEncodingProfile;
import com.zhifaios.eyes.aispect.AispectContactPatch;
import com.zhifaios.eyes.aispect.AispectModels;
import com.zhifaios.eyes.aispect.AispectSignalWindowBuilder;
import com.zhifaios.eyes.aispect.AispectTouchNormalizationProfile;

import java.util.HashMap;
import java.util.Map;

/**
 * 历史模型的特征读取器。
 *
 * 该类只为旧模型复现保留，允许读取整次触摸聚合字段；任何 causal_* 合同都不能进入此类。
 */
public final class AispectLegacyFeatureBuilder {
    private static final int CONTACT_MAP_SIZE = 9;
    private static final double CONTACT_GRID_RADIUS_PX = 48.0;
    private static final double CONTACT_DEFAULT_SIGMA_PX = 12.0;

    private AispectLegacyFeatureBuilder() {
    }

    public static double[][] build(
            AispectSignalWindowBuilder.Window window,
            int[] frameIndices,
            int inputChannels,
            String[] featureNames,
            double[] center,
            double[] scale,
            AispectContactPatch contactPatch,
            AispectTouchNormalizationProfile normalizationProfile,
            AispectContactEncodingProfile contactEncodingProfile
    ) {
        if (window == null || frameIndices == null || featureNames == null
                || center == null || scale == null
                || frameIndices.length != featureNames.length
                || inputChannels != featureNames.length
                || center.length != inputChannels
                || scale.length != inputChannels) {
            return null;
        }
        Map<Integer, AispectModels.ImpactFrame> byIndex = new HashMap<>();
        for (int i = 0; i < window.frames.length; i++) {
            byIndex.put(window.firstFrameIndex + i, window.frames[i]);
        }
        ContactFeatureCache contactFeatures = new ContactFeatureCache(
                contactPatch,
                normalizationProfile,
                contactEncodingProfile
        );
        double[][] output = new double[frameIndices.length][inputChannels];
        for (int frame = 0; frame < frameIndices.length; frame++) {
            AispectModels.ImpactFrame source = byIndex.get(frameIndices[frame]);
            if (source == null) {
                source = AispectModels.ImpactFrame.zero();
            }
            for (int channel = 0; channel < inputChannels; channel++) {
                double raw = featureValue(
                        source,
                        featureNames[channel],
                        contactPatch,
                        normalizationProfile,
                        contactFeatures
                );
                output[frame][channel] = (raw - center[channel]) / sanitizeScale(scale[channel]);
            }
        }
        return output;
    }

    private static double sanitizeScale(double value) {
        return Double.isFinite(value) && Math.abs(value) >= 1e-12 ? value : 1.0;
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
                return index >= 0 && index < map.length ? map[index] : 0;
            }
            if ("gaussian.center".equals(name)) {
                return mapValues()[(CONTACT_MAP_SIZE * CONTACT_MAP_SIZE) / 2];
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
            if (encodingProfile != null && encodingProfile.gaussianMap != null
                    && encodingProfile.gaussianMap.heatmap != null) {
                gaussianMap = encodingProfile.gaussianMap.heatmap.clone();
                if (gaussianMap.length == CONTACT_MAP_SIZE * CONTACT_MAP_SIZE) {
                    return gaussianMap;
                }
            }
            gaussianMap = buildGaussianMap(
                    sigmaXPx(),
                    sigmaYPx(),
                    patch == null ? 0 : patch.orientationRad
            );
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
            return validMask() > 0 && raw > 0 ? Math.max(1.0, raw * 0.5) : CONTACT_DEFAULT_SIGMA_PX;
        }

        private double sigmaYPx() {
            double raw = rawMinorPx();
            return validMask() > 0 && raw > 0 ? Math.max(1.0, raw * 0.5) : CONTACT_DEFAULT_SIGMA_PX;
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
                output[index++] = Math.exp(exponent);
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

    private static double touchNormalizationValue(
            AispectTouchNormalizationProfile normalizationProfile,
            String name
    ) {
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
        if ("touchNormalization.screenAreaValue".equals(name)
                && normalizationProfile.screenScale != null) {
            return normalizationProfile.screenScale.areaValue;
        }
        if ("touchNormalization.covarianceAreaValue".equals(name)
                && normalizationProfile.covarianceMatrix != null) {
            return normalizationProfile.covarianceMatrix.areaValue;
        }
        if ("touchNormalization.baselineAreaValue".equals(name)
                && normalizationProfile.sessionBaseline != null) {
            return normalizationProfile.sessionBaseline.areaValue;
        }
        return 0;
    }
}
