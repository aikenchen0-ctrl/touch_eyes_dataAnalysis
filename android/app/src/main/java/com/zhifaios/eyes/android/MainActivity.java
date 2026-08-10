package com.zhifaios.eyes.android;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.res.ColorStateList;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Shader;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.Switch;
import android.widget.TextView;
import android.text.InputType;

import com.zhifaios.eyes.aispect.AispectCollectionController;
import com.zhifaios.eyes.aispect.AispectContactEncodingProfile;
import com.zhifaios.eyes.aispect.AispectDataLogger;
import com.zhifaios.eyes.aispect.AispectDeviceCapabilityProfiler;
import com.zhifaios.eyes.aispect.AispectImpactCNNClassifier;
import com.zhifaios.eyes.aispect.AispectModels;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.util.List;
import java.util.Locale;

public final class MainActivity extends Activity implements AispectCollectionController.Listener {
    private static final int COLOR_BG = Color.rgb(0, 0, 0);
    private static final int COLOR_PANEL = Color.argb(215, 8, 9, 12);
    private static final int COLOR_PANEL_COLLAPSED = Color.argb(130, 8, 9, 12);
    private static final int COLOR_PANEL_BUTTON = Color.argb(40, 255, 255, 255);
    private static final int COLOR_PANEL_DISABLED = Color.argb(55, 255, 255, 255);
    private static final int COLOR_TEXT = Color.rgb(246, 246, 242);
    private static final int COLOR_MUTED = Color.argb(190, 246, 246, 242);
    private static final int COLOR_DIM = Color.argb(120, 246, 246, 242);
    private static final int COLOR_ACCENT = Color.rgb(112, 217, 169);
    private static final int COLOR_BLUE = Color.rgb(59, 134, 246);
    private static final int COLOR_FORCE = Color.rgb(255, 89, 64);
    private static final int PANEL_EXPANDED_WIDTH_DP = 204;
    private static final int PANEL_COLLAPSED_WIDTH_DP = 104;
    private static final int PANEL_COLLAPSED_HEIGHT_DP = 52;
    private static final int PANEL_TOGGLE_WIDTH_DP = 84;
    private static final int PANEL_TOGGLE_HEIGHT_DP = 38;
    private static final int PANEL_TOGGLE_HIT_SLOP_DP = 22;

    private AispectCollectionController controller;
    private List<AispectImpactCNNClassifier.ModelInfo> modelInfos;
    private int selectedModelIndex;
    private AispectModels.MotionState selectedMotionState = AispectModels.MotionState.HANDHELD_STATIC;
    private AispectModels.TouchStrength selectedStrength = AispectModels.TouchStrength.LIGHT;
    private AispectModels.FingerType selectedFingerType = AispectModels.FingerType.THUMB;
    private FrameLayout rootView;
    private DemoHomeView homeView;
    private LinearLayout impactPanel;
    private ScrollView panelScrollView;
    private Button panelToggleButton;
    private TextView eventValueText;
    private TextView thresholdText;
    private SeekBar thresholdSeekBar;
    private TextView rateText;
    private TextView impactLogText;
    private TextView firstTouchModelText;
    private TextView modelNameText;
    private TextView modelDetailText;
    private TextView normalizationDetailText;
    private TextView collectorGuideText;
    private EditText collectorUserNameInput;
    private TextView collectorStatusText;
    private TextView latestRawSampleText;
    private TextView resultText;
    private TextView fourClassRecallText;
    private Button validationModeButton;
    private Button previousModelButton;
    private Button nextModelButton;
    private Button normScreenButton;
    private Button normCovarianceButton;
    private Button normBaselineButton;
    private LinearLayout collectorSwitchRow;
    private Switch collectorSwitch;
    private Button lightButton;
    private Button heavyButton;
    private Button thumbButton;
    private Button indexButton;
    private Button handheldButton;
    private Button placedButton;
    private Button walkingButton;
    private Button copyDatasetPathButton;
    private Button copyLogButton;
    private boolean isPanelCollapsed;
    private boolean fullScreenCaptureActive;
    private boolean isUpdatingSwitch;
    private boolean panelToggleTouchActive;
    private boolean validationModeEnabled = true;
    private float impactThreshold = 0.25f;
    private String lastFourClassRecallSignature;
    private final FourClassRecallCounter fourClassRecallCounter = new FourClassRecallCounter();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        controller = new AispectCollectionController(this);
        controller.setSelectedModelId(BuildConfig.DEFAULT_MODEL_ID);
        controller.config().enableCnnDiagnostics = false;
        controller.interactionEngine().config().minimumSampleIntervalMs = 4L;
        controller.interactionEngine().config().dragActivationRadiusPx = dp(28);
        controller.interactionEngine().config().heavyLiftOffsetThresholdPx = dp(4);
        controller.setListener(this);
        modelInfos = controller.availableModels();
        selectedModelIndex = defaultModelIndex(controller.selectedModelInfo());
        buildUi();
        applyModelSelection();
        applyLabel();
        updateStatus(controller.status());
    }

    @Override
    protected void onDestroy() {
        controller.setEnabled(false);
        controller.shutdown();
        super.onDestroy();
    }

    @Override
    public boolean dispatchTouchEvent(MotionEvent event) {
        if (handlePanelToggleTouch(event)) {
            return true;
        }
        int action = event.getActionMasked();
        boolean panelTouch = isInsidePanel(event);
        if (!panelTouch) {
            float forceIntensity = forceIntensity(event);
            homeView.updateTouch(event, forceIntensity);
        }
        if (controller == null) {
            fullScreenCaptureActive = false;
            return super.dispatchTouchEvent(event);
        }
        if (!controller.isEnabled()) {
            if (!panelTouch && validationModeEnabled) {
                if (action == MotionEvent.ACTION_DOWN) {
                    fullScreenCaptureActive = true;
                }
                if (fullScreenCaptureActive) {
                    updateMotionEventText(event);
                    boolean handled = controller.handleDiagnosticMotionEvent(event, Math.max(1, rootView.getWidth()), Math.max(1, rootView.getHeight()));
                    if (action == MotionEvent.ACTION_UP || action == MotionEvent.ACTION_CANCEL) {
                        fullScreenCaptureActive = false;
                    }
                    return handled || super.dispatchTouchEvent(event);
                }
            }
            fullScreenCaptureActive = false;
            return super.dispatchTouchEvent(event);
        }
        if (action == MotionEvent.ACTION_DOWN) {
            fullScreenCaptureActive = !panelTouch;
        }
        if (fullScreenCaptureActive) {
            updateMotionEventText(event);
            boolean handled = controller.handleMotionEvent(event, Math.max(1, rootView.getWidth()), Math.max(1, rootView.getHeight()));
            if (action == MotionEvent.ACTION_UP || action == MotionEvent.ACTION_CANCEL) {
                fullScreenCaptureActive = false;
            }
            return handled || super.dispatchTouchEvent(event);
        }
        if (action == MotionEvent.ACTION_UP || action == MotionEvent.ACTION_CANCEL) {
            fullScreenCaptureActive = false;
        }
        return super.dispatchTouchEvent(event);
    }

    private static float forceIntensity(MotionEvent event) {
        if (event.getPointerCount() <= 0) {
            return 0f;
        }
        int pointerIndex = Math.max(0, Math.min(event.getActionIndex(), event.getPointerCount() - 1));
        float touchMajor = positive(event.getTouchMajor(pointerIndex));
        float touchMinor = positive(event.getTouchMinor(pointerIndex));
        float toolMajor = positive(event.getToolMajor(pointerIndex));
        float toolMinor = positive(event.getToolMinor(pointerIndex));
        float major = Math.max(touchMajor, toolMajor);
        float minor = Math.max(touchMinor, toolMinor);
        if (minor <= 0f && major > 0f) {
            minor = major;
        }
        float radiusIntensity = 0f;
        if (major > 0f) {
            float areaRoot = (float) Math.sqrt(Math.max(0f, major * Math.max(minor, 1f)));
            radiusIntensity = Math.max(normalized(major, 8f, 18f), normalized(areaRoot, 7f, 18f) * 0.95f);
        }
        float pressureIntensity = normalized(positive(event.getPressure(pointerIndex)), 1.05f, 1.8f);
        float sizeIntensity = normalized(positive(event.getSize(pointerIndex)), 0.008f, 0.026f);
        if (major > 0f) {
            return clampUnit(Math.max(radiusIntensity, sizeIntensity * 0.45f));
        }
        return clampUnit(Math.max(pressureIntensity, sizeIntensity));
    }

    @Override
    public void onStatusChanged(AispectCollectionController.Status status) {
        updateFourClassRecall(status);
        updateStatus(status);
    }

    @Override
    public void onRecordSaved(AispectDataLogger.SavedRecord record, AispectCollectionController.Status status) {
        updateLatestRawSample(record);
        updateStatus(status);
    }

    @Override
    public void onRecordFailed(Exception error, AispectCollectionController.Status status) {
        updateStatus(status);
    }

    private void buildUi() {
        rootView = new FrameLayout(this);
        rootView.setBackgroundColor(COLOR_BG);
        setContentView(rootView);

        homeView = new DemoHomeView(this);
        rootView.addView(homeView, new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        impactPanel = new LinearLayout(this);
        impactPanel.setOrientation(LinearLayout.VERTICAL);
        impactPanel.setPadding(dp(12), dp(10), dp(12), dp(10));
        impactPanel.setBackground(panelBackground(COLOR_PANEL, dp(10), Color.argb(90, 255, 255, 255)));
        impactPanel.setClipToOutline(false);
        impactPanel.setClickable(true);
        FrameLayout.LayoutParams panelParams = new FrameLayout.LayoutParams(dp(PANEL_EXPANDED_WIDTH_DP), ViewGroup.LayoutParams.WRAP_CONTENT);
        panelParams.gravity = Gravity.TOP | Gravity.RIGHT;
        panelParams.rightMargin = dp(12);
        panelParams.topMargin = dp(28);
        rootView.addView(impactPanel, panelParams);

        panelToggleButton = panelButton(getString(R.string.impact_panel_collapse), 11f);
        panelToggleButton.setBackground(panelBackground(COLOR_BLUE, dp(8), Color.argb(70, 255, 255, 255)));
        LinearLayout.LayoutParams toggleParams = new LinearLayout.LayoutParams(dp(PANEL_TOGGLE_WIDTH_DP), dp(PANEL_TOGGLE_HEIGHT_DP));
        toggleParams.gravity = Gravity.RIGHT;
        impactPanel.addView(panelToggleButton, toggleParams);

        panelScrollView = new ScrollView(this);
        panelScrollView.setFillViewport(false);
        panelScrollView.setVerticalScrollBarEnabled(true);
        LinearLayout.LayoutParams scrollParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(660));
        scrollParams.topMargin = dp(6);
        impactPanel.addView(panelScrollView, scrollParams);

        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        panelScrollView.addView(content, new ScrollView.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        TextView title = sectionTitle(R.string.impact_panel_title, 12f);
        content.addView(title, fullWidthWrap());

        eventValueText = monospaceText(10f, COLOR_TEXT);
        eventValueText.setText(R.string.impact_event_idle);
        eventValueText.setLineSpacing(dp(2), 1f);
        addWithTopMargin(content, eventValueText, dp(6), ViewGroup.LayoutParams.WRAP_CONTENT);

        thresholdText = sectionTitle(R.string.impact_threshold_title, 11f);
        addWithTopMargin(content, thresholdText, dp(8), ViewGroup.LayoutParams.WRAP_CONTENT);

        thresholdSeekBar = new SeekBar(this);
        thresholdSeekBar.setEnabled(false);
        thresholdSeekBar.setMax(950);
        thresholdSeekBar.setProgress(200);
        thresholdSeekBar.setPadding(0, 0, 0, 0);
        if (android.os.Build.VERSION.SDK_INT >= 21) {
            thresholdSeekBar.setProgressTintList(ColorStateList.valueOf(COLOR_ACCENT));
            thresholdSeekBar.setThumbTintList(ColorStateList.valueOf(COLOR_TEXT));
            thresholdSeekBar.setProgressBackgroundTintList(ColorStateList.valueOf(Color.argb(70, 255, 255, 255)));
        }
        addWithTopMargin(content, thresholdSeekBar, dp(4), dp(24));

        rateText = bodyText(10f, COLOR_MUTED);
        rateText.setText(R.string.impact_rate_idle);
        addWithTopMargin(content, rateText, dp(4), ViewGroup.LayoutParams.WRAP_CONTENT);

        impactLogText = monospaceText(9f, COLOR_MUTED);
        impactLogText.setText(R.string.impact_log_idle);
        impactLogText.setMinLines(6);
        impactLogText.setLineSpacing(dp(1), 1f);
        addWithTopMargin(content, impactLogText, dp(6), ViewGroup.LayoutParams.WRAP_CONTENT);
        impactLogText.setVisibility(View.GONE);

        copyLogButton = panelButton(getString(R.string.impact_copy_log), 10f);
        addWithTopMargin(content, copyLogButton, dp(6), dp(26));
        copyLogButton.setVisibility(View.GONE);

        TextView firstTouchModelTitle = sectionTitle(R.string.first_touch_model_title, 11f);
        addWithTopMargin(content, firstTouchModelTitle, dp(10), ViewGroup.LayoutParams.WRAP_CONTENT);

        firstTouchModelText = monospaceText(9f, COLOR_ACCENT);
        firstTouchModelText.setText(getString(
                R.string.first_touch_model_default_format,
                controller.firstTouchModelId()
        ));
        addWithTopMargin(content, firstTouchModelText, dp(5), ViewGroup.LayoutParams.WRAP_CONTENT);

        TextView modelTitle = sectionTitle(R.string.impact_model_title, 11f);
        addWithTopMargin(content, modelTitle, dp(10), ViewGroup.LayoutParams.WRAP_CONTENT);

        LinearLayout modelRow = new LinearLayout(this);
        modelRow.setOrientation(LinearLayout.HORIZONTAL);
        addWithTopMargin(content, modelRow, dp(6), dp(30));
        previousModelButton = panelButton(getString(R.string.model_previous), 9f);
        nextModelButton = panelButton(getString(R.string.model_next), 9f);
        modelNameText = bodyText(9f, COLOR_TEXT);
        modelNameText.setGravity(Gravity.CENTER);
        modelNameText.setTypeface(Typeface.DEFAULT_BOLD);
        modelNameText.setSingleLine(false);
        modelRow.addView(previousModelButton, new LinearLayout.LayoutParams(dp(42), ViewGroup.LayoutParams.MATCH_PARENT));
        LinearLayout.LayoutParams nameParams = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f);
        nameParams.leftMargin = dp(4);
        nameParams.rightMargin = dp(4);
        modelRow.addView(modelNameText, nameParams);
        modelRow.addView(nextModelButton, new LinearLayout.LayoutParams(dp(42), ViewGroup.LayoutParams.MATCH_PARENT));

        modelDetailText = monospaceText(9f, COLOR_MUTED);
        modelDetailText.setLineSpacing(dp(1), 1f);
        addWithTopMargin(content, modelDetailText, dp(6), ViewGroup.LayoutParams.WRAP_CONTENT);

        TextView normalizationTitle = sectionTitle(R.string.touch_normalization_title, 11f);
        addWithTopMargin(content, normalizationTitle, dp(10), ViewGroup.LayoutParams.WRAP_CONTENT);

        LinearLayout normalizationRow = new LinearLayout(this);
        normalizationRow.setOrientation(LinearLayout.HORIZONTAL);
        addWithTopMargin(content, normalizationRow, dp(6), dp(28));
        normScreenButton = panelButton(getString(R.string.touch_norm_screen_short), 8f);
        normCovarianceButton = panelButton(getString(R.string.touch_norm_cov_short), 8f);
        normBaselineButton = panelButton(getString(R.string.touch_norm_baseline_short), 8f);
        normalizationRow.addView(normScreenButton, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f));
        addRowGap(normalizationRow, dp(4));
        normalizationRow.addView(normCovarianceButton, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f));
        addRowGap(normalizationRow, dp(4));
        normalizationRow.addView(normBaselineButton, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f));

        normalizationDetailText = monospaceText(9f, COLOR_MUTED);
        normalizationDetailText.setLineSpacing(dp(1), 1f);
        addWithTopMargin(content, normalizationDetailText, dp(6), ViewGroup.LayoutParams.WRAP_CONTENT);
        normalizationDetailText.setVisibility(View.GONE);

        TextView collectorTitle = sectionTitle(R.string.collector_title, 11f);
        addWithTopMargin(content, collectorTitle, dp(10), ViewGroup.LayoutParams.WRAP_CONTENT);

        collectorUserNameInput = new EditText(this);
        collectorUserNameInput.setSingleLine(true);
        collectorUserNameInput.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_CAP_WORDS);
        collectorUserNameInput.setHint(R.string.collector_user_name_hint);
        collectorUserNameInput.setTextColor(COLOR_TEXT);
        collectorUserNameInput.setHintTextColor(COLOR_DIM);
        collectorUserNameInput.setTextSize(10f);
        collectorUserNameInput.setPadding(dp(8), 0, dp(8), 0);
        collectorUserNameInput.setBackground(panelBackground(COLOR_PANEL_BUTTON, dp(7), Color.argb(60, 255, 255, 255)));
        addWithTopMargin(content, collectorUserNameInput, dp(6), dp(34));

        collectorSwitchRow = new LinearLayout(this);
        collectorSwitchRow.setOrientation(LinearLayout.HORIZONTAL);
        collectorSwitchRow.setGravity(Gravity.CENTER_VERTICAL);
        collectorSwitchRow.setClickable(true);
        collectorSwitchRow.setBackground(panelBackground(COLOR_PANEL_BUTTON, dp(7), Color.argb(40, 255, 255, 255)));
        addWithTopMargin(content, collectorSwitchRow, dp(6), dp(36));
        collectorSwitch = new Switch(this);
        collectorSwitch.setText("");
        collectorSwitch.setPadding(0, 0, 0, 0);
        if (android.os.Build.VERSION.SDK_INT >= 21) {
            collectorSwitch.setThumbTintList(switchThumbTint());
            collectorSwitch.setTrackTintList(switchTrackTint());
        }
        LinearLayout.LayoutParams collectorSwitchParams = new LinearLayout.LayoutParams(dp(52), dp(28));
        collectorSwitchParams.leftMargin = dp(4);
        collectorSwitchRow.addView(collectorSwitch, collectorSwitchParams);

        LinearLayout motionRow = new LinearLayout(this);
        motionRow.setOrientation(LinearLayout.HORIZONTAL);
        addWithTopMargin(content, motionRow, dp(6), dp(28));
        handheldButton = panelButton(getString(R.string.motion_handheld_short), 8f);
        placedButton = panelButton(getString(R.string.motion_placed_short), 8f);
        walkingButton = panelButton(getString(R.string.motion_walking), 8f);
        motionRow.addView(handheldButton, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f));
        addRowGap(motionRow, dp(4));
        motionRow.addView(placedButton, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f));
        addRowGap(motionRow, dp(4));
        motionRow.addView(walkingButton, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f));

        LinearLayout strengthRow = new LinearLayout(this);
        strengthRow.setOrientation(LinearLayout.HORIZONTAL);
        addWithTopMargin(content, strengthRow, dp(6), dp(28));
        lightButton = panelButton(getString(R.string.touch_light), 10f);
        heavyButton = panelButton(getString(R.string.touch_heavy), 10f);
        strengthRow.addView(lightButton, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f));
        addRowGap(strengthRow, dp(5));
        strengthRow.addView(heavyButton, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f));

        LinearLayout fingerRow = new LinearLayout(this);
        fingerRow.setOrientation(LinearLayout.HORIZONTAL);
        addWithTopMargin(content, fingerRow, dp(6), dp(28));
        thumbButton = panelButton(getString(R.string.finger_thumb), 10f);
        indexButton = panelButton(getString(R.string.finger_index), 10f);
        fingerRow.addView(thumbButton, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f));
        addRowGap(fingerRow, dp(5));
        fingerRow.addView(indexButton, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f));

        collectorGuideText = bodyText(9f, COLOR_MUTED);
        collectorGuideText.setLineSpacing(dp(1), 1f);
        collectorGuideText.setMinLines(5);
        collectorGuideText.setMaxLines(5);
        collectorGuideText.setGravity(Gravity.TOP);
        addWithTopMargin(content, collectorGuideText, dp(7), dp(78));

        collectorStatusText = monospaceText(9f, COLOR_MUTED);
        collectorStatusText.setLineSpacing(dp(1), 1f);
        addWithTopMargin(content, collectorStatusText, dp(7), ViewGroup.LayoutParams.WRAP_CONTENT);

        latestRawSampleText = monospaceText(9f, COLOR_TEXT);
        latestRawSampleText.setLineSpacing(dp(1), 1f);
        latestRawSampleText.setText(R.string.raw_sample_waiting);
        latestRawSampleText.setPadding(dp(8), dp(7), dp(8), dp(7));
        latestRawSampleText.setBackground(panelBackground(Color.argb(32, 255, 255, 255), dp(7), Color.argb(70, 255, 255, 255)));
        addWithTopMargin(content, latestRawSampleText, dp(7), ViewGroup.LayoutParams.WRAP_CONTENT);

        resultText = monospaceText(9f, COLOR_ACCENT);
        resultText.setLineSpacing(dp(1), 1f);
        resultText.setMinLines(5);
        resultText.setMaxLines(5);
        resultText.setPadding(dp(8), dp(7), dp(8), dp(7));
        resultText.setBackground(panelBackground(Color.argb(30, 112, 217, 169), dp(7), Color.argb(80, 112, 217, 169)));
        addWithTopMargin(content, resultText, dp(7), dp(116));

        TextView fourClassRecallTitle = sectionTitle(R.string.four_class_recall_title, 11f);
        addWithTopMargin(content, fourClassRecallTitle, dp(9), ViewGroup.LayoutParams.WRAP_CONTENT);

        validationModeButton = panelButton(getString(R.string.validation_mode_on), 10f);
        addWithTopMargin(content, validationModeButton, dp(5), dp(28));
        styleModelButton(validationModeButton, true);

        fourClassRecallText = monospaceText(9f, COLOR_TEXT);
        fourClassRecallText.setLineSpacing(dp(1), 1f);
        fourClassRecallText.setMinLines(6);
        fourClassRecallText.setPadding(dp(8), dp(7), dp(8), dp(7));
        fourClassRecallText.setBackground(panelBackground(Color.argb(35, 112, 217, 169), dp(7), Color.argb(80, 112, 217, 169)));
        addWithTopMargin(content, fourClassRecallText, dp(5), ViewGroup.LayoutParams.WRAP_CONTENT);

        copyDatasetPathButton = panelButton(getString(R.string.collector_copy_path), 10f);
        addWithTopMargin(content, copyDatasetPathButton, dp(7), dp(26));

        panelToggleButton.setOnClickListener(v -> togglePanel());
        previousModelButton.setOnClickListener(v -> stepModel(-1));
        nextModelButton.setOnClickListener(v -> stepModel(1));
        collectorSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (!isUpdatingSwitch) {
                setCollectionEnabled(isChecked);
            }
        });
        collectorSwitchRow.setOnClickListener(v -> {
            boolean nextEnabled = !controller.isEnabled();
            setCollectionEnabled(nextEnabled);
        });
        thresholdSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                impactThreshold = 0.05f + progress / 1000f;
                updateThresholdText();
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });
        copyLogButton.setOnClickListener(v -> copyImpactLog());
        copyDatasetPathButton.setOnClickListener(v -> copyDatasetDirectory());
        normScreenButton.setOnClickListener(v -> setContactEncodingScheme(AispectContactEncodingProfile.Scheme.GAUSSIAN_MAP));
        normCovarianceButton.setOnClickListener(v -> setContactEncodingScheme(AispectContactEncodingProfile.Scheme.MULTI_CHANNEL));
        normBaselineButton.setOnClickListener(v -> setContactEncodingScheme(AispectContactEncodingProfile.Scheme.COVARIANCE_MATRIX));
        validationModeButton.setOnClickListener(v -> toggleValidationMode());
        lightButton.setOnClickListener(v -> setStrength(AispectModels.TouchStrength.LIGHT));
        heavyButton.setOnClickListener(v -> setStrength(AispectModels.TouchStrength.HEAVY));
        thumbButton.setOnClickListener(v -> setFingerType(AispectModels.FingerType.THUMB));
        indexButton.setOnClickListener(v -> setFingerType(AispectModels.FingerType.INDEX));
        handheldButton.setOnClickListener(v -> setMotionState(AispectModels.MotionState.HANDHELD_STATIC));
        placedButton.setOnClickListener(v -> setMotionState(AispectModels.MotionState.PLACED_STATIC));
        walkingButton.setOnClickListener(v -> setMotionState(AispectModels.MotionState.WALKING));
    }

    private boolean isInsidePanel(MotionEvent event) {
        if (impactPanel == null || impactPanel.getVisibility() != View.VISIBLE) {
            return false;
        }
        Rect bounds = new Rect();
        impactPanel.getGlobalVisibleRect(bounds);
        float rawX = event.getRawX();
        float rawY = event.getRawY();
        return bounds.contains(Math.round(rawX), Math.round(rawY));
    }

    private boolean handlePanelToggleTouch(MotionEvent event) {
        int action = event.getActionMasked();
        if (action == MotionEvent.ACTION_DOWN) {
            panelToggleTouchActive = isInsidePanelToggleHit(event);
            return panelToggleTouchActive;
        }
        if (!panelToggleTouchActive) {
            return false;
        }
        if (action == MotionEvent.ACTION_UP) {
            if (isInsidePanelToggleHit(event)) {
                togglePanel();
            }
            panelToggleTouchActive = false;
            return true;
        }
        if (action == MotionEvent.ACTION_CANCEL) {
            panelToggleTouchActive = false;
            return true;
        }
        return true;
    }

    private boolean isInsidePanelToggleHit(MotionEvent event) {
        if (impactPanel == null || impactPanel.getVisibility() != View.VISIBLE) {
            return false;
        }
        Rect bounds = new Rect();
        if (panelToggleButton == null || panelToggleButton.getVisibility() != View.VISIBLE) {
            impactPanel.getGlobalVisibleRect(bounds);
        } else {
            panelToggleButton.getGlobalVisibleRect(bounds);
        }
        int slop = dp(PANEL_TOGGLE_HIT_SLOP_DP);
        bounds.inset(-slop, -slop);
        return bounds.contains(Math.round(event.getRawX()), Math.round(event.getRawY()));
    }

    private void togglePanel() {
        isPanelCollapsed = !isPanelCollapsed;
        panelScrollView.setVisibility(isPanelCollapsed ? View.GONE : View.VISIBLE);
        panelToggleButton.setText(isPanelCollapsed ? R.string.impact_panel_expand : R.string.impact_panel_collapse);
        panelToggleButton.setTextSize(isPanelCollapsed ? 12f : 11f);
        impactPanel.setPadding(isPanelCollapsed ? 0 : dp(12), isPanelCollapsed ? 0 : dp(10), isPanelCollapsed ? 0 : dp(12), isPanelCollapsed ? 0 : dp(10));
        panelToggleButton.setBackground(panelBackground(isPanelCollapsed ? COLOR_PANEL_BUTTON : COLOR_BLUE, dp(isPanelCollapsed ? 17 : 8), Color.argb(70, 255, 255, 255)));
        impactPanel.setBackground(panelBackground(isPanelCollapsed ? COLOR_PANEL_COLLAPSED : COLOR_PANEL, dp(isPanelCollapsed ? 17 : 10), Color.argb(90, 255, 255, 255)));
        ViewGroup.LayoutParams params = impactPanel.getLayoutParams();
        params.width = isPanelCollapsed ? dp(PANEL_COLLAPSED_WIDTH_DP) : dp(PANEL_EXPANDED_WIDTH_DP);
        params.height = isPanelCollapsed ? dp(PANEL_COLLAPSED_HEIGHT_DP) : ViewGroup.LayoutParams.WRAP_CONTENT;
        impactPanel.setLayoutParams(params);
        ViewGroup.LayoutParams buttonParams = panelToggleButton.getLayoutParams();
        buttonParams.width = isPanelCollapsed ? ViewGroup.LayoutParams.MATCH_PARENT : dp(PANEL_TOGGLE_WIDTH_DP);
        buttonParams.height = isPanelCollapsed ? ViewGroup.LayoutParams.MATCH_PARENT : dp(PANEL_TOGGLE_HEIGHT_DP);
        panelToggleButton.setLayoutParams(buttonParams);
        impactPanel.bringToFront();
    }

    private void stepModel(int direction) {
        if (controller != null && controller.isTouchActive()) {
            return;
        }
        if (modelInfos == null || modelInfos.isEmpty()) {
            return;
        }
        int count = modelInfos.size();
        selectedModelIndex = (selectedModelIndex + direction + count) % count;
        lastFourClassRecallSignature = null;
        applyModelSelection();
    }

    private void toggleValidationMode() {
        validationModeEnabled = !validationModeEnabled;
        if (validationModeButton != null) {
            validationModeButton.setText(validationModeEnabled ? R.string.validation_mode_on : R.string.validation_mode_off);
            styleModelButton(validationModeButton, true);
        }
        if (fourClassRecallText != null) {
            fourClassRecallText.setVisibility(validationModeEnabled ? View.VISIBLE : View.GONE);
        }
        updateFourClassRecallBox(controller == null ? null : controller.status());
    }

    private void setStrength(AispectModels.TouchStrength strength) {
        if (controller.isEnabled()) {
            return;
        }
        selectedStrength = strength;
        applyLabel();
    }

    private void setMotionState(AispectModels.MotionState state) {
        if (controller.isEnabled()) {
            return;
        }
        selectedMotionState = state;
        applyLabel();
    }

    private void setFingerType(AispectModels.FingerType fingerType) {
        if (controller.isEnabled()) {
            return;
        }
        selectedFingerType = fingerType;
        applyLabel();
    }

    private void setContactEncodingScheme(AispectContactEncodingProfile.Scheme scheme) {
        if (controller.isEnabled()) {
            return;
        }
        controller.setSelectedContactEncodingScheme(scheme);
        styleNormalizationButtons(controller.status().selectedContactEncodingScheme, false);
    }

    private void applyModelSelection() {
        if (modelInfos == null || modelInfos.isEmpty()) {
            return;
        }
        AispectImpactCNNClassifier.ModelInfo info = modelInfos.get(selectedModelIndex);
        controller.setSelectedModelId(info.id);
        updateModelPanel(info);
    }

    private void applyLabel() {
        if (controller != null) {
            controller.setSelectedLabel(new AispectModels.DatasetLabel(selectedMotionState, selectedStrength, selectedFingerType));
        }
        styleSegmentButtons(controller != null && controller.isEnabled());
    }

    private void updateStatus(AispectCollectionController.Status status) {
        if (status == null || impactPanel == null) {
            return;
        }
        if (status.selectedModelInfo != null) {
            selectedModelIndex = modelIndexForId(status.selectedModelInfo.id);
            updateModelPanel(status.selectedModelInfo);
        }
        boolean locked = status.isLabelLocked;
        syncCollectorSwitch(status.enabled);
        collectorGuideText.setText(locked ? R.string.collector_guide_locked : R.string.collector_guide_text);
        collectorStatusText.setText(collectionStatusWithUser(status));
        impactLogText.setText(impactLog(status));
        normalizationDetailText.setText(normalizationText(status));
        updateThresholdText();
        updateRateText(status.latestSampleRateHz);
        updateAispectEventText(status.latestTouchEvent);
        homeView.setCollecting(status.enabled);
        homeView.setStatusText(status.enabled ? getString(R.string.fullscreen_capture_on) : getString(R.string.fullscreen_capture_off));
        styleSegmentButtons(locked);
        styleNormalizationButtons(status.selectedContactEncodingScheme, locked);
        if (status.latestPrediction != null) {
            resultText.setText(getString(
                    R.string.result_format,
                    status.latestPrediction.modelId,
                    predictionLabelTitle(status.latestPrediction.predictedLabel),
                    status.latestPrediction.predictedProbability,
                    labelFingerTitle(status.selectedLabel.fingerType),
                    labelStrengthTitle(status.selectedLabel.touchStrength),
                    labelMotionTitle(status.selectedLabel.motionState)
            ));
        } else {
            resultText.setText(getString(
                    R.string.result_unavailable,
                    labelFingerTitle(status.selectedLabel.fingerType),
                    labelStrengthTitle(status.selectedLabel.touchStrength),
                    labelMotionTitle(status.selectedLabel.motionState)
            ));
        }
        updateFourClassRecallBox(status);
    }

    private void updateModelPanel(AispectImpactCNNClassifier.ModelInfo info) {
        if (info == null) {
            modelNameText.setText(R.string.impact_model_empty);
            modelDetailText.setText(R.string.impact_model_empty);
            return;
        }
        modelNameText.setText(info.displayName);
        modelDetailText.setText(getString(
                R.string.impact_model_detail_format,
                frameRangeText(info),
                info.inputChannels,
                featureNamesText(info)
        ));
        boolean canSwitch = controller == null || !controller.isTouchActive();
        styleModelButton(previousModelButton, canSwitch);
        styleModelButton(nextModelButton, canSwitch);
    }

    private String collectionStatus(AispectCollectionController.Status status) {
        AispectDeviceCapabilityProfiler.Snapshot capability = status.capability;
        String level = capability == null ? "unknown" : capability.summaryKey();
        String eventText = collectorEventText(status.lastEventKey);
        String stateText = getString(status.enabled ? R.string.collector_state_on : R.string.collector_state_off);
        return getString(
                R.string.collector_status_format,
                stateText,
                labelMotionTitle(status.selectedLabel.motionState),
                labelStrengthTitle(status.selectedLabel.touchStrength),
                labelFingerTitle(status.selectedLabel.fingerType),
                status.acceptedCount,
                status.acceptedCount + status.rejectedCount,
                status.rejectedCount,
                status.acceptedCount,
                level,
                contactEncodingTitle(status.selectedContactEncodingScheme),
                eventText,
                status.datasetDirectoryPath
        );
    }

    private String collectionStatusWithUser(AispectCollectionController.Status status) {
        String userName = controller == null ? "" : controller.collectorUserName();
        return getString(R.string.collector_user_name_status, userName.isEmpty() ? "--" : userName)
                + "\n"
                + collectionStatus(status);
    }

    private String normalizationText(AispectCollectionController.Status status) {
        AispectContactEncodingProfile profile = status.latestContactEncodingProfile;
        if (profile == null) {
            return getString(
                    R.string.touch_normalization_idle_format,
                    contactEncodingTitle(status.selectedContactEncodingScheme)
            );
        }
        AispectContactEncodingProfile.Result selected = profile.selectedResult;
        return getString(
                R.string.touch_normalization_detail_format,
                contactEncodingTitle(profile.selectedScheme),
                selected.rawMajorPx,
                selected.rawMinorPx,
                selected.sigmaXPx,
                selected.sigmaYPx,
                selected.validMask,
                profile.gaussianMap.sigmaXPx,
                profile.multiChannel.rawMajorPx,
                profile.covarianceMatrix.covariance11
        );
    }

    private void updateLatestRawSample(AispectDataLogger.SavedRecord record) {
        if (latestRawSampleText == null || record == null || record.json == null) {
            return;
        }
        JSONObject json = record.json;
        JSONObject label = json.optJSONObject("label");
        JSONArray touchFrames = json.optJSONArray("touchFrames");
        JSONArray motionFrames = json.optJSONArray("frames");
        JSONObject motionSignal = json.optJSONObject("motionSignal");
        JSONObject lastTouch = lastObject(touchFrames);
        File file = record.file;
        latestRawSampleText.setText(getString(
                R.string.raw_sample_format,
                label == null ? "--" : label.optString("fingerType", "--"),
                label == null ? "--" : label.optString("touchStrength", "--"),
                label == null ? "--" : label.optString("motionState", "--"),
                touchFrames == null ? 0 : touchFrames.length(),
                motionFrames == null ? 0 : motionFrames.length(),
                json.optDouble("sampleRateHz", 0.0),
                lastTouch == null ? 0.0 : lastTouch.optDouble("touchMajor", 0.0),
                lastTouch == null ? 0.0 : lastTouch.optDouble("touchMinor", 0.0),
                lastTouch == null ? 0.0 : lastTouch.optDouble("pressure", 0.0),
                motionSignal == null ? 0 : motionSignal.optInt("frameCount", 0),
                file == null ? "--" : file.getName()
        ));
    }

    private static JSONObject lastObject(JSONArray array) {
        if (array == null || array.length() == 0) {
            return null;
        }
        return array.optJSONObject(array.length() - 1);
    }

    private String impactLog(AispectCollectionController.Status status) {
        if (status.latestImpactLog != null && !status.latestImpactLog.isEmpty()) {
            return status.latestImpactLog;
        }
        return getString(
                R.string.impact_log_format,
                status.lastEventDetail == null ? "--" : status.lastEventDetail,
                status.acceptedCount,
                status.rejectedCount
        );
    }

    private void syncCollectorSwitch(boolean enabled) {
        if (collectorSwitch == null) {
            return;
        }
        isUpdatingSwitch = true;
        collectorSwitch.setChecked(enabled);
        isUpdatingSwitch = false;
    }

    private void setCollectionEnabled(boolean enabled) {
        if (enabled) {
            String userName = collectorUserNameInput == null ? "" : collectorUserNameInput.getText().toString();
            controller.setCollectorUserName(userName);
            if (!controller.canStartCollection()) {
                if (collectorUserNameInput != null) {
                    collectorUserNameInput.setError(getString(R.string.collector_user_name_required));
                    collectorUserNameInput.requestFocus();
                }
                syncCollectorSwitch(false);
                return;
            }
        }
        controller.setEnabled(enabled);
        syncCollectorSwitch(controller.isEnabled());
    }

    private void updateThresholdText() {
        if (thresholdText != null) {
            thresholdText.setText(R.string.impact_threshold_format);
        }
    }

    private void updateRateText(double sampleRateHz) {
        if (rateText == null) {
            return;
        }
        if (sampleRateHz > 0) {
            rateText.setText(getString(R.string.impact_rate_format, sampleRateHz));
        } else {
            rateText.setText(R.string.impact_rate_idle);
        }
    }

    private void updateAispectEventText(AispectModels.TouchEvent event) {
        if (event == null || eventValueText == null) {
            return;
        }
        String liftText = event.liftOffset == null ? "--" : String.format(Locale.US, "%.2f", event.liftOffset);
        eventValueText.setText(getString(
                R.string.impact_event_format,
                touchKindText(event.kind),
                event.x,
                event.y,
                liftText,
                event.timestampSeconds
        ));
    }

    private void updateMotionEventText(MotionEvent event) {
        if (event == null || eventValueText == null) {
            return;
        }
        eventValueText.setText(getString(
                R.string.impact_event_format,
                motionActionText(event.getActionMasked()),
                event.getX(),
                event.getY(),
                "--",
                event.getEventTime() / 1000.0
        ));
    }

    private String touchKindText(AispectModels.TouchKind kind) {
        if (kind == AispectModels.TouchKind.LIGHT_TAP) {
            return getString(R.string.impact_event_light_tap);
        }
        if (kind == AispectModels.TouchKind.HEAVY_TAP) {
            return getString(R.string.impact_event_heavy_tap);
        }
        if (kind == AispectModels.TouchKind.LIGHT_HOLD) {
            return getString(R.string.impact_event_light_hold);
        }
        if (kind == AispectModels.TouchKind.HEAVY_HOLD) {
            return getString(R.string.impact_event_heavy_hold);
        }
        if (kind == AispectModels.TouchKind.HEAVY_PRESS) {
            return getString(R.string.impact_event_heavy_press);
        }
        if (kind == AispectModels.TouchKind.DRAG_START) {
            return getString(R.string.impact_event_drag_start);
        }
        if (kind == AispectModels.TouchKind.DRAG) {
            return getString(R.string.impact_event_drag);
        }
        if (kind == AispectModels.TouchKind.DRAG_END) {
            return getString(R.string.impact_event_drag_end);
        }
        if (kind == AispectModels.TouchKind.CANCEL) {
            return getString(R.string.impact_event_cancel);
        }
        return getString(R.string.impact_event_sample);
    }

    private String motionActionText(int action) {
        if (action == MotionEvent.ACTION_UP) {
            return getString(R.string.impact_event_action_up);
        }
        if (action == MotionEvent.ACTION_MOVE) {
            return getString(R.string.impact_event_sample);
        }
        if (action == MotionEvent.ACTION_CANCEL) {
            return getString(R.string.impact_event_cancel);
        }
        return getString(R.string.impact_event_sample);
    }

    private String collectorEventText(String key) {
        if ("collector_session_ready".equals(key)) {
            return getString(R.string.collector_session_ready);
        }
        if ("collector_session_stopped".equals(key)) {
            return getString(R.string.collector_session_stopped);
        }
        if ("collector_task_updated".equals(key)) {
            return getString(R.string.collector_task_updated);
        }
        if ("collector_waiting_for_touch_end".equals(key)) {
            return getString(R.string.collector_waiting_for_touch_end);
        }
        if ("collector_down_window_waiting".equals(key)) {
            return getString(R.string.collector_down_window_waiting);
        }
        if ("collector_diagnostic_down_window_waiting".equals(key)) {
            return getString(R.string.collector_diagnostic_down_window_waiting);
        }
        if ("collector_down_window_ready_model_pending".equals(key)) {
            return getString(R.string.collector_down_window_ready_model_pending);
        }
        if ("collector_diagnostic_ready_model_pending".equals(key)) {
            return getString(R.string.collector_diagnostic_ready_model_pending);
        }
        if ("collector_collecting_postroll".equals(key)) {
            return getString(R.string.collector_collecting_postroll);
        }
        if ("collector_event_saved".equals(key)) {
            return getString(R.string.collector_event_saved);
        }
        if ("collector_event_write_failed".equals(key)) {
            return getString(R.string.collector_event_write_failed);
        }
        return getString(R.string.collector_idle);
    }

    private void styleSegmentButtons(boolean locked) {
        if (collectorUserNameInput != null) {
            collectorUserNameInput.setEnabled(!locked);
        }
        styleButton(lightButton, selectedStrength == AispectModels.TouchStrength.LIGHT, locked);
        styleButton(heavyButton, selectedStrength == AispectModels.TouchStrength.HEAVY, locked);
        styleButton(thumbButton, selectedFingerType == AispectModels.FingerType.THUMB, locked);
        styleButton(indexButton, selectedFingerType == AispectModels.FingerType.INDEX, locked);
        styleButton(handheldButton, selectedMotionState == AispectModels.MotionState.HANDHELD_STATIC, locked);
        styleButton(placedButton, selectedMotionState == AispectModels.MotionState.PLACED_STATIC, locked);
        styleButton(walkingButton, selectedMotionState == AispectModels.MotionState.WALKING, locked);
    }

    private void styleNormalizationButtons(AispectContactEncodingProfile.Scheme scheme, boolean locked) {
        styleButton(normScreenButton, scheme == AispectContactEncodingProfile.Scheme.GAUSSIAN_MAP, locked);
        styleButton(normCovarianceButton, scheme == AispectContactEncodingProfile.Scheme.MULTI_CHANNEL, locked);
        styleButton(normBaselineButton, scheme == AispectContactEncodingProfile.Scheme.COVARIANCE_MATRIX, locked);
    }

    private void copyDatasetDirectory() {
        AispectCollectionController.Status status = controller.status();
        ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
        if (clipboard != null) {
            clipboard.setPrimaryClip(ClipData.newPlainText("datasetPath", status.datasetDirectoryPath));
        }
        collectorStatusText.setText(collectionStatusWithUser(status));
    }

    private void copyImpactLog() {
        ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
        if (clipboard != null) {
            clipboard.setPrimaryClip(ClipData.newPlainText("impactLog", impactLogText.getText()));
        }
    }

    private int defaultModelIndex(AispectImpactCNNClassifier.ModelInfo selectedInfo) {
        if (modelInfos == null || modelInfos.isEmpty() || selectedInfo == null) {
            return 0;
        }
        return modelIndexForId(selectedInfo.id);
    }

    private int modelIndexForId(String modelId) {
        if (modelInfos == null || modelInfos.isEmpty() || modelId == null) {
            return 0;
        }
        for (int i = 0; i < modelInfos.size(); i++) {
            if (modelId.equals(modelInfos.get(i).id)) {
                return i;
            }
        }
        return 0;
    }

    private String labelMotionTitle(AispectModels.MotionState state) {
        if (state == AispectModels.MotionState.PLACED_STATIC) {
            return getString(R.string.motion_placed_static);
        }
        if (state == AispectModels.MotionState.WALKING) {
            return getString(R.string.motion_walking);
        }
        return getString(R.string.motion_handheld_static);
    }

    private String labelStrengthTitle(AispectModels.TouchStrength strength) {
        return strength == AispectModels.TouchStrength.HEAVY ? getString(R.string.touch_heavy) : getString(R.string.touch_light);
    }

    private String labelFingerTitle(AispectModels.FingerType fingerType) {
        return fingerType == AispectModels.FingerType.INDEX ? getString(R.string.finger_index) : getString(R.string.finger_thumb);
    }

    private String predictionLabelTitle(String label) {
        if ("thumb_light".equals(label)) {
            return getString(R.string.finger_thumb) + "+" + getString(R.string.touch_light);
        }
        if ("thumb_heavy".equals(label)) {
            return getString(R.string.finger_thumb) + "+" + getString(R.string.touch_heavy);
        }
        if ("index_light".equals(label)) {
            return getString(R.string.finger_index) + "+" + getString(R.string.touch_light);
        }
        if ("index_heavy".equals(label)) {
            return getString(R.string.finger_index) + "+" + getString(R.string.touch_heavy);
        }
        if ("heavy".equals(label)) {
            return getString(R.string.touch_heavy);
        }
        if ("light".equals(label)) {
            return getString(R.string.touch_light);
        }
        return label == null || label.isEmpty() ? "--" : label;
    }

    private String predictionProbabilityText(AispectModels.ImpactPrediction prediction) {
        if (prediction == null || prediction.probabilities.length == 0 || prediction.labelOrder.length == 0) {
            return "--";
        }
        StringBuilder builder = new StringBuilder();
        int count = Math.min(prediction.probabilities.length, prediction.labelOrder.length);
        for (int i = 0; i < count; i++) {
            if (i > 0) {
                builder.append(" / ");
            }
            builder.append(predictionLabelTitle(prediction.labelOrder[i]));
            builder.append(' ');
            builder.append(String.format(Locale.US, "%.3f", prediction.probabilities[i]));
        }
        return builder.toString();
    }

    private void updateFourClassRecall(AispectCollectionController.Status status) {
        if (status == null || !validationModeEnabled || status.latestPrediction == null || status.selectedLabel == null) {
            return;
        }
        if (!isCompletedPredictionEvent(status.lastEventKey)) {
            return;
        }
        String truth = fourClassLabel(status.selectedLabel);
        String predicted = status.latestPrediction.predictedLabel;
        if (!fourClassRecallCounter.isFourClassLabel(truth) || !fourClassRecallCounter.isFourClassLabel(predicted)) {
            return;
        }
        String signature = truth
                + "|"
                + predicted
                + "|"
                + status.latestPrediction.modelId
                + "|"
                + status.latestPredictionSequence;
        if (signature.equals(lastFourClassRecallSignature)) {
            return;
        }
        lastFourClassRecallSignature = signature;
        fourClassRecallCounter.observe(truth, predicted);
    }

    private static boolean isCompletedPredictionEvent(String key) {
        return "collector_event_saved".equals(key);
    }

    private void updateFourClassRecallBox(AispectCollectionController.Status status) {
        if (fourClassRecallText == null) {
            return;
        }
        if (!validationModeEnabled) {
            fourClassRecallText.setVisibility(View.GONE);
            return;
        }
        fourClassRecallText.setVisibility(View.VISIBLE);
        String modelId = "--";
        String truth = status == null || status.selectedLabel == null ? "" : fourClassLabel(status.selectedLabel);
        String latest = getString(R.string.validation_waiting);
        String currentCombination = getString(R.string.validation_current_empty);
        if (status != null && status.latestPrediction != null) {
            modelId = status.latestPrediction.modelId;
            String predicted = status.latestPrediction.predictedLabel;
            boolean correct = truth.equals(predicted);
            latest = getString(
                    R.string.validation_latest_format,
                    predictionLabelTitle(truth),
                    predictionLabelTitle(predicted),
                    correct ? getString(R.string.validation_correct) : getString(R.string.validation_wrong),
                    status.latestPrediction.predictedProbability,
                    predictionProbabilityText(status.latestPrediction)
            );
        }
        if (fourClassRecallCounter.isFourClassLabel(truth)) {
            currentCombination = fourClassRecallCounter.line(truth, predictionLabelTitle(truth));
        }
        fourClassRecallText.setText(getString(
                R.string.validation_box_format,
                modelId,
                latest,
                currentCombination,
                fourClassRecallCounter.line("thumb_light", getString(R.string.finger_thumb) + "+" + getString(R.string.touch_light)),
                fourClassRecallCounter.line("thumb_heavy", getString(R.string.finger_thumb) + "+" + getString(R.string.touch_heavy)),
                fourClassRecallCounter.line("index_light", getString(R.string.finger_index) + "+" + getString(R.string.touch_light)),
                fourClassRecallCounter.line("index_heavy", getString(R.string.finger_index) + "+" + getString(R.string.touch_heavy)),
                fourClassRecallCounter.summary()
        ));
    }

    private static String fourClassLabel(AispectModels.DatasetLabel label) {
        String finger = label.fingerType == AispectModels.FingerType.INDEX ? "index" : "thumb";
        String strength = label.touchStrength == AispectModels.TouchStrength.HEAVY ? "heavy" : "light";
        return finger + "_" + strength;
    }

    private String contactEncodingTitle(AispectContactEncodingProfile.Scheme scheme) {
        if (scheme == AispectContactEncodingProfile.Scheme.GAUSSIAN_MAP) {
            return getString(R.string.touch_norm_screen);
        }
        if (scheme == AispectContactEncodingProfile.Scheme.MULTI_CHANNEL) {
            return getString(R.string.touch_norm_baseline);
        }
        return getString(R.string.touch_norm_covariance);
    }

    private static String frameRangeText(AispectImpactCNNClassifier.ModelInfo info) {
        if (info.frameIndices.length == 0) {
            return "";
        }
        int first = info.frameIndices[0];
        int last = info.frameIndices[info.frameIndices.length - 1];
        return first + "..." + last;
    }

    private static String featureNamesText(AispectImpactCNNClassifier.ModelInfo info) {
        if (info.featureNames.length == 0) {
            return info.id;
        }
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < info.featureNames.length; i++) {
            if (i > 0) {
                builder.append(", ");
            }
            builder.append(info.featureNames[i]);
        }
        return builder.toString();
    }

    private Button panelButton(String title, float textSize) {
        Button button = new Button(this);
        button.setText(title);
        button.setTextSize(textSize);
        button.setAllCaps(false);
        button.setGravity(Gravity.CENTER);
        button.setPadding(dp(4), 0, dp(4), 0);
        button.setMinWidth(0);
        button.setMinimumWidth(0);
        button.setMinHeight(0);
        button.setMinimumHeight(0);
        button.setTextColor(COLOR_TEXT);
        button.setBackground(panelBackground(COLOR_PANEL_BUTTON, dp(7), Color.argb(60, 255, 255, 255)));
        return button;
    }

    private TextView sectionTitle(int stringRes, float size) {
        TextView text = bodyText(size, COLOR_TEXT);
        text.setText(stringRes);
        text.setTypeface(Typeface.DEFAULT_BOLD);
        return text;
    }

    private TextView bodyText(float size, int color) {
        TextView text = new TextView(this);
        text.setTextSize(size);
        text.setTextColor(color);
        text.setIncludeFontPadding(true);
        return text;
    }

    private TextView monospaceText(float size, int color) {
        TextView text = bodyText(size, color);
        text.setTypeface(Typeface.MONOSPACE);
        return text;
    }

    private void styleModelButton(Button button, boolean enabled) {
        if (button == null) {
            return;
        }
        button.setEnabled(enabled);
        button.setTextColor(enabled ? COLOR_TEXT : COLOR_DIM);
        button.setBackground(panelBackground(enabled ? COLOR_PANEL_BUTTON : COLOR_PANEL_DISABLED, dp(7), Color.argb(60, 255, 255, 255)));
    }

    private void styleButton(Button button, boolean selected, boolean locked) {
        if (button == null) {
            return;
        }
        button.setEnabled(!locked);
        if (selected) {
            button.setTextColor(Color.rgb(8, 18, 14));
            button.setBackground(panelBackground(COLOR_ACCENT, dp(7), Color.argb(90, 255, 255, 255)));
        } else {
            button.setTextColor(locked ? COLOR_DIM : COLOR_TEXT);
            button.setBackground(panelBackground(locked ? COLOR_PANEL_DISABLED : COLOR_PANEL_BUTTON, dp(7), Color.argb(60, 255, 255, 255)));
        }
    }

    private ColorStateList switchThumbTint() {
        int[][] states = new int[][]{
                new int[]{android.R.attr.state_checked},
                new int[]{-android.R.attr.state_checked}
        };
        int[] colors = new int[]{COLOR_TEXT, Color.rgb(214, 214, 214)};
        return new ColorStateList(states, colors);
    }

    private ColorStateList switchTrackTint() {
        int[][] states = new int[][]{
                new int[]{android.R.attr.state_checked},
                new int[]{-android.R.attr.state_checked}
        };
        int[] colors = new int[]{COLOR_ACCENT, Color.argb(80, 255, 255, 255)};
        return new ColorStateList(states, colors);
    }

    private static GradientDrawable panelBackground(int color, int radius, int strokeColor) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(color);
        drawable.setCornerRadius(radius);
        drawable.setStroke(1, strokeColor);
        return drawable;
    }

    private LinearLayout.LayoutParams fullWidthWrap() {
        return new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    private void addWithTopMargin(LinearLayout parent, View child, int topMargin, int height) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, height);
        params.topMargin = topMargin;
        parent.addView(child, params);
    }

    private void addRowGap(LinearLayout row, int width) {
        View gap = new View(this);
        row.addView(gap, new LinearLayout.LayoutParams(width, 1));
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private static float positive(float value) {
        if (!Float.isFinite(value) || value <= 0f) {
            return 0f;
        }
        return value;
    }

    private static float normalized(float value, float low, float high) {
        if (!Float.isFinite(value) || high <= low) {
            return 0f;
        }
        return clampUnit((value - low) / (high - low));
    }

    private static float clampUnit(float value) {
        if (!Float.isFinite(value) || value < 0f) {
            return 0f;
        }
        if (value > 1f) {
            return 1f;
        }
        return value;
    }

    private static int blendColor(int from, int to, float fraction) {
        float clamped = clampUnit(fraction);
        int alpha = Math.round(Color.alpha(from) + (Color.alpha(to) - Color.alpha(from)) * clamped);
        int red = Math.round(Color.red(from) + (Color.red(to) - Color.red(from)) * clamped);
        int green = Math.round(Color.green(from) + (Color.green(to) - Color.green(from)) * clamped);
        int blue = Math.round(Color.blue(from) + (Color.blue(to) - Color.blue(from)) * clamped);
        return Color.argb(alpha, red, green, blue);
    }

    private static final class FourClassRecallCounter {
        private static final String[] LABELS = new String[]{"thumb_light", "thumb_heavy", "index_light", "index_heavy"};
        private final int[] correct = new int[LABELS.length];
        private final int[] total = new int[LABELS.length];

        void reset() {
            for (int i = 0; i < LABELS.length; i++) {
                correct[i] = 0;
                total[i] = 0;
            }
        }

        void observe(String truth, String predicted) {
            int index = labelIndex(truth);
            if (index < 0) {
                return;
            }
            total[index] += 1;
            if (truth.equals(predicted)) {
                correct[index] += 1;
            }
        }

        boolean isFourClassLabel(String label) {
            return labelIndex(label) >= 0;
        }

        String line(String label, String title) {
            int index = labelIndex(label);
            if (index < 0 || total[index] == 0) {
                return title + " --/-- --";
            }
            double recall = correct[index] / (double) total[index];
            return String.format(Locale.US, "%s %d/%d %.1f%%", title, correct[index], total[index], recall * 100.0);
        }

        String summary() {
            int observedClasses = 0;
            double totalRecall = 0;
            int allCorrect = 0;
            int allTotal = 0;
            for (int i = 0; i < LABELS.length; i++) {
                allCorrect += correct[i];
                allTotal += total[i];
                if (total[i] > 0) {
                    observedClasses += 1;
                    totalRecall += correct[i] / (double) total[i];
                }
            }
            if (allTotal == 0 || observedClasses == 0) {
                return "macro --  accuracy --";
            }
            double macro = totalRecall / observedClasses;
            double accuracy = allCorrect / (double) allTotal;
            return String.format(Locale.US, "macro %.1f%%  accuracy %d/%d %.1f%%", macro * 100.0, allCorrect, allTotal, accuracy * 100.0);
        }

        private int labelIndex(String label) {
            if (label == null) {
                return -1;
            }
            for (int i = 0; i < LABELS.length; i++) {
                if (LABELS[i].equals(label)) {
                    return i;
                }
            }
            return -1;
        }
    }

    private static final class DemoHomeView extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final RectF surfaceRect = new RectF();
        private float touchX = -1f;
        private float touchY = -1f;
        private float forceIntensity;
        private boolean collecting;
        private String statusText = "";

        DemoHomeView(Context context) {
            super(context);
            paint.setTypeface(Typeface.DEFAULT);
        }

        void setCollecting(boolean collecting) {
            this.collecting = collecting;
            invalidate();
        }

        void setStatusText(String statusText) {
            this.statusText = statusText;
            invalidate();
        }

        void updateTouch(MotionEvent event, float forceIntensity) {
            int action = event.getActionMasked();
            if (action == MotionEvent.ACTION_CANCEL || action == MotionEvent.ACTION_UP) {
                touchX = -1f;
                touchY = -1f;
                this.forceIntensity = 0f;
            } else {
                touchX = event.getX();
                touchY = event.getY();
                this.forceIntensity = clampUnit(forceIntensity);
            }
            invalidate();
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            int width = getWidth();
            int height = getHeight();
            drawBackground(canvas, width, height);
            drawTrainingSurface(canvas, width, height);
            drawStatus(canvas, width, height);
            drawTouch(canvas);
        }

        private void drawBackground(Canvas canvas, int width, int height) {
            LinearGradient gradient = new LinearGradient(
                    0f,
                    0f,
                    0f,
                    Math.max(1, height),
                    Color.rgb(4, 6, 7),
                    Color.rgb(0, 0, 0),
                    Shader.TileMode.CLAMP
            );
            paint.setStyle(Paint.Style.FILL);
            paint.setShader(gradient);
            canvas.drawRect(0, 0, width, height, paint);
            paint.setShader(null);
        }

        private void drawTrainingSurface(Canvas canvas, int width, int height) {
            float left = Math.max(18f, width * 0.045f);
            float top = Math.max(92f, height * 0.12f);
            float right = width - Math.max(18f, width * 0.045f);
            float bottom = height - Math.max(96f, height * 0.12f);
            surfaceRect.set(left, top, right, bottom);

            paint.setStyle(Paint.Style.FILL);
            paint.setColor(collecting ? Color.argb(24, 112, 217, 169) : Color.argb(18, 255, 255, 255));
            canvas.drawRoundRect(surfaceRect, 22f, 22f, paint);

            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(2f);
            paint.setColor(collecting ? Color.argb(145, 112, 217, 169) : Color.argb(62, 255, 255, 255));
            canvas.drawRoundRect(surfaceRect, 22f, 22f, paint);

            paint.setStrokeWidth(1f);
            paint.setColor(Color.argb(24, 255, 255, 255));
            float grid = 52f;
            for (float x = left + grid; x < right; x += grid) {
                canvas.drawLine(x, top, x, bottom, paint);
            }
            for (float y = top + grid; y < bottom; y += grid) {
                canvas.drawLine(left, y, right, y, paint);
            }

            float centerX = (left + right) * 0.5f;
            float centerY = (top + bottom) * 0.5f;
            paint.setColor(collecting ? Color.argb(85, 112, 217, 169) : Color.argb(42, 255, 255, 255));
            canvas.drawLine(centerX - 34f, centerY, centerX + 34f, centerY, paint);
            canvas.drawLine(centerX, centerY - 34f, centerX, centerY + 34f, paint);
            paint.setStyle(Paint.Style.STROKE);
            canvas.drawCircle(centerX, centerY, 48f, paint);

            paint.setStyle(Paint.Style.FILL);
            paint.setTypeface(Typeface.DEFAULT_BOLD);
            paint.setTextSize(18f);
            paint.setColor(Color.argb(210, 255, 255, 255));
            canvas.drawText("ZHIFAeyes", left, Math.max(42f, top - 38f), paint);
            paint.setTypeface(Typeface.DEFAULT);
        }

        private void drawStatus(Canvas canvas, int width, int height) {
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(collecting ? Color.argb(210, 112, 217, 169) : Color.argb(165, 255, 255, 255));
            paint.setTextSize(14f);
            paint.setTypeface(Typeface.DEFAULT_BOLD);
            canvas.drawText(statusText, 92f, height - 128f, paint);
            paint.setTypeface(Typeface.DEFAULT);
        }

        private void drawTouch(Canvas canvas) {
            if (touchX < 0 || touchY < 0) {
                return;
            }
            float hot = clampUnit(forceIntensity);
            int forceColor = Color.argb(220, 112, 217, 169);
            float outerRadius = 38f + 34f * hot;
            float innerRadius = 22f + 32f * hot;
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(3f + 5f * hot);
            paint.setColor(forceColor);
            canvas.drawCircle(touchX, touchY, outerRadius, paint);
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(Color.argb(70, 112, 217, 169));
            canvas.drawCircle(touchX, touchY, innerRadius, paint);
            if (hot > 0.42f) {
                paint.setStyle(Paint.Style.STROKE);
                paint.setStrokeWidth(2f);
                paint.setColor(Color.argb((int) (95 * hot), 112, 217, 169));
                canvas.drawCircle(touchX, touchY, outerRadius + 16f + hot * 20f, paint);
            }
        }

    }
}
