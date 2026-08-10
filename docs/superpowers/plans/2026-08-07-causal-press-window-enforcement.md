# Causal Press Window Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every down-anchor training and validation window reject samples lifted before its final post-down frame, then rank window candidates by a four-objective Pareto frontier.

**Architecture:** Enforce the invariant in the shared four-class sample loader so a command flag or cache cannot bypass it. The dense-window and schema-v5 runners will explicitly declare the invariant, verify both split counts, calculate output coverage from all candidate rows, and serialize Pareto fronts instead of a single weighted rank.

**Tech Stack:** Python standard library, existing PyTorch trainer, `unittest`, CSV, JSON.

---

### Task 1: Lock the Loader Invariant

**Files:**
- Modify: `android/analysis/tests/test_train_android_finger_four_class_cnn.py`
- Modify: `android/analysis/train_android_finger_four_class_cnn.py`

- [x] **Step 1: Write failing tests**

```python
def test_load_samples_rejects_post_lift_window_when_flag_is_omitted(self):
    samples = trainer.load_samples(manifest, ["delta"], list(range(-5, 10)), "error", "down")
    self.assertEqual(["complete"], [sample["id"] for sample in samples])

def test_load_samples_rejects_attempt_to_disable_press_active_filter(self):
    with self.assertRaisesRegex(RuntimeError, "press_active_filter_required"):
        trainer.load_samples(manifest, ["delta"], [0, 1], "error", "down", False)
```

- [x] **Step 2: Run the focused test and verify RED**

Run: `python -m unittest android.analysis.tests.test_train_android_finger_four_class_cnn`

Expected: FAIL because omitting the optional flag accepts the early-lift record and passing `False` disables the filter.

- [x] **Step 3: Implement the smallest loader contract**

```python
def load_samples(..., anchor_mode="down", require_press_active=True):
    if anchor_mode != "down":
        raise RuntimeError("press_phase_requires_down_anchor")
    if not require_press_active:
        raise RuntimeError("press_active_filter_required")
    # Skip before sample_matrix_for_anchor when liftReferenceFrameIndex < max(post frame).
```

Add a `causalPressActiveV1` cache-signature field and invalidate incompatible cached samples.

- [x] **Step 4: Run the focused test and verify GREEN**

Run: `python -m unittest android.analysis.tests.test_train_android_finger_four_class_cnn`

Expected: PASS.

### Task 2: Verify Coverage and Pareto Selection

**Files:**
- Modify: `android/analysis/tests/test_run_android_down_window_252_ablation.py`
- Modify: `android/analysis/run_android_down_window_252_ablation.py`

- [x] **Step 1: Write failing tests for split count enforcement and Pareto fronts**

```python
def test_completed_run_rejects_train_count_that_includes_post_lift_samples(self):
    with self.assertRaisesRegex(RuntimeError, "press_active_train_count_mismatch"):
        runner.completed_run_rows([job], {})

def test_aggregate_marks_only_non_dominated_windows_as_pareto_front_one(self):
    rows = [
        make_row("pre1_post2", accuracy=0.80, recall=0.60, coverage=1.00),
        make_row("pre5_post9", accuracy=0.79, recall=0.58, coverage=0.95),
        make_row("pre0_post1", accuracy=0.75, recall=0.55, coverage=1.00),
    ]
    summary = runner.aggregate_window_rows(rows)
    self.assertEqual(1, by_id(summary, "pre1_post2")["paretoFront"])
    self.assertGreater(by_id(summary, "pre5_post9")["paretoFront"], 1)
```

- [x] **Step 2: Run the focused test and verify RED**

Run: `python -m unittest android.analysis.tests.test_run_android_down_window_252_ablation`

Expected: FAIL because no train-count check or `paretoFront` exists.

- [x] **Step 3: Implement coverage and non-dominated sorting**

```python
def dominates(left, right):
    return (
        left["meanValidAccuracy"] >= right["meanValidAccuracy"]
        and left["worstValidMinRecall"] >= right["worstValidMinRecall"]
        and left["meanOutputCoverage"] >= right["meanOutputCoverage"]
        and left["postFrames"] <= right["postFrames"]
        and at_least_one_objective_is_strict(left, right)
    )
```

Emit train, validation, and total coverage; retain balanced accuracy as a diagnostic; write `paretoFront`, `waitFrames`, and `estimatedWaitMs` to CSV/JSON.

- [x] **Step 4: Run the focused test and verify GREEN**

Run: `python -m unittest android.analysis.tests.test_run_android_down_window_252_ablation`

Expected: PASS.

### Task 3: Keep Full-Factorial Commands Causal and Record the Contract

**Files:**
- Modify: `android/analysis/tests/test_run_schema5_full_factorial_ablation.py`
- Modify: `android/analysis/run_schema5_full_factorial_ablation.py`
- Modify: `FunctionAndDataStructureDescription.md`

- [x] **Step 1: Write a failing command-contract test**

```python
def test_training_command_requires_press_active_windows(self):
    command = runner.build_command(spec)
    self.assertIn("--require-press-active", command)
```

- [x] **Step 2: Run the focused test and verify RED**

Run: `python -m unittest android.analysis.tests.test_run_schema5_full_factorial_ablation`

Expected: FAIL because the full-factorial runner omits the flag.

- [x] **Step 3: Add the flag and update documentation**

Insert `--require-press-active` beside `--anchor-mode down`. Update the technical description with the mandatory loader invariant, cache version, count checks, coverage definition, and four Pareto objectives.

- [x] **Step 4: Run all causal-window tests and static checks**

Run: `python -m unittest android.analysis.tests.test_train_android_finger_four_class_cnn android.analysis.tests.test_run_android_down_window_252_ablation android.analysis.tests.test_run_schema5_full_factorial_ablation`

Expected: PASS with no failures.

Run: `python -m py_compile android/analysis/train_android_finger_four_class_cnn.py android/analysis/run_android_down_window_252_ablation.py android/analysis/run_schema5_full_factorial_ablation.py`

Expected: exit code `0`.

### Task 4: Rebuild Only Affected Summaries

**Files:**
- Output: `models/bbsiak_schema5_down_window_252_20260807/ablation_progress.json`
- Output: `models/bbsiak_schema5_down_window_252_20260807/window_run_summary.csv`
- Output: `models/bbsiak_schema5_down_window_252_20260807/window_candidate_summary.csv`

- [x] **Step 1: Generate summaries from completed reports without retraining**

Run: `python android/analysis/run_android_down_window_252_ablation.py --output-root models/bbsiak_schema5_down_window_252_20260807 --fold-root android/analysis/datasets/bbsiak_schema5_down_anchor_20260806/folds --dry-run`

Expected: summary files contain Pareto fields and no training command is run for completed jobs.

- [x] **Step 2: Validate the serialized contract**

Run: `python -c "import json; p=json.load(open('models/bbsiak_schema5_down_window_252_20260807/ablation_progress.json', encoding='utf-8')); assert p['requirePressActive'] is True; assert p['selectionPolicy']=='pareto_four_objective_v1'; assert p['paretoObjectives'][0]=='meanValidAccuracy'"`

Expected: exit code `0`.
