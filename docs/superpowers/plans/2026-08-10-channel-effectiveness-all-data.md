# Channel Effectiveness All-Data Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable all-data channel ablation runner without changing the strict cross-session baseline.

**Architecture:** A dedicated trainer loads all cleaned rows, selects a named channel contract, fits its scaler on all rows, and trains for a fixed budget. A runner launches isolated worker processes and writes a manifest plus ranked aggregate report.

**Tech Stack:** Python, PyTorch, unittest, JSON, CSV, ProcessPoolExecutor.

---

### Task 1: Define and test all-data channel contracts

- [ ] Add failing unit tests for contract channel selection and all-data loading.
- [ ] Implement the all-data trainer with fixed-epoch training and auditable reports.
- [ ] Run the focused trainer tests.

### Task 2: Define and test bounded parallel execution

- [ ] Add failing unit tests for nine experiment specifications and seven-worker bounds.
- [ ] Implement the parallel runner and aggregate ranking output.
- [ ] Run the focused runner tests.

### Task 3: Document, validate, and run

- [ ] Update FunctionAndDataStructureDescription.md with the all-data-only boundary.
- [ ] Run the analysis test suite.
- [ ] Snapshot the strict baseline configuration and launch the nine contracts.
