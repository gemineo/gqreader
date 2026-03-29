# Integration Testing Guide

This document describes how to run the integration tests and benchmarks against
the reference QR code dataset, how to interpret the results, and how to use
those results to measure the impact of pipeline improvements over time.

---

## Table of Contents

1. [Overview](#overview)
2. [Reference Dataset](#reference-dataset)
3. [Prerequisites](#prerequisites)
4. [Running the Tests](#running-the-tests)
5. [Understanding Test Output](#understanding-test-output)
6. [Understanding Benchmark Output](#understanding-benchmark-output)
7. [Tracking Pipeline Improvements](#tracking-pipeline-improvements)
8. [Current Baseline](#current-baseline)

---

## Overview

The integration tests exercise the full `DetectAndDecodeFile` pipeline — ONNX
inference, image preprocessing, perspective correction, and QR decoding — against
99 real-world QR code images with known ground-truth content.

Unlike the unit tests (which use mocks and synthetic images), these tests
require the ONNX model and produce **pass/fail results that directly reflect
the real-world accuracy of the pipeline**. Failed sub-tests are intentionally
left as failures: they serve as improvement targets and as a regression guard.

Three test functions are provided:

| Function | Purpose |
|---|---|
| `TestDatasetDetectionOnly` | Verifies YOLO detects at least one QR per image (detection layer only) |
| `TestDatasetDecodeAll` | Verifies decoded content matches ground truth (full pipeline) |
| `BenchmarkDataset` | Measures throughput with `DefaultConfig` |
| `BenchmarkDatasetMinimal` | Measures throughput with `MinimalConfig` (fast path) |

---

## Reference Dataset

The dataset lives at `testdata/Dataset-QR-2020/` and contains 99 JPEG images
across four categories, originally published by Björn Krüger (2020).

| Category | Files | Count | Characteristics |
|---|---|---|---|
| Synthetic | `QR1*.jpg` | 5 | Low-resolution, synthetic QR codes |
| Internet | `QR2*.jpg`, `QR3*.jpg` | 50 | Low-resolution images from the web, usually one QR per image |
| Manufacturing | `QR4*.jpg`, `QR6*.jpg` | 19 | High-resolution (2592×1920), many small QR codes per image, `QR6*` have damaged finder patterns |
| Sörös (2013) | `QR5*.jpg` | 25 | 640×480 px, one or two QR codes per image, non-blurred |

Total QR codes encoded across the dataset: **595**.

### annotations.txt

The file `testdata/Dataset-QR-2020/annotations.txt` maps each image stem to
the expected decoded content of all QR codes it contains:

```
QR1a=123456789|ABCDE|ABCDEFGH123456789|ABCDEFGHIJKLMNO
QR2a=https://goo.gl/HzULC2
QR4b=81HA001X4925|37HA001X4905|...  (50 entries for 50 QR codes)
```

**Format**: `<stem>=<content1>|<content2>|...`

- Each `|`-separated entry corresponds to one QR code in the image.
- Entries with identical content are **preserved** (not deduplicated). For
  example, `QR1e` contains 5 QR codes all encoding the same long text string,
  and `QR2b` contains 2 QR codes both encoding the same URL. The pipe count
  is the true QR code count in the image.

---

## Prerequisites

### 1. ONNX model

The integration tests load the ONNX model from `qrdet/models/qrdet-s.onnx`.
If the file is absent, `TestMain` prints a skip message and exits with code 0
(no test failure, CI-safe):

```
SKIP: QReader init failed: ...File doesn't exist
ok  github.com/gemineo/gqreader  0.000s
```

To prepare the model:

```bash
task setup-python   # one-time: creates .venv with ultralytics + onnx
task model          # downloads qrdet-s.pt and exports to qrdet/models/qrdet-s.onnx
```

If the ONNX file already exists, `task model` is a no-op (cached via Taskfile
`status`/`generates` guards).

### 2. System libraries

Same requirements as the main build: OpenCV and ONNX Runtime must be installed
(see the main `README.md` Installation section).

---

## Running the Tests

### Standard test run

```bash
task test-integration
```

This is equivalent to:

```bash
go test -v -tags integration -timeout 300s ./...
```

The `-timeout 300s` guard is necessary because the full dataset (99 images ×
full preprocessing pipeline) can take 1–3 minutes on a CPU-only machine.

### Run a single sub-test

Use the `-run` flag with the sub-test name to debug a specific image:

```bash
go test -v -tags integration -run TestDatasetDecodeAll/QR2b .
go test -v -tags integration -run TestDatasetDetectionOnly/QR5d1 .
```

### Run only one of the two test functions

```bash
go test -v -tags integration -run TestDatasetDetectionOnly -timeout 300s .
go test -v -tags integration -run TestDatasetDecodeAll    -timeout 300s .
```

### Unit tests are unaffected

The `//go:build integration` tag completely isolates the integration tests.
Running `task test` (or `go test ./...` without `-tags integration`) never
picks them up:

```bash
task test   # runs only unit tests — integration_test.go is invisible
```

---

## Understanding Test Output

### TestDatasetDetectionOnly

Each image is a sub-test. A sub-test passes if `DetectAndDecodeFile` returns
at least one result (regardless of whether it was decoded). This exercises the
ONNX inference layer in isolation.

```
=== RUN   TestDatasetDetectionOnly
=== RUN   TestDatasetDetectionOnly/QR1a
    --- PASS: TestDatasetDetectionOnly/QR1a (0.28s)
=== RUN   TestDatasetDetectionOnly/QR2b
    --- PASS: TestDatasetDetectionOnly/QR2b (0.32s)
...
--- PASS: TestDatasetDetectionOnly (26.10s)
```

A failure here means the YOLO model produced zero detections for that image —
a regression in the detection layer.

### TestDatasetDecodeAll

Each image is a sub-test. The test compares decoded content against
`annotations.txt` using **multiset matching** and a **per-image success
threshold**:

| Expected QR codes in image | Required minimum decoded |
|---|---|
| 1 | 1 (all) |
| 2–5 | 1 |
| 6 or more | ceil(n × 30%) |

The threshold for batch images (QR4, QR6 categories) acknowledges that
decoding all 50 tiny QR codes in a single 2592×1920 image is not a realistic
goal at this stage.

**Passing sub-test:**

```
=== RUN   TestDatasetDecodeAll/QR2a
    --- PASS: TestDatasetDecodeAll/QR2a (0.16s)
```

**Failing sub-test:**

```
=== RUN   TestDatasetDecodeAll/QR2b
    integration_test.go:200: QR2b: decoded 0/2 expected QR codes (threshold 1); 2 detections total
    --- FAIL: TestDatasetDecodeAll/QR2b (0.32s)
```

The failure message tells you:
- `decoded 0/2` — how many expected QR codes were matched vs. total expected
- `threshold 1` — the minimum required to pass
- `2 detections total` — the YOLO detector found the QR codes, but the decode
  step failed

**Summary block** (logged at the end of `TestDatasetDecodeAll`):

```
    integration_test.go:206: === Dataset Summary ===
    integration_test.go:207: Images tested:             99
    integration_test.go:208: Images meeting threshold:  88/99 (88.9%)
    integration_test.go:209: QR codes decoded:          XX/595 (XX.X%)
```

This summary is always printed, even when some sub-tests fail. Use it as the
primary metric when comparing pipeline versions.

### Reading the final result line

```
--- FAIL: TestDatasetDecodeAll (25.80s)
FAIL    github.com/gemineo/gqreader    51.646s
```

The overall test binary exits with code 1 as long as any sub-test fails. This
is expected and intentional — the failures are improvement targets, not bugs
in the test infrastructure.

---

## Understanding Benchmark Output

### Running benchmarks

```bash
task bench-integration
```

Equivalent to:

```bash
go test -tags integration -bench=. -benchmem -benchtime=3x -run=^$ ./...
```

- `-run=^$` suppresses all test functions so only benchmarks execute.
- `-benchtime=3x` runs each benchmark 3 times (three full passes over the
  dataset), giving a stable average.
- Override the iteration count: `task bench-integration BENCH_TIME=10s`

### Benchmark output format

```
BenchmarkDataset-10          3    38_450_201_000 ns/op    99 images/op    ...
BenchmarkDatasetMinimal-10   3    12_130_088_000 ns/op    99 images/op    ...
```

| Field | Meaning |
|---|---|
| `-10` suffix | Number of CPU threads used |
| `3` | Number of iterations (set by `-benchtime=3x`) |
| `ns/op` | Nanoseconds for one complete pass over all 99 images |
| `images/op` | Custom metric: number of images per pass (always 99) |

**Deriving per-image latency:**

```
per-image latency = ns/op ÷ images/op

Example:
  38_450_201_000 ns/op ÷ 99 images/op = ~388 ms/image  (DefaultConfig)
  12_130_088_000 ns/op ÷ 99 images/op = ~122 ms/image  (MinimalConfig)
```

`BenchmarkDatasetMinimal` uses `MinimalConfig` (no scaling, no perspective
correction). The gap between the two benchmarks quantifies the cost of the
full preprocessing pipeline.

### Saving benchmark results for comparison

Use `tee` to capture the output:

```bash
go test -tags integration -bench=. -benchmem -benchtime=3x -run=^$ . \
  | tee benchmarks/$(date +%Y-%m-%d)_baseline.txt
```

Then use the standard `benchstat` tool to compare two runs:

```bash
go install golang.org/x/perf/cmd/benchstat@latest
benchstat benchmarks/2026-03-29_before.txt benchmarks/2026-03-29_after.txt
```

`benchstat` output example:

```
name                  old time/op    new time/op    delta
BenchmarkDataset-10    38.5s ± 2%     31.2s ± 1%   -19.0%  (p=0.008)
BenchmarkDatasetMinimal-10  12.1s ± 1%  11.8s ± 2%  -2.5%  (p=0.210)
```

---

## Tracking Pipeline Improvements

The integration tests are designed to be a living benchmark. The workflow for
measuring an improvement is:

### Step 1 — Establish a baseline

Before making any changes, record the current state:

```bash
# Capture test results
go test -v -tags integration -timeout 300s -run TestDatasetDecodeAll . \
  | tee benchmarks/$(git rev-parse --short HEAD)_decode.txt

# Capture benchmark results
go test -tags integration -bench=. -benchmem -benchtime=5x -run=^$ . \
  | tee benchmarks/$(git rev-parse --short HEAD)_bench.txt
```

The key numbers to note from the summary block:
- **Images meeting threshold**: `88/99 (88.9%)` — primary accuracy metric
- **QR codes decoded**: overall decode rate across all 595 QR codes
- **Per-image latency** (from benchmark): `~388 ms/image` for `DefaultConfig`

### Step 2 — Make your change

Typical improvement areas and their expected test impact:

| Change | Expected impact on TestDatasetDecodeAll |
|---|---|
| Improve OpenCV decoder (e.g. add zbar) | More sub-tests pass; summary rate increases |
| Add new preprocessing strategy | Some currently failing images may start passing |
| Tune confidence threshold | May affect detection rate (TestDatasetDetectionOnly) |
| Optimize ONNX session (threads, batch) | No accuracy change; benchmark time decreases |
| Add perspective preprocessing variants | Some failing images may start passing |

### Step 3 — Re-run and compare

```bash
# Compare accuracy
go test -v -tags integration -timeout 300s -run TestDatasetDecodeAll . \
  | tee benchmarks/$(git rev-parse --short HEAD)_decode.txt

diff <(grep "FAIL\|PASS\|Summary" benchmarks/BEFORE_decode.txt) \
     <(grep "FAIL\|PASS\|Summary" benchmarks/AFTER_decode.txt)

# Compare throughput
go test -tags integration -bench=. -benchmem -benchtime=5x -run=^$ . \
  | tee benchmarks/$(git rev-parse --short HEAD)_bench.txt

benchstat benchmarks/BEFORE_bench.txt benchmarks/AFTER_bench.txt
```

### Step 4 — Interpret the diff

**A sub-test that was FAIL and is now PASS** means the pipeline improvement
successfully handled a previously undecodable image. Check that the image
category (QR2/QR3 vs QR5 vs QR4/QR6) matches your expectation — for example,
a better decoder should primarily improve QR2/QR3/QR5 images, not QR4/QR6
manufacturing images (which are limited by the density of QR codes rather
than decoder quality).

**A sub-test that was PASS and is now FAIL** is a regression. Investigate
before merging.

**The summary rate** (`Images meeting threshold`) is the single headline
number to report in commit messages and PR descriptions:

```
fix(qrdecode): replace OpenCV decoder with zbar

Decode accuracy: 88/99 → 94/99 images (+6)
Throughput: unchanged (decoder cost is negligible vs. ONNX inference)
```

### Interpreting the 11 known failing images

As of the baseline (2026-03-29), these images are detected but not decoded:

| Image | Category | QR codes | Likely cause |
|---|---|---|---|
| QR2b | Internet | 2 | Low-resolution, content partially obscured |
| QR2i | Internet | 1 | Low-resolution URL QR, tight crop |
| QR2j | Internet | 1 | Low-resolution, small module size |
| QR2p | Internet | 1 | Complex URL, high-density QR version |
| QR3c | Internet | 1 | Printed on curved surface |
| QR3e | Internet | 1 | Low contrast, perspective distortion |
| QR3h | Internet | 1 | Alphanumeric content, complex encoding |
| QR3k | Internet | 1 | Japanese site URL, encoding issues |
| QR3q | Internet | 1 | Compressed URL, marginal module size |
| QR5d1 | Sörös | 1 | Small QR at low resolution |
| QR5l3 | Sörös | 1 | QR partially out of frame |

All 11 are detected (YOLO confidence > threshold), so the failure is
entirely in the decode step. A better decoder (e.g. zbar) or additional
preprocessing variants would be the most likely path to resolving them.

---

## Current Baseline

Recorded on 2026-03-29, commit `1c10747`, Apple M3 Pro (CPU-only):

| Metric | Value |
|---|---|
| `TestDatasetDetectionOnly` | **99/99** (100%) |
| `TestDatasetDecodeAll` — images meeting threshold | **88/99** (88.9%) |
| `TestDatasetDecodeAll` — known failures | 11 images (all decode failures, not detection failures) |
| `BenchmarkDataset` per-image latency | ~388 ms/image (DefaultConfig) |
| `BenchmarkDatasetMinimal` per-image latency | ~122 ms/image (MinimalConfig) |
| Total test suite duration | ~52 seconds |

Store your own baselines in `benchmarks/` (directory is gitignored) to track
your changes locally without polluting the repository history.
