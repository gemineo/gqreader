# Integration Testing Guide

This document describes how to obtain the reference dataset, how to run the
integration tests and benchmarks, how to interpret the results, and how to
use those results to measure the impact of pipeline improvements over time.

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

The integration tests exercise the full `DetectAndDecodeFile` pipeline —
ONNX inference, image preprocessing, perspective correction, and QR decoding
— against 99 real-world QR code images with known ground-truth content.

Unlike the unit tests (which use mocks and synthetic images), these tests
require the ONNX model and a separately obtained dataset. They produce
**pass/fail results that directly reflect the real-world accuracy of the
pipeline**. Failed sub-tests are intentionally left as failures: they serve
as improvement targets and as a regression guard.

Three test functions are provided:

| Function | Purpose |
|---|---|
| `TestDatasetDetectionOnly` | Verifies YOLO detects at least one QR per image (detection layer only) |
| `TestDatasetDecodeAll` | Verifies decoded content matches ground truth (full pipeline) |
| `BenchmarkDataset` | Measures throughput with `DefaultConfig` |
| `BenchmarkDatasetMinimal` | Measures throughput with `MinimalConfig` (fast path) |

> For cross-tool comparisons (gqreader vs. Python QReader vs. zbar), see the
> companion project `gqreader-benchmark`.

---

## Reference Dataset

The dataset is **not included in this repository**. It must be obtained
separately and placed at the expected path before running the integration
tests.

### Dataset: QR Code Images (Dataset-QR-2020)

The dataset was assembled by Björn Krüger (2020) and contains 99 JPEG images
across four categories, totalling 595 QR codes:

| Category | Files | Images | QR codes | Characteristics |
|---|---|---|---|---|
| Synthetic | `QR1*.jpg` | 5 | 25 | Low-resolution synthetic QR codes |
| Internet | `QR2*.jpg`, `QR3*.jpg` | 42 | 76 | Low-resolution web images, usually one QR per image |
| Manufacturing | `QR4*.jpg` | 7 | 180 | High-resolution (2592×1920), many small QR codes per image |
| Manufacturing (damaged) | `QR6*.jpg` | 6 | 250 | Same as QR4, but QR codes have damaged finder patterns |
| Sörös (2013) | `QR5*.jpg` | 39 | 64 | 640×480 px, one or two QR codes per image, non-blurred |

The dataset is accompanied by an `annotations.txt` file that maps each image
to the expected decoded content of all QR codes it contains.

### Obtaining the Dataset

Download the archive and unzip it:

```
https://ram-verlag.org/wp-content/uploads/Dataset-QR-2020.zip
```

Reference publication: Krüger, B. (2020). *Location and Recognition of QR Codes in 2D Images*.
https://ram-verlag.org/wp-content/uploads/vol-8-022-1_location_and_recognition.pdf

Once downloaded, place the files so that the directory structure matches:

```
<repo-root>/
└── testdata/
    └── Dataset-QR-2020/
        ├── annotations.txt
        ├── QR1a.jpg
        ├── QR1b.jpg
        ...
        └── QR6f.jpg
```

The directory `testdata/Dataset-QR-2020/` is listed in `.gitignore` and will
never be committed to the repository.

### Using a Non-Default Dataset Path

If you store the dataset outside the repository (recommended for shared
machines or CI environments), set the `QREADER_DATASET_DIR` environment
variable:

```bash
QREADER_DATASET_DIR=/data/qr-datasets/Dataset-QR-2020 task test-integration
```

The model path can similarly be overridden:

```bash
QREADER_MODEL_PATH=/opt/models/qrdet-s.onnx task test-integration
```

Both variables default to paths relative to the repository root:

| Variable | Default |
|---|---|
| `QREADER_DATASET_DIR` | `testdata/Dataset-QR-2020` |
| `QREADER_MODEL_PATH` | `qrdet/models/qrdet-s.onnx` |

---

## Prerequisites

### 1. ONNX model

The integration tests load the ONNX model (default path:
`qrdet/models/qrdet-s.onnx`). If the file is absent, `TestMain` prints a
skip message and exits with code 0 (no test failure, CI-safe):

```
SKIP: QReader init failed: ...File doesn't exist
ok  github.com/gemineo/gqreader  0.000s
```

To prepare the model:

```bash
task setup-python   # one-time: creates .venv with ultralytics + onnx
task model          # downloads qrdet-s.pt and exports to qrdet/models/qrdet-s.onnx
```

If the ONNX file already exists, `task model` is a no-op (Taskfile
`status`/`generates` guards skip the export step).

### 2. Dataset

Place the dataset at `testdata/Dataset-QR-2020/` (or set
`QREADER_DATASET_DIR`). See [Reference Dataset](#reference-dataset) above.

### 3. System libraries

Same requirements as the main build: OpenCV and ONNX Runtime must be
installed. See the main `README.md` for installation instructions.

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
go test -v -tags integration -run TestDatasetDecodeAll     -timeout 300s .
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
at least one result (regardless of whether it was decoded). This exercises
the ONNX inference layer in isolation.

```
=== RUN   TestDatasetDetectionOnly
=== RUN   TestDatasetDetectionOnly/QR1a
    --- PASS: TestDatasetDetectionOnly/QR1a (0.28s)
=== RUN   TestDatasetDetectionOnly/QR2b
    --- PASS: TestDatasetDetectionOnly/QR2b (0.32s)
...
--- PASS: TestDatasetDetectionOnly (26.10s)
```

A failure here means the YOLO model produced zero detections — a regression
in the detection layer.

### TestDatasetDecodeAll

Each image is a sub-test. The test compares decoded content against
`annotations.txt` using **multiset matching** and a **per-image success
threshold**:

| Expected QR codes in image | Required minimum decoded |
|---|---|
| 1 | 1 (all) |
| 2–5 | 1 |
| 6 or more | ceil(n × 30%) |

The threshold for batch images (QR4, QR6) acknowledges that decoding all 50
tiny QR codes in a single 2592×1920 image is not a realistic goal at the
current pipeline stage.

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

The message tells you:
- `decoded 0/2` — matched vs. total expected
- `threshold 1` — minimum required to pass
- `2 detections total` — YOLO found the QR codes; the decode step failed

**Summary block** (always logged at the end of `TestDatasetDecodeAll`):

```
    integration_test.go:206: === Dataset Summary ===
    integration_test.go:207: Images tested:             99
    integration_test.go:208: Images meeting threshold:  88/99 (88.9%)
    integration_test.go:209: QR codes decoded:          XX/595 (XX.X%)
```

This summary is the primary metric to watch across pipeline versions.

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

**Deriving per-image latency:**

```
per-image latency = ns/op ÷ images/op

Example:
  38_450_201_000 ns ÷ 99 = ~388 ms/image  (DefaultConfig)
  12_130_088_000 ns ÷ 99 = ~122 ms/image  (MinimalConfig)
```

`BenchmarkDatasetMinimal` uses `MinimalConfig` (no scaling, no perspective
correction). The gap between the two benchmarks quantifies the cost of the
full preprocessing pipeline.

### Saving and comparing benchmark results

```bash
# Save a baseline
go test -tags integration -bench=. -benchmem -benchtime=5x -run=^$ . \
  | tee benchmarks/$(git rev-parse --short HEAD)_bench.txt

# Compare two runs with benchstat
go install golang.org/x/perf/cmd/benchstat@latest
benchstat benchmarks/BEFORE_bench.txt benchmarks/AFTER_bench.txt
```

`benchstat` output:

```
name                     old time/op    new time/op    delta
BenchmarkDataset-10       38.5s ± 2%     31.2s ± 1%   -19.0%  (p=0.008)
BenchmarkDatasetMinimal-10  12.1s ± 1%  11.8s ± 2%    -2.5%  (p=0.210)
```

The `benchmarks/` directory is gitignored — store runs locally without
polluting the repository.

---

## Tracking Pipeline Improvements

### Workflow

**Step 1 — Record the baseline before making changes:**

```bash
go test -v -tags integration -timeout 300s -run TestDatasetDecodeAll . \
  | tee benchmarks/$(git rev-parse --short HEAD)_decode.txt

go test -tags integration -bench=. -benchmem -benchtime=5x -run=^$ . \
  | tee benchmarks/$(git rev-parse --short HEAD)_bench.txt
```

**Step 2 — Make your change.**

**Step 3 — Re-run and diff:**

```bash
# Accuracy diff
diff <(grep "FAIL\|Summary" benchmarks/BEFORE_decode.txt) \
     <(grep "FAIL\|Summary" benchmarks/AFTER_decode.txt)

# Throughput diff
benchstat benchmarks/BEFORE_bench.txt benchmarks/AFTER_bench.txt
```

**Step 4 — Report in the commit message:**

```
fix(qrdecode): replace OpenCV decoder with zbar

Decode accuracy: 88/99 → 94/99 images (+6)
Throughput: unchanged (~388 ms/image)
```

### What each failure category tells you

| Failure pattern | Likely cause | Relevant improvement |
|---|---|---|
| Detected but not decoded (0 decoded, N detections > 0) | Decoder limitation | Better decoder (zbar), more preprocessing variants |
| Not detected (0 detections) | YOLO threshold too high, very small QR | Lower confidence threshold, rescaling |
| Partial decode (k decoded, k < expected) | Dense batch images, overlapping QR codes | Improved NMS, multi-crop strategy |

### Typical improvement areas and their expected impact

| Change | Expected impact on `TestDatasetDecodeAll` |
|---|---|
| Add zbar decoder | More sub-tests pass; summary rate increases |
| New preprocessing strategy | Some currently failing images start passing |
| Tune confidence threshold | May affect `TestDatasetDetectionOnly` |
| Optimize ONNX session (threads, batch) | No accuracy change; benchmark time decreases |
| Improve perspective correction | Some failing images may start passing |

---

## Current Baseline

Recorded on 2026-03-29, commit `1c10747`, Apple M3 Pro (CPU-only):

| Metric | Value |
|---|---|
| `TestDatasetDetectionOnly` | **99/99** (100%) |
| `TestDatasetDecodeAll` — images meeting threshold | **88/99** (88.9%) |
| Known failing images | 11 (all decode failures, not detection failures) |
| `BenchmarkDataset` per-image latency | ~388 ms/image (DefaultConfig) |
| `BenchmarkDatasetMinimal` per-image latency | ~122 ms/image (MinimalConfig) |
| Total integration test suite duration | ~52 seconds |

### Known failing images

All 11 failures are detected by YOLO but not decoded by OpenCV:

| Image | Expected QR codes | Likely cause |
|---|---|---|
| QR2b | 2 | Low-resolution, small module size |
| QR2i | 1 | Low-resolution URL QR, tight crop |
| QR2j | 1 | Low-resolution, small module size |
| QR2p | 1 | Complex URL, high-density QR version |
| QR3c | 1 | Printed on curved surface |
| QR3e | 1 | Low contrast, perspective distortion |
| QR3h | 1 | Short alphanumeric code, marginal quality |
| QR3k | 1 | Japanese site URL, encoding edge case |
| QR3q | 1 | Compressed URL, small module size |
| QR5d1 | 1 | Small QR at 640×480 resolution |
| QR5l3 | 1 | QR partially out of frame |

These are the primary targets for decoder improvements or additional
preprocessing strategies.
