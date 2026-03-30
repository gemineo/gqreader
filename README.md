# gQReader — Go QR Code Detection & Decoding Library

[![Go Reference](https://pkg.go.dev/badge/github.com/gemineo/gqreader.svg)](https://pkg.go.dev/github.com/gemineo/gqreader)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

> ⚠️ **Work in progress — API unstable**
>
> This project is in early development. The API, package structure, configuration types, and
> binary formats are all subject to **breaking changes at any time and without notice**.
> Do not use in production. Feedback and contributions are welcome, but expect churn.

A Go port of the Python [QReader](https://github.com/Eric-Canas/QReader/) library: high-accuracy QR code detection and decoding using YOLOv8, without Docker or Python runtime dependencies.

---

## Why gQReader?

### vs. Python QReader

| Metric | gQReader (ZBar + CoreML) | Python QReader |
|--------|--------------------------|----------------|
| Images decoded | 95/99 **(96.0%)** | 96/99 (97.0%) |
| QR codes decoded | 554/595 **(93.1%)** | 581/595 (97.6%) |
| Throughput | **37 ms/img** | 159 ms/img |
| Speedup | **4.3× faster** | baseline |
| Runtime dependencies | OpenCV, ONNX Runtime, libzbar | Python, PyTorch, ultralytics, pyzbar |
| Deployment | Single binary | Python virtual environment |

*Measured on Apple Silicon M3 Pro, Dataset-QR-2020 (99 images, 595 QR codes), 10 benchmark iterations. See [Benchmark Notes](#benchmark-notes) for methodology and platform limitations.*

### vs. zbar Alone (No YOLO Detection)

| Metric | gQReader (ZBar + YOLO) | zbar standalone |
|--------|------------------------|-----------------|
| Images decoded | **96.0%** | 86.9% |
| QR codes decoded | **93.1%** | 40.5% |
| Damaged/partial QR (QR6 dataset, 6 images) | **6/6** | 0/6 |

The YOLO preprocessing step is essential. Damaged or partially occluded QR codes (broken finder
patterns, low contrast, skewed perspective) are detected reliably only because YOLO localises the
QR region first, even when the barcode is not directly scannable. Without YOLO, zbar cannot find
them at all.

### Benchmark Notes

> **Platform**: All benchmarks were run on **Apple Silicon M3 Pro** only. Benchmarks on other
> architectures (Linux x86-64 / AMD64, NVIDIA GPU, Intel, ARM Linux) have not yet been performed
> due to lack of available hardware. Results on those platforms will be added in a future update.

**Methodology**: each configuration was run for **10 complete passes** over the full 99-image
dataset (990 image processing calls per configuration). Measurements were taken in isolation, one
configuration at a time, to avoid contention. Raw Go benchmark output:

```
BenchmarkDataset-11              10   27915457754 ns/op   99.00 images/op   # CPU + OpenCV
BenchmarkDatasetZBar-11          10   14503552275 ns/op   99.00 images/op   # CPU + ZBar
BenchmarkDatasetCoreML-11        10   18166656908 ns/op   99.00 images/op   # CoreML ALL + OpenCV
BenchmarkDatasetZBarCoreML-11    10    3705074292 ns/op   99.00 images/op   # CoreML ALL + ZBar
BenchmarkDatasetMinimal-11       10   15262272729 ns/op   99.00 images/op   # Minimal (no preproc)
BenchmarkDatasetCoreMLCPU-11     10   20795136238 ns/op   99.00 images/op   # CoreML CPU-only
BenchmarkDatasetCoreMLGPU-11     10   18150495267 ns/op   99.00 images/op   # CoreML CPUAndGPU
```

Python QReader reference (159 ms/img) is from a separate run and is not part of the Go benchmark
suite. Python was measured under similar but not identical conditions.

---

## Performance Summary (Apple M3 Pro)

| Configuration | ms/img | vs. Python QReader | Notes |
|--------------|--------|--------------------|-------|
| **ZBarConfig + CoreML ALL** | **37** | **4.3× faster** | Recommended on Apple Silicon |
| ZBarConfig + CPU | 147 | 1.1× faster | Portable, all platforms |
| DefaultConfig + CoreML ALL | 184 | — | OpenCV decoder, CoreML inference |
| DefaultConfig + CPU | 282 | 1.8× slower | OpenCV decoder, CPU inference |
| MinimalConfig (no preproc) | 154 | — | No scaling / perspective (−6pp accuracy) |
| CoreML CPU-only | 210 | — | CoreML graph, CPU execution |
| CoreML CPUAndGPU | 183 | — | GPU offload, no ANE |
| **Python QReader** | **159** | baseline | ultralytics + Apple MPS + pyzbar |

**Key insight — ZBar unlocks CoreML's full benefit**: with OpenCV as the decoder, CoreML saves
98 ms/img (282 → 184 ms). With ZBar as the decoder, CoreML saves 110 ms/img more (147 → 37 ms).
When OpenCV's multi-scale decode loop is the dominant cost, CoreML's inference speedup is masked.
Switching to ZBar makes ONNX inference the bottleneck again, so CoreML provides a much larger
relative gain.

---

## Accuracy Summary (Dataset-QR-2020, 99 images, 595 QR codes)

| Configuration | Images | QR codes |
|--------------|--------|----------|
| gQReader (ZBar decoder) | 95/99 (96.0%) | 554/595 (93.1%) |
| gQReader (OpenCV decoder) | 89/99 (89.9%) | 531/595 (89.2%) |
| Python QReader | 96/99 (97.0%) | 581/595 (97.6%) |
| zbar standalone (no YOLO) | 86/99 (86.9%) | 241/595 (40.5%) |

The accuracy is **decoder-independent**: CoreML, CPU, and all other execution providers produce
identical detections. The provider selection only affects inference speed.

---

## What Is QReader?

gQReader combines three stages:

1. **Detection** — A YOLOv8-seg neural network (QRdet) localises every QR code in the image with bounding box, segmentation polygon, and confidence score.
2. **Preprocessing** — The detected crop is subjected to a brute-force pipeline of scale factors and image transformations (grayscale, contrast, thresholding, sharpening, inversion) to maximise decode success on degraded or difficult images.
3. **Decoding** — The prepared crop is decoded by a backend (OpenCV or ZBar).

---

## Quick Start

### Prerequisites

| Dependency | Purpose | Required? |
|-----------|---------|-----------|
| Go 1.24+ | Build | Yes |
| OpenCV | Image processing (GoCV) | Yes |
| ONNX Runtime | YOLOv8 inference | Yes |
| libzbar | Fast QR decoding | Optional (strongly recommended) |
| Python 3.8+ | One-time model export | Yes (one-time setup only) |
| Task | Build automation | Recommended |

### Installation

#### 1. Install System Libraries

**macOS (Homebrew):**

```bash
brew install opencv onnxruntime zbar
```

> `zbar` is optional. If you skip it, use `DefaultConfig()` (OpenCV decoder) instead of
> `ZBarConfig()`. You will lose ~6 percentage points of accuracy and the decoding step will be
> 3–5× slower.

**Linux (Debian/Ubuntu):**

```bash
sudo apt-get install libopencv-dev libzbar-dev
```

For ONNX Runtime on Linux, download the prebuilt package from the
[official releases](https://github.com/microsoft/onnxruntime/releases) and install to `/usr/local`
or set `CGO_LDFLAGS` to point to the library directory.

#### 2. Install Task (build automation)

```bash
# macOS
brew install go-task

# Linux / other
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin
```

#### 3. Clone and Download Go Dependencies

```bash
git clone https://github.com/gemineo/gqreader.git
cd gqreader
task deps
```

#### 4. Export the ONNX Model (One-Time Setup)

The YOLOv8 model is distributed as a PyTorch `.pt` file and must be converted to ONNX format
once. This requires Python and the Ultralytics package (for model export only — Python is **not**
required at runtime after this step).

```bash
# Step 1: Create a Python virtual environment and install conversion tools
task setup-python
# Installs: ultralytics, onnx, requests, tqdm into .venv/

# Step 2: Download the QRdet weights and export to ONNX
task model
# Downloads: models/qrdet-s.pt (~24 MB) from GitHub releases
# Exports:   qrdet/models/qrdet-s.onnx (~47 MB)
```

This step:
- Requires an internet connection
- Runs only once; subsequent builds reuse the cached ONNX file
- Creates a Python virtual environment in `.venv/` that can be removed afterwards

If you already have a Python environment with `ultralytics` installed, you can export manually:

```bash
python scripts/export_model.py --output qrdet/models/qrdet-s.onnx
```

#### 5. Build

```bash
# Development build: loads model from file at startup (~4.5 MB binary)
task build

# Production build: model embedded in binary via go:embed (~52 MB, no external files)
task build-embedded
```

---

## Using gQReader

### As a Library

#### End-to-End Detection and Decoding

```go
package main

import (
    "fmt"
    "log"
    "github.com/gemineo/gqreader"
)

func main() {
    // ZBarConfig: recommended — uses libzbar for 3-5× faster decoding
    // DefaultConfig: portable — uses OpenCV (no libzbar required)
    qr, err := gqreader.New(gqreader.ZBarConfig())
    if err != nil {
        log.Fatal(err)
    }
    defer qr.Close()

    results, err := qr.DetectAndDecodeFile("photo.jpg")
    if err != nil {
        log.Fatal(err)
    }

    for _, r := range results {
        if r.Decoded {
            fmt.Printf("QR: %s (confidence: %.0f%%, strategy: %s)\n",
                r.Content, r.Detection.GetConfidence()*100, r.Strategy)
        } else {
            fmt.Printf("detected but not decoded (confidence: %.0f%%)\n",
                r.Detection.GetConfidence()*100)
        }
    }
}
```

#### Detection Only (Without Decoding)

```go
detector, err := qrdet.NewDetector(qrdet.DefaultConfig())
if err != nil {
    log.Fatal(err)
}
defer detector.Close()

detections, err := detector.DetectFile("photo.jpg")
for i, det := range detections {
    fmt.Printf("QR %d: conf=%.0f%% bbox=%v center=%v\n",
        i+1, det.GetConfidence()*100, det.BBox(), det.Center())
}
```

#### Hardware Acceleration (Apple Silicon, NVIDIA, Intel)

```go
import "github.com/gemineo/gqreader/qrdet"

// Apple Silicon — CoreML (GPU + Neural Engine), best on macOS
cfg := gqreader.ZBarConfig()
cfg.DetectionConfig = qrdet.DefaultConfig().WithProvider(qrdet.CoreMLProviderConfig())

// NVIDIA — CUDA
cfg.DetectionConfig = qrdet.DefaultConfig().WithProvider(qrdet.CUDAProviderConfig())

// Portable: read provider from the QREADER_PROVIDER environment variable
//   QREADER_PROVIDER=coreml ./bin/qrdetect -input photo.jpg
//   QREADER_PROVIDER=cuda   ./bin/qrdetect -input photo.jpg
cfg.DetectionConfig = qrdet.DefaultConfig().WithProvider(qrdet.ProviderFromEnv())
```

---

## CLI Reference

```
qrdetect -input <image> [options]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-input` | string | **required** | Path to input image |
| `-output` | string | `output.png` | Path for annotated output image |
| `-model` | string | (embedded) | Path to ONNX model (dev build only) |
| `-conf` | float | `0.5` | Confidence threshold (0.0–1.0) |
| `-nms` | float | `0.3` | NMS IoU threshold (0.0–1.0) |
| `-json` | bool | `false` | Output detections as JSON |
| `-v` | bool | `false` | Verbose/debug output |

```bash
# Annotated image output
./bin/qrdetect -input photo.jpg -output result.png

# JSON for scripting
./bin/qrdetect -input photo.jpg -json | jq '.[] | .confidence'

# CoreML acceleration
QREADER_PROVIDER=coreml ./bin/qrdetect -input photo.jpg -v
```

---

## Development Workflow

### Task Commands

```bash
task setup-python       # Create Python venv for model conversion (once)
task model              # Download and export ONNX model (once)
task deps               # Download Go module dependencies

task build              # Build CLI (dev mode, external model)
task build-embedded     # Build CLI with embedded model

task test               # Run unit tests
task test-coverage      # Run tests with HTML coverage report
task test-integration   # Accuracy test against full dataset (requires model)
task bench              # Unit-level benchmarks
task bench-integration  # Dataset throughput benchmarks

# Hardware provider benchmarks (Apple Silicon)
task bench-providers                    # All provider variants
QREADER_PROVIDER=coreml task bench-env # CoreML only

task fmt                # gofmt
task lint               # golangci-lint
task clean              # Remove build artefacts
task clean-all          # Remove artefacts + models + Python venv
```

### Project Structure

```
gqreader/
├── qreader.go              # Main API: QReader, DetectAndDecode*
├── config.go               # Config, DefaultConfig, ZBarConfig, MinimalConfig
├── pipeline.go             # Brute-force decode pipeline
├── result.go               # QRResult type
├── crop.go                 # BBox and perspective crop helpers
│
├── qrdet/                  # YOLOv8 detection
│   ├── detector.go
│   ├── session.go          # ONNX Runtime wrapper
│   ├── preprocess.go       # Image → model tensor
│   ├── postprocess.go      # Tensor → detections (NMS)
│   ├── provider.go         # Hardware EP config (CoreML, CUDA, …)
│   ├── types.go
│   ├── quadrilateral.go
│   ├── image.go            # ImageSource (file / bytes / Mat)
│   ├── model.go            # Embedded model (embed_model build tag)
│   └── models/qrdet-s.onnx # Generated by `task model`
│
├── qrdecode/               # QR decoder backends
│   ├── decoder.go          # Decoder interface
│   ├── opencv.go           # OpenCV QRCodeDetector
│   └── zbar_decoder.go     # libzbar (3-5× faster)
│
├── imgprep/                # Preprocessing strategies
├── internal/               # geometry, mathutil
├── cmd/qrdetect/           # CLI
├── scripts/export_model.py # PyTorch → ONNX (Python, one-time)
├── models/qrdet-s.pt       # Downloaded weights
└── testdata/Dataset-QR-2020/
```

---

## Hardware Acceleration

| Platform | Provider | `QREADER_PROVIDER` | Notes |
|----------|----------|--------------------|-------|
| macOS (Apple Silicon) | `CoreMLProviderConfig()` | `coreml` | GPU + Neural Engine |
| Linux (NVIDIA GPU) | `CUDAProviderConfig()` | `cuda` | CUDA toolkit required |
| Linux (NVIDIA, max perf) | `TensorRTProviderConfig()` | `tensorrt` | TensorRT required |
| Linux/Windows (Intel) | `OpenVINOProviderConfig()` | `openvino` | OpenVINO runtime required |
| Windows (any GPU) | `DirectMLProviderConfig()` | `directml` | Windows only |
| All platforms | `CPUProviderConfig()` | `cpu` | Safe portable fallback |

ONNX Runtime falls back to CPU silently if the requested provider is unavailable.

**CoreML compute units** (macOS):

| Value | What runs where | ms/img (M3 Pro) |
|-------|----------------|-----------------|
| `"ALL"` (default) | CPU + GPU + Neural Engine | 184 ms (OpenCV), 37 ms (ZBar) |
| `"CPUAndGPU"` | CPU + GPU, no ANE | 183 ms |
| `"CPUOnly"` | CoreML graph, CPU execution | 210 ms |

On this M3 Pro, `"ALL"` and `"CPUAndGPU"` are equivalent — the ANE contribution is negligible
for this model size. Results may differ on other Apple Silicon generations.

---

## Differences from Python QReader

### ZBar ISO-8859-1 / Shift-JIS Encoding Edge Case

When a QR code stores ISO-8859-1 (Latin-1) bytes that happen to form valid Shift-JIS sequences,
ZBar may misidentify the encoding and return garbled text.

**Example**: the Latin-1 character `á` (0xE1) followed by `l` (0x6C) forms the Shift-JIS
character `獺`. ZBar returns `獺`; the correct value is `ál`.

Python QReader works around this via a Big5 re-encoding trick that recovers *some* Latin-1 values
but is not a general solution. gQReader documents this as a known limitation of libzbar itself.

**Affected image in Dataset-QR-2020**: `QR2c`.

**Workaround**: if your QR codes contain Central or Eastern European Latin-1 text, use
`DefaultConfig()` (OpenCV decoder) instead of `ZBarConfig()`. The OpenCV decoder handles encoding
correctly.

### Remaining Accuracy Gap

As of 2026-03-30 (ZBar decoder), gQReader scores 95/99 images vs Python QReader's 96/99:

| Image | gQReader (ZBar) | Python QReader | Root cause |
|-------|----------------|----------------|-----------|
| QR2c | ❌ | ✅ | ZBar ISO-8859-1 encoding bug (see above) |
| QR3e | ❌ | ❌ | Both fail |
| QR3k | ❌ | ❌ | Both fail |
| QR3q | ❌ | ❌ | Both fail |

Three of the four failing images also fail Python QReader. The only image exclusively failing
gQReader is `QR2c` (ZBar encoding issue), making the **effective recoverable gap one image**.

---

## Build Modes

| Mode | Command | Binary size | Model source |
|------|---------|-------------|-------------|
| Development | `task build` | ~4.5 MB | `qrdet/models/qrdet-s.onnx` (external file) |
| Production | `task build-embedded` | ~52 MB | Embedded via `go:embed` |

---

## Troubleshooting

### "ONNX model not found"

```bash
task model
```

### OpenCV not found (macOS)

```bash
export CGO_CFLAGS="-I/opt/homebrew/opt/opencv/include/opencv4"
export CGO_LDFLAGS="-L/opt/homebrew/opt/opencv/lib"
task build
```

### ONNX Runtime not found

```bash
brew reinstall onnxruntime
# or
export CGO_LDFLAGS="-L/opt/homebrew/opt/onnxruntime/lib"
```

### libzbar not found (ZBarConfig build errors)

```bash
# macOS
brew install zbar

# Linux
sudo apt-get install libzbar-dev
```

If you do not want to install libzbar, replace `ZBarConfig()` with `DefaultConfig()` (OpenCV
decoder) in your code. You will lose ~6 percentage points of accuracy and decoding will be slower.

### Python environment issues (model export)

```bash
rm -rf .venv
task setup-python
task model
```

---

## License and Attribution

- **Original QReader**: https://github.com/Eric-Canas/QReader/
- **QRdet (detection model)**: https://github.com/Eric-Canas/qrdet/
- **ONNX Runtime**: https://onnxruntime.ai/
- **GoCV (OpenCV bindings)**: https://gocv.io/

Released under GNU AGPL v3 to comply with the Ultralytics YOLOv8 licence (via the QRdet model
adaptation by Eric Cañas). This may change in the future depending on licence clarification.

See `CLAUDE.md` for technical roadmap, model format documentation, and extension paths for
multi-format barcode support.
