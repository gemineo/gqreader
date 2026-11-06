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
| **DefaultConfig + CoreML ALL** | **~50** | **3.2× faster** | Recommended on Apple Silicon — ZBar+OpenCV fallback |
| DefaultConfig + CPU | ~136 | 1.2× faster | ZBar+OpenCV fallback, all platforms |
| FastConfig + CoreML ALL | ~37 | 4.3× faster | ZBar only, no fallback |
| FastConfig + CPU | ~147 | 1.1× faster | ZBar only, portable |
| MinimalConfig (no preproc) | 154 | — | No scaling / perspective (−41pp accuracy) |
| CoreML CPU-only | 210 | — | CoreML graph, CPU execution |
| CoreML CPUAndGPU | 183 | — | GPU offload, no ANE |
| **Python QReader** | **159** | baseline | ultralytics + Apple MPS + pyzbar |

**Key insight — ZBar unlocks CoreML's full benefit**: with OpenCV as the primary decoder, CoreML
saves only ~98 ms/img. With ZBar as the primary decoder, CoreML saves ~110 ms/img more because
ONNX inference becomes the bottleneck again. `DefaultConfig` uses ZBar as primary (fast path) and
only falls back to OpenCV when ZBar fails — so it gets most of ZBar's speed advantage while
recovering the few QR codes that ZBar mis-encodes.

---

## Accuracy Summary (Dataset-QR-2020, 99 images, 595 QR codes)

| Configuration | Images | QR codes |
|--------------|--------|----------|
| **gQReader DefaultConfig (tiled detection, ZBar → OpenCV → structured), 2026-11-06** | **99/99 (100%)** | **595/595 (100%)** |
| gQReader DefaultConfig without tiled detection | 99/99 (100%) | 568/595 (95.5%) |
| gQReader DefaultConfig without Phase 3 (ZBar → OpenCV), 2026-03-31 | 98/99 (99.0%) | 560/595 (94.1%) |
| gQReader (ZBar decoder only) | 95/99 (96.0%) | 554/595 (93.1%) |
| gQReader (OpenCV decoder only) | 89/99 (89.9%) | 531/595 (89.2%) |
| Python QReader 3.16 | 97/99 (98.0%) | 582/595 (97.8%) |
| zbar standalone (no YOLO) | 86/99 (86.9%) | 241/595 (40.5%) |

The 2026-11-06 figures come from four changes: Phase 3 structured decoding (`QR3q`, and the
damaged-finder codes on `QR6a`, `QR6b`, `QR6f`), tiled detection for images larger than
1280 px (the 26 small symbols of the 2592×1920 `QR4`/`QR6` sheets that the 640 px letterbox
could not resolve), a rewritten quadrilateral fit (a third of the dataset quads had corners far
outside their bbox) and a dedupe that keeps distinct detections sharing a payload (`QR1e`,
`QR2b`, `QR2u`). Per-image time on Apple M3 Pro with CoreML: median 160 ms, mean 384 ms,
p95 1.9 s (sheets of 50 labels).

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

> `zbar` is required for `DefaultConfig()` and `FastConfig()`. If you cannot install it, use
> `MinimalConfig()` or replace the decoder factory with `qrdecode.NewOpenCVDecoder`. You will
> lose accuracy and the decoding step will be 3–5× slower.

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
    // DefaultConfig: ZBar primary + OpenCV fallback — highest accuracy (recommended)
    // FastConfig: ZBar only, no fallback — maximum throughput
    qr, err := gqreader.New(gqreader.DefaultConfig())
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
cfg := gqreader.DefaultConfig()
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
├── config.go               # Config, DefaultConfig, FastConfig, MinimalConfig
├── pipeline.go             # Brute-force decode pipeline (Phase 1 + 2)
├── structured.go           # Phase 3: structured decoding of undecoded detections
├── result.go               # QRResult type
├── crop.go                 # BBox and perspective crop helpers
│
├── qrstruct/               # Structured QR decoder (pure Go, no CGO)
│   ├── tables.go           # ISO 18004 version / EC block tables
│   ├── layout.go           # Function modules, format info, data ordering
│   ├── rs.go               # Reed-Solomon decoder with erasures
│   ├── grid.go             # Homography + cylinder grid model, sampler, Otsu
│   ├── fit.go              # Known-module objective, global search, descent
│   ├── decode.go           # Decode(): identify version/orientation, fit, decode
│   └── testdata/           # Encoder matrices (v1-40) and warped renders
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

**Workaround**: `DefaultConfig()` (ZBar primary + OpenCV fallback) recovers this automatically —
the OpenCV fallback decoder handles the encoding correctly whenever ZBar fails or returns an
unexpected result. Only `FastConfig()` (ZBar only) is affected.

### Phase 3: Structured Decoding

As of 2026-11-06 (DefaultConfig: ZBar → OpenCV → structured decoder), gQReader scores
**99/99** images vs Python QReader's 97/99. The last image, `QR3q` (a QR code printed on a
cup, ~4 px per module, cylindrical distortion), fails every off-the-shelf decoder: ZBar,
OpenCV (standard, curved and ArUco detectors), zxing-cpp, rqrr, rxing and WeChatQR.

Those decoders *locate* the symbol but sample it through a grid derived from a few anchor
points (finder and alignment pattern centres). At 4 px per module that grid is 1-3 px off,
which corrupts more codewords than Reed-Solomon can repair. Phase 3 (`qrstruct`) takes the
opposite route and never hands an image back to a black-box decoder:

1. The sampling grid is parametrised (a homography from the module lattice to the image,
   optionally composed with a tilted-axis cylinder model).
2. Its parameters are fitted by maximising agreement on the modules whose value is known
   before decoding: finder patterns, separators, timing, alignment, the dark module, the
   BCH-protected format information and, for versions 7+, the version information. That is
   roughly 40 % of a small symbol, so the symbol calibrates its own grid.
3. Version and orientation are unknown on entry and identified by running the fit for every
   candidate (versions 1-10 × 4 rotations) and keeping the best-scoring ones.
4. The module centres are sampled, binarised (Otsu), unmasked, de-interleaved and corrected
   with Reed-Solomon, erasing progressively more of the least reliable codewords when needed.

Phase 3 only runs on detections that Phase 1, Phase 2 and the decoder fallback chain all
failed to decode, so it cannot regress an image that already passes. It is pure Go (no CGO)
and costs 30-80 ms per undecoded detection on Apple M3 Pro. On its own, without any
preprocessing or ZBar, it decodes 17 of 17 dataset images it was tried on, including the
bottle and cup images. Disable it with `Config.WithStructured(false)`; `FastConfig()` does
not enable it.

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

### libzbar not found

```bash
# macOS
brew install zbar

# Linux
sudo apt-get install libzbar-dev
```

If you cannot install libzbar, use `MinimalConfig()` or build a custom config with the OpenCV
decoder factory: `gqreader.DefaultConfig().WithDecoderFactory(func() (qrdecode.Decoder, error) { return qrdecode.NewOpenCVDecoder() })`.
You will lose accuracy and decoding will be 3–5× slower.

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
