# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QReader_PoC is a Go port of the Python QReader library (https://github.com/Eric-Canas/QReader/). The goal is to eliminate Docker/Python dependencies by converting both QRdet (YOLOv8-based QR code detection) and QReader (image preprocessing) into pure Go with acceptable CGO dependencies.

### Architecture Components

The project converts two Python libraries into Go:

1. **QReader (Image Preprocessing)**: Classical computer vision operations to optimize images for QR code reading
   - Grayscale conversion
   - Contrast adjustment (global and adaptive)
   - Binarization (fixed and adaptive thresholding)
   - Sharpening
   - Resizing (upscale/downscale)
   - Inversion (black/white swap)
   - Cropping and padding

2. **QRdet (QR Detection)**: YOLOv8-based CNN model for detecting and segmenting QR codes in images
   - Uses a trained YOLOv8 model (.pt format in Python) for QR code localization
   - Does NOT decode QR codes, only detects/segments the probable QR zone
   - Requires conversion from PyTorch model to ONNX format for Go usage

### Target Architecture

```
┌─────────────────────────────┐
│           Go API            │
├─────────────────────────────┤
│  Image preprocessing (Go)   │  ← GoCV / OpenCV
├─────────────────────────────┤
│  QR detection (YOLO)        │  ← CGO + ONNX Runtime
│   - bounding boxes          │
│   - confidence              │
├─────────────────────────────┤
│  Crop / refine ROI (Go)     │
├─────────────────────────────┤
│  QR decoding                │  ← zbar / OpenCV
└─────────────────────────────┘
```

## Technical Stack

### Primary Dependencies (to be implemented)

- **GoCV**: Go bindings for OpenCV - handles all classical image processing operations
- **ONNX Runtime (via CGO)**: Inference engine for the converted YOLOv8 model
  - Python YOLOv8 .pt model must be exported to .onnx format first
  - ONNX Runtime has stable C API with Go bindings
  - No PyTorch required at runtime
- **zbar (via CGO)** or **OpenCV QRCodeDetector**: For actual QR code decoding (planned for later phase)

### CGO Acceptability

This project accepts CGO dependencies for:
- ONNX Runtime (YOLOv8 inference)
- OpenCV (via GoCV)
- zbar (QR decoding)

CGO is preferred over Docker/Python microservices for performance and deployment simplicity.

**The design goal is to eliminate the Python/Docker runtime, not to eliminate CGO.** A binary
that links against `libonnxruntime` and `libopencv` is far simpler to deploy than a Python
environment with torch, ultralytics, qrdet, etc. Users install two system libraries
(`brew install onnxruntime opencv`) and ship a single binary.

### Why ONNX Runtime and Not PyTorch

PyTorch has no native Go bindings. The only options for running a `.pt` model in Go are:

| Option | Verdict |
|--------|---------|
| Native PyTorch (Python subprocess) | Reintroduces the Python dependency — defeats the purpose |
| LibTorch (PyTorch C++ frontend via CGO) | 2 GB+ install, complex CGO integration, Anthropic explicitly ruled it out |
| ONNX Runtime via CGO | ~50 MB, stable C API, clean Go wrapper (`yalue/onnxruntime_go`), no PyTorch at runtime |

ONNX is the only realistic choice. The model is exported once (Python, one-time step with
`task model`) and ONNX Runtime handles inference in Go with no Python dependency.

## Implementation Strategy

### Phase 1: Image Preprocessing (Low complexity)
Port QReader's image transformation pipeline using GoCV:
- Generate multiple image variants from a single source image
- Each variant applies different transformations (grayscale, threshold, resize, etc.)
- Create API: `func GenerateVariants(src gocv.Mat) []PreprocessedImage`

**Validation**: Compare output images from Python QReader vs Go implementation (visual or histogram comparison)

### Phase 2: YOLOv8 Detection (High complexity)
1. Export the YOLOv8 model from QRdet (.pt → .onnx)
2. Integrate ONNX Runtime with Go via CGO
3. Implement YOLOv8 post-processing in Go:
   - Confidence filtering
   - Non-Maximum Suppression (NMS)
   - Bounding box to ROI conversion

**Critical**: Start with model export and validate basic ONNX inference before porting full logic

### Phase 3: QR Decoding (Future)
Options:
- zbar via CGO (maximum fidelity with Python's pyzbar)
- OpenCV QRCodeDetector (zero external deps, potentially less robust)

## GoCV Operation Mapping

| QReader Operation | GoCV Equivalent |
|-------------------|-----------------|
| Grayscale | `gocv.CvtColor` |
| Resize | `gocv.Resize` |
| Threshold | `gocv.Threshold` |
| Adaptive Threshold | `gocv.AdaptiveThreshold` |
| Blur | `gocv.GaussianBlur` |
| Sharpen | Custom convolution |
| Invert | `gocv.BitwiseNot` |

## Key Constraints

- **No Python/Docker in production**: The entire pipeline must run as native Go binary
- **No ML training**: Only inference of pre-trained YOLOv8 model
- **No PyTorch at runtime**: Model must be converted to ONNX format
- **Avoid LibTorch**: Too heavy, complex CGO integration, unnecessary with ONNX

## Code Standards

### Language Policy
- **All code comments, documentation, commit messages, and code identifiers MUST be in English**
- Chat/conversation may be in French, but all written artifacts in the codebase use English
- This applies to:
  - Go code comments
  - Function/variable/type names
  - README files and markdown documentation
  - Git commit messages
  - API documentation
  - Test descriptions

## Development Workflow

### Setting up the project (to be defined)
```bash
# Install GoCV (requires OpenCV)
# Install ONNX Runtime
# Export YOLOv8 model to ONNX (Python step, one-time)
```

### Testing strategy
- Unit tests for each image preprocessing operation
- Integration tests comparing Go output with Python QReader output
- Visual regression tests for image variants
- Benchmark tests for performance comparison

## Reference Materials

- **Original Python Library**: https://github.com/Eric-Canas/QReader/
- **Requirements Document**: `Requirements/chatGPT_requirements_2025-12-19.txt`
- **YOLOv8 Documentation**: Ultralytics YOLOv8
- **ONNX Runtime**: https://onnxruntime.ai/
- **GoCV**: https://gocv.io/

## Estimated Effort (from requirements doc)

| Task | Estimated Time |
|------|----------------|
| Image preprocessing (Go) | 1-2 days |
| Export YOLOv8 → ONNX | 1 day |
| ONNX Runtime Go integration | 2-3 days |
| YOLO post-processing | 1-2 days |
| zbar integration | 1 day |
| Testing/tuning | 3-5 days |
| **Total** | ~2 weeks |

## IMPORTANT: Model Adaptation for QRdet (Single-Class Detection)

### ⚠️ Critical Observation: QRdet Output Format

**QRdet is single-class detection, NOT multi-class YOLO**

The QRdet model (from GitHub releases v2.0) has been specifically trained for QR code detection only. This affects the ONNX output tensor format:

#### Output Channel Differences

| Model Type | Channel Layout | Total Channels | Notes |
|-----------|----------------|-----------------|-------|
| **QRdet (single-class)** | 4 bbox + 1 confidence + 32 masks | **37** | Current implementation |
| Standard YOLOv8-seg | 4 bbox + 80 COCO classes + 32 masks | **116** | Generic multi-class YOLO |
| Generic YOLOv8-det | 4 bbox + 80 COCO classes | **84** | Detection only (no segmentation) |

#### QRdet ONNX Model Specifications

```
Model: qrdet-s.pt → qrdet-s.onnx (from v2.0 releases)
Input:  [1, 3, 640, 640]
Output0: [1, 37, 8400]  ← channels are: cx, cy, w, h, confidence, mask_coeff[0-31]
Output1: [1, 32, 160, 160] ← segmentation prototypes

Confidence is at channel index 4 (not index 5 like multi-class YOLO)
Mask coefficients occupy channels 5-36 (indices, not 85-116)
```

### Why This Matters for Future Multi-Format Support

**Current implementation assumes single-class detection.**

If you want to extend QReader to detect **multiple barcode formats** (QR, DataMatrix, Code128, I2of5, etc.), you MUST:

1. **Train or obtain a multi-class model**, NOT use the current QRdet model
2. **Rewrite the output parsing logic** to handle variable class counts
3. **Update constants in `qrdet/types.go`**:
   - `modelOutputChannels` will be different
   - Class extraction logic will change
   - Confidence score indexing will change

### Example: Multi-Class Model Expected Format

If training a YOLOv8-seg model with N barcode classes:
```
Output0: [1, 4 + N + 32, 8400]  ← (bbox + N-class scores + mask coefficients)

For N=5 classes (QR, DataMatrix, Code128, I2of5, Codabar):
Output0: [1, 41, 8400]  ← 4 + 5 + 32
```

### Migration Path: Single-Class → Multi-Class

To support multiple barcode types, you would need to:

1. **Model selection**:
   - Option A: Train your own YOLOv8-seg model with multiple barcode classes
   - Option B: Use separate single-class models per format (current approach, less efficient)

2. **Code refactoring** (estimate: 1-2 days):
   - Parameterize `modelOutputChannels` and `modelNumClasses`
   - Update `parseOutput0()` to dynamically extract class scores
   - Update `buildDetection()` to populate class information
   - Update domain types (`Detection` struct) to include detected class

3. **Post-processing adjustments**:
   - NMS may need class-aware filtering (current: agnostic NMS only)
   - Confidence thresholds may vary per class
   - ROI extraction logic remains the same

### Recommendations for Future Development

**If staying QR-only**:
- Current implementation is optimal and efficient
- No changes needed

**If planning multi-format support**:
- Document this constraint early
- Plan model retraining or multi-model strategy
- Keep `qrdet/` modular to allow class configuration
- Consider versioning the Go API separately from model format

## Critical Success Factors

1. **Exact parameter preservation**: Maintain same kernel sizes, thresholds, and scaling factors as Python version
2. **Image variant equivalence**: Generated variants must match Python output for regression prevention
3. **Model export validation**: Verify ONNX model produces same detections as PyTorch original
4. **Post-processing fidelity**: NMS and bbox handling must match QRdet behavior
5. **Model format awareness**: Keep track of single-class vs multi-class assumptions for future extensibility

## Benchmark Baseline (2026-03-29, initial)

Measured on Dataset-QR-2020 (99 images, 595 QR codes). Full data in
`../gqreader-benchmark/results/2026-03-29/`.

| Tool | Images pass% | QR codes % | Avg ms/img |
|------|-------------|-----------|-----------|
| gqreader DefaultConfig (OpenCV decoder) | 88/99 (88.9%) | 526/595 (88.4%) | 256 ms |
| gqreader MinimalConfig | 47/99 (47.5%) | 307/595 (51.6%) | 151 ms |
| Python QReader (original) | 96/99 (97.0%) | 581/595 (97.6%) | 159 ms |
| zbar standalone (no YOLO) | 86/99 (86.9%) | 241/595 (40.5%) | 45 ms |

## Optimized Results (2026-03-29, feat/onnx-optimization)

After ZBar decoder integration, CoreML EP, and generateMask 4.4× optimization:

| Configuration | Images pass% | QR codes % | Avg ms/img | Notes |
|--------------|-------------|-----------|-----------|-------|
| CPU + OpenCV (baseline) | 88/99 (88.9%) | 526/595 (88.4%) | 256 ms | |
| CPU + ZBar | 94/99 (94.9%) | 553/595 (92.9%) | 136 ms | +6pp accuracy, 1.9× faster |
| CoreML ALL + ZBar | 94/99 (94.9%) | 553/595 (92.9%) | **33 ms** | **4.8× faster than Python** |
| Python QReader | 96/99 (97.0%) | 581/595 (97.6%) | 159 ms | Target |

**Gap to Python QReader**: narrowed from 8.1pp → 2.1pp with ZBar decoder.
Remaining 2.1pp gap (QR2c, QR3c, QR3e, QR3k, QR3q): QR3e/k/q also fail Python;
QR2c and QR3c are the two images where the preprocessing pipeline diverges.

### Key Findings

**ZBar vs OpenCV decoder**: switching from OpenCV QRCodeDetector to libzbar (same library as
Python's pyzbar) recovers 6 images and 27 QR codes. One regression: QR2c passes with OpenCV
but fails with ZBar — a decoder-specific difference for that image's encoding.

**YOLO detection value proven**: zbar without YOLO detection scores 86.9% on images but only
40.5% on QR codes and completely fails QR6 (damaged finder patterns, 0/6 images). Both gqreader
and Python QReader pass all 6 QR6 images. The YOLO preprocessing step is essential for
robustness on real-world images.

**Preprocessing pipeline value**: removing scaling + perspective correction (MinimalConfig) saves
105 ms/image but drops accuracy by 41 pp (88.9% → 47.5%), confirming the preprocessing pipeline
is the core value-add of the library.

**Latency**: CoreML + ZBar combination achieves 33 ms/img, 4.8× faster than Python QReader's
159 ms/img (which uses Apple MPS via ultralytics + pyzbar). The speedup comes from two
orthogonal improvements: CoreML offloads ONNX inference to GPU/ANE (35% faster inference),
ZBar replaces OpenCV QRCodeDetector (3-5× faster per-crop decoding).

## Execution Provider Benchmarks (2026-03-29)

Measured on Apple M3 Pro, Dataset-QR-2020 (99 images), `feat/onnx-optimization` branch.
Run with: `task bench-providers` or `QREADER_PROVIDER=coreml task bench-env`.

| Provider | ms/image | Δ vs CPU | Notes |
|----------|----------|----------|-------|
| CPU (ORT_ENABLE_ALL) | 260 ms | baseline | Default, all platforms |
| CoreML CPU-only | 201 ms | −23% | CoreML graph compilation gain only |
| CoreML CPU+GPU | 169 ms | −35% | GPU offload on top |
| **CoreML ALL** | **170 ms** | **−35%** | Best on Apple Silicon |
| MinimalConfig CPU | 151 ms | −42% | No scaling/perspective (−41pp accuracy) |
| **Python QReader** | **159 ms** | target | ultralytics + Apple MPS |

**Conclusion**: CoreML ALL closes 35% of the latency gap (260→170 ms). The remaining 11 ms
gap vs Python lives in pre/postprocessing (numpy SIMD in Python vs Go scalar loops).
Accuracy is unchanged by provider selection.

### Provider Architecture (Cross-Platform)

`qrdet.ProviderConfig` abstracts all ONNX Runtime execution providers. The `ProviderFromEnv()`
function enables hardware-adaptive deployments via `QREADER_PROVIDER`:

| Platform | Recommended Provider | `QREADER_PROVIDER` value |
|----------|---------------------|--------------------------|
| macOS (Apple Silicon) | `CoreMLProviderConfig()` | `coreml` |
| Linux (NVIDIA GPU) | `CUDAProviderConfig()` | `cuda` |
| Linux (NVIDIA, max perf) | `TensorRTProviderConfig()` | `tensorrt` |
| Linux (Intel CPU/GPU) | `OpenVINOProviderConfig()` | `openvino` |
| Windows (any GPU) | `DirectMLProviderConfig()` | `directml` |
| All (portable fallback) | `CPUProviderConfig()` | `cpu` |

**CoreML is macOS-only** (Apple framework). On Linux/Windows, the equivalent is CUDA (NVIDIA)
or DirectML (Windows/AMD). ONNX Runtime falls back to CPU silently if an EP is unavailable.

### ORT Build Requirements per Provider

Not all ONNX Runtime distributions include all execution providers:

| Provider | Homebrew ORT 1.24.x | Official ORT binary | Notes |
|----------|--------------------|--------------------|-------|
| CPU | ✅ | ✅ | Always available |
| CoreML | ✅ (macOS only) | ✅ | CoreML.framework required |
| CUDA | ❌ | ✅ (separate pkg) | CUDA toolkit required |
| TensorRT | ❌ | ✅ (separate pkg) | TensorRT + CUDA required |
| DirectML | ❌ (Windows only) | ✅ | Windows only |
| OpenVINO | ❌ | ✅ (separate pkg) | Intel OpenVINO runtime required |

**Note**: `"CPUAndNE"` (Neural Engine only) is absent from ORT 1.24.x builds. Valid
CoreML `MLComputeUnits` values: `"ALL"`, `"CPUAndGPU"`, `"CPUOnly"` (case-sensitive).
