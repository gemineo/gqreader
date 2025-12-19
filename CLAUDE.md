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

## Critical Success Factors

1. **Exact parameter preservation**: Maintain same kernel sizes, thresholds, and scaling factors as Python version
2. **Image variant equivalence**: Generated variants must match Python output for regression prevention
3. **Model export validation**: Verify ONNX model produces same detections as PyTorch original
4. **Post-processing fidelity**: NMS and bbox handling must match QRdet behavior
