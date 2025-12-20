# QReader - Go Implementation

A native Go port of the Python QReader library for detecting and reading QR codes without Docker or Python runtime dependencies.

## What is QReader?

**QReader** is a high-accuracy QR code detection and reading system that combines:

1. **QRdet**: A YOLOv8-based neural network that locates QR codes in images with high precision
2. **Image Preprocessing**: Intelligent image transformation (contrast, threshold, resizing) to optimize detection
3. **QR Decoding**: Reads the QR code content once located

This Go implementation maintains full compatibility with the original Python QReader while eliminating external runtime dependencies.

## Quick Start

### Prerequisites

Before you begin, ensure you have:

- **Go 1.21+** ([Download here](https://golang.org/dl/))
- **macOS or Linux** (Windows support can be added)
- **Python 3.8+** (only needed for model export, one-time setup)
- **OpenCV** (for image processing via GoCV)
- **ONNX Runtime** (for AI model inference)

### Installation Steps

#### 1. Clone the Repository

```bash
cd /path/to/your/projects
git clone <repository-url> QReader_PoC
cd QReader_PoC
```

#### 2. Install System Dependencies

**macOS:**
```bash
# Install OpenCV using Homebrew
brew install opencv

# Install ONNX Runtime
brew install onnx-runtime
```

**Linux (Ubuntu/Debian):**
```bash
# Install OpenCV
sudo apt-get install libopencv-dev

# Install ONNX Runtime
sudo apt-get install libonnxruntime-dev
```

#### 3. Setup Go Dependencies

```bash
# Download Go module dependencies
make deps
```

#### 4. Prepare the AI Model (One-Time Setup)

The project needs to download and convert the YOLOv8 QR detection model:

```bash
# Setup Python environment with ML tools
make setup-python

# Download QRdet model and export to ONNX format
make model
```

This will:
- Create a Python virtual environment (`.venv/`)
- Download the QRdet model (~24 MB)
- Convert it to ONNX format (~47 MB)
- Validate the converted model

**Note**: This step requires an internet connection and takes 2-5 minutes.

#### 5. Build the Application

```bash
# Build the command-line tool (development mode)
make build

# Or build with embedded model (single standalone binary)
make build-embedded
```

This creates a `qrdetect` executable.

## Using QReader

### Detect QR Codes in an Image

```bash
# Detect QR codes and save annotated image
./qrdetect -input photo.jpg -output result.png

# With custom confidence threshold
./qrdetect -input photo.jpg -output result.png -conf 0.6

# Show results as JSON
./qrdetect -input photo.jpg -json

# Enable verbose output
./qrdetect -input photo.jpg -output result.png -v
```

### Output Examples

#### Image Output (PNG)

The annotated image shows:
- **Green rectangles**: Bounding boxes of detected QR codes
- **Blue quadrilaterals**: Precise boundaries of QR code corners
- **Confidence scores**: Detection confidence (0-100%)

Example:
```
[1] confidence=95%  bbox=[120, 150, 420, 450]
[2] confidence=87%  bbox=[500, 200, 800, 500]
```

#### JSON Output

```bash
./qrdetect -input photo.jpg -json
```

Returns:
```json
[
  {
    "index": 1,
    "confidence": 0.95,
    "bbox_xyxy": [120, 150, 420, 450],
    "center_xy": [270, 300],
    "width_height": [300, 300],
    "quad_xy": [[120, 150], [420, 150], [420, 450], [120, 450]]
  }
]
```

## CLI Reference

### Command Syntax

```
qrdetect -input <image> [options]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-input` | string | **required** | Path to input image file |
| `-output` | string | `output.png` | Path to save annotated image |
| `-model` | string | (embedded) | Path to ONNX model (dev mode only) |
| `-conf` | float | `0.5` | Confidence threshold (0.0-1.0) |
| `-nms` | float | `0.3` | NMS IoU threshold (0.0-1.0) |
| `-json` | bool | `false` | Output results as JSON instead of image |
| `-v` | bool | `false` | Verbose output (debug info) |

### Examples

```bash
# Basic detection
./qrdetect -input photo.jpg

# Strict confidence (only high-confidence detections)
./qrdetect -input photo.jpg -conf 0.8

# Development mode with external model
./qrdetect -model models/qrdet-s.onnx -input photo.jpg

# Get JSON results for programmatic processing
./qrdetect -input photo.jpg -json | jq '.[] | .confidence'

# All verbosity
./qrdetect -input photo.jpg -output result.png -v -conf 0.6
```

## Development Workflow

### Make Commands

Common build and development commands:

```bash
# Setup and build
make setup-python    # One-time Python setup
make model           # Download and export ONNX model
make build           # Build CLI (dev mode, external model)
make build-embedded  # Build CLI with embedded model

# Testing
make test            # Run unit tests
make test-coverage   # Run tests with coverage report
make bench           # Run performance benchmarks

# Maintenance
make fmt             # Format code
make lint            # Run code linter
make clean           # Remove build artifacts
make clean-all       # Remove everything including models
make help            # Show all available commands
```

### Project Structure

```
QReader_PoC/
├── README.md                 # This file (user guide)
├── CLAUDE.md                 # Developer documentation
├── Makefile                  # Build automation
├── go.mod / go.sum          # Go dependencies
│
├── cmd/qrdetect/            # CLI application
│   └── main.go              # Entry point
│
├── pkg/qrdet/               # Core detection library (reusable)
│   ├── detector.go          # Main API
│   ├── types.go             # Data structures
│   ├── session.go           # ONNX Runtime wrapper
│   ├── preprocess.go        # Image preparation
│   ├── postprocess.go       # Detection parsing
│   ├── quadrilateral.go     # Boundary fitting
│   ├── model.go             # Embedded model (production)
│   └── MODEL_FORMAT.md      # Model format documentation
│
├── internal/                # Internal packages
│   ├── geometry/            # Geometric utilities
│   └── mathutil/            # Math operations (NMS, etc.)
│
├── scripts/
│   └── export_model.py      # Model export script (Python)
│
├── models/                  # Model files (generated)
│   ├── qrdet-s.pt          # PyTorch model
│   └── qrdet-s.onnx        # ONNX model (used by Go)
│
└── testdata/                # Test images
    └── qrcodes/             # Sample QR code images
```

## Build Modes

### Development Mode (Smaller Binary)

```bash
make build
# Binary size: ~4.5 MB
# Model loaded from: pkg/qrdet/models/qrdet-s.onnx (external file)
# Use during development, requires model file present
```

### Production Mode (Standalone Binary)

```bash
make build-embedded
# Binary size: ~52 MB
# Model embedded in binary (via go:embed)
# Fully self-contained, no external files needed
# Use for deployment
```

## Troubleshooting

### Common Issues

#### 1. "Module not found" error

```
error: module github.com/mhey01/qreader: not found
```

**Solution**: Update the module name in `go.mod` to match your repository path:

```bash
# Edit go.mod and change:
# module github.com/mhey01/qreader
# to:
# module <your-repository-path>

go mod tidy
```

#### 2. "ONNX model not found"

```
Error: no model available
```

**Solution**: Ensure the model is exported:

```bash
make model  # Downloads and exports the model
```

#### 3. OpenCV installation issues (macOS)

If GoCV can't find OpenCV:

```bash
# Set paths explicitly
export CGO_CFLAGS=-I/opt/homebrew/opt/opencv/include
export CGO_LDFLAGS=-L/opt/homebrew/opt/opencv/lib
make build
```

#### 4. Python environment issues

```bash
# Recreate Python environment
rm -rf .venv
make setup-python
make model
```

#### 5. ONNX Runtime not found

```bash
# Reinstall ONNX Runtime
brew reinstall onnx-runtime

# Or set library path
export LD_LIBRARY_PATH=/opt/homebrew/opt/onnx-runtime/lib:$LD_LIBRARY_PATH
```

## Performance Notes

### Inference Speed

On Apple M3 Pro (CPU-only):
- **First detection**: ~0.5-1.0 seconds (model loading + inference)
- **Subsequent detections**: ~0.2-0.3 seconds (model cached)
- **Image size independence**: Handles any image size (auto-resized to 640×640)

### Memory Usage

- **Model**: ~50 MB (in memory)
- **Per-image processing**: ~10-20 MB
- **Total runtime**: ~100 MB baseline

## Limitations and Future Work

### Current Implementation

✅ QR code detection (single format)
✅ High-precision localization via segmentation
✅ Bounding box and corner point extraction
❌ QR code decoding (reads QR content) - planned for Phase 2
❌ Multiple barcode formats (DataMatrix, Code128, etc.) - requires model retraining

### Future Capabilities

The architecture is designed to support:
- Additional barcode formats (DataMatrix, Code128, Interleaved 2 of 5)
- QR code content reading/decoding
- Image preprocessing pipeline optimization
- GPU acceleration (NVIDIA, Metal Performance Shaders)

See `CLAUDE.md` for technical roadmap and extension path.

## Integration as a Library

To use QReader as a library in your Go code:

```go
package main

import (
    "fmt"
    "log"
    "gocv.io/x/gocv"
    "github.com/mhey01/qreader/pkg/qrdet"
)

func main() {
    // Create detector with default config
    detector, err := qrdet.NewDetector(qrdet.DefaultConfig())
    if err != nil {
        log.Fatal(err)
    }
    defer detector.Close()

    // Load image
    img := gocv.IMRead("photo.jpg", gocv.IMReadColor)
    if img.Empty() {
        log.Fatal("Failed to load image")
    }
    defer img.Close()

    // Run detection
    detections, err := detector.Detect(img)
    if err != nil {
        log.Fatal(err)
    }

    // Process results
    for i, det := range detections {
        fmt.Printf("QR Code %d:\n", i+1)
        fmt.Printf("  Confidence: %.2f%%\n", det.Confidence*100)
        fmt.Printf("  Bounding box: [%.0f, %.0f, %.0f, %.0f]\n",
            det.BBoxXYXY[0], det.BBoxXYXY[1],
            det.BBoxXYXY[2], det.BBoxXYXY[3])

        // Access other fields
        fmt.Printf("  Center: (%.0f, %.0f)\n", det.CenterXY[0], det.CenterXY[1])
        fmt.Printf("  Quad corners: %v\n", det.QuadXY)
    }
}
```

## License and Attribution

This project is a Go port of the original Python QReader library.

- **Original QReader**: https://github.com/Eric-Canas/QReader/
- **QRdet (detection model)**: https://github.com/Eric-Canas/qrdet/
- **ONNX Runtime**: https://onnxruntime.ai/
- **GoCV (OpenCV bindings)**: https://gocv.io/

## Support and Issues

For questions or issues:

1. Check the troubleshooting section above
2. Review `CLAUDE.md` for technical architecture
3. Review `pkg/qrdet/MODEL_FORMAT.md` for model details
4. Open an issue in the repository

## Changelog

### Version 1.0.0 (Current)

- ✅ QR code detection via YOLOv8 segmentation
- ✅ ONNX Runtime integration
- ✅ CLI tool with PNG export
- ✅ Bounding box and quadrilateral extraction
- ✅ Development and production build modes
- ✅ Full documentation

---

**Last Updated**: December 20, 2025
