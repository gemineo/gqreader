package qrdet

import (
	"fmt"
	"os"

	"gocv.io/x/gocv"
)

// ImageSource represents an input image for QR code detection.
// It provides an abstraction layer that hides CGO-specific types from the public API.
//
// ImageSource supports multiple input formats:
//   - File paths (JPEG, PNG, BMP, etc.)
//   - Encoded image bytes (JPEG, PNG, etc.)
//   - Raw pixel data (BGR format)
//
// Always call Close() when done with an ImageSource to release resources.
type ImageSource struct {
	mat gocv.Mat // internal: GoCV/OpenCV Mat (CGO type, not exposed)
}

// ImageFormat specifies the format of encoded image data.
type ImageFormat int

const (
	// ImageFormatAuto attempts to detect the format from file extension or magic bytes.
	ImageFormatAuto ImageFormat = iota
	// ImageFormatJPEG represents JPEG image format.
	ImageFormatJPEG
	// ImageFormatPNG represents PNG image format.
	ImageFormatPNG
	// ImageFormatBMP represents BMP image format.
	ImageFormatBMP
)

// NewImageFromMat creates an ImageSource from an existing gocv.Mat.
// This is useful when you already have a Mat from other GoCV operations.
//
// Note: The ImageSource will NOT take ownership of the Mat. The caller is
// responsible for closing the Mat when done. Do NOT call Close() on the
// ImageSource created from an external Mat.
//
// This function is primarily intended for internal use and advanced scenarios.
//
// Example:
//
//	mat := gocv.IMRead("photo.jpg", gocv.IMReadColor)
//	defer mat.Close()
//	img := qrdet.NewImageFromMat(mat)
//	detections, _ := detector.Detect(img)
func NewImageFromMat(mat gocv.Mat) *ImageSource {
	return &ImageSource{mat: mat}
}

// NewImageFromFile loads an image from a file path.
// The image format is automatically detected from the file.
//
// Returns an error if the file cannot be read or is not a valid image.
//
// Example:
//
//	img, err := qrdet.NewImageFromFile("photo.jpg")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer img.Close()
func NewImageFromFile(path string) (*ImageSource, error) {
	// Check if file exists
	if _, err := os.Stat(path); err != nil {
		return nil, fmt.Errorf("file not found: %s", path)
	}

	// Load image using OpenCV
	mat := gocv.IMRead(path, gocv.IMReadColor)
	if mat.Empty() {
		return nil, fmt.Errorf("failed to load image: %s (file may be corrupted or format unsupported)", path)
	}

	return &ImageSource{mat: mat}, nil
}

// NewImageFromBytes loads an image from encoded bytes (JPEG, PNG, etc.).
// The format parameter specifies the encoding format. Use ImageFormatAuto to auto-detect.
//
// Returns an error if the data cannot be decoded as a valid image.
//
// Example:
//
//	data, _ := os.ReadFile("photo.jpg")
//	img, err := qrdet.NewImageFromBytes(data, qrdet.ImageFormatAuto)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer img.Close()
func NewImageFromBytes(data []byte, format ImageFormat) (*ImageSource, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("image data is empty")
	}

	// Decode image using OpenCV
	// Note: IMDecode auto-detects format from magic bytes, so format parameter
	// is currently unused but kept for future extensibility
	mat, err := gocv.IMDecode(data, gocv.IMReadColor)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}

	if mat.Empty() {
		return nil, fmt.Errorf("decoded image is empty (invalid format or corrupted data)")
	}

	return &ImageSource{mat: mat}, nil
}

// NewImageFromRaw creates an image from raw pixel data in BGR format.
// The data must be in row-major order with 3 bytes per pixel (BGR channels).
//
// Parameters:
//   - data: Raw pixel data (length must equal width * height * 3)
//   - width: Image width in pixels
//   - height: Image height in pixels
//
// Returns an error if the data size doesn't match the dimensions.
//
// Example:
//
//	// Create a 640x480 blue image
//	width, height := 640, 480
//	data := make([]byte, width*height*3)
//	for i := 0; i < len(data); i += 3 {
//	    data[i] = 255 // B
//	    data[i+1] = 0 // G
//	    data[i+2] = 0 // R
//	}
//	img, err := qrdet.NewImageFromRaw(data, width, height)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer img.Close()
func NewImageFromRaw(data []byte, width, height int) (*ImageSource, error) {
	expectedSize := width * height * 3 // BGR format: 3 bytes per pixel
	if len(data) != expectedSize {
		return nil, fmt.Errorf("data size mismatch: got %d bytes, expected %d (width=%d, height=%d, 3 channels)",
			len(data), expectedSize, width, height)
	}

	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid dimensions: width=%d, height=%d", width, height)
	}

	// Create Mat from raw data
	mat, err := gocv.NewMatFromBytes(height, width, gocv.MatTypeCV8UC3, data)
	if err != nil {
		return nil, fmt.Errorf("failed to create image from raw data: %w", err)
	}

	if mat.Empty() {
		return nil, fmt.Errorf("created image is empty")
	}

	return &ImageSource{mat: mat}, nil
}

// Close releases resources associated with the ImageSource.
// Always call Close() when done with an ImageSource to avoid memory leaks.
//
// After calling Close(), the ImageSource should not be used anymore.
func (i *ImageSource) Close() error {
	if i == nil || i.mat.Empty() {
		return nil
	}
	return i.mat.Close()
}

// Width returns the image width in pixels.
func (i *ImageSource) Width() int {
	if i == nil || i.mat.Empty() {
		return 0
	}
	return i.mat.Cols()
}

// Height returns the image height in pixels.
func (i *ImageSource) Height() int {
	if i == nil || i.mat.Empty() {
		return 0
	}
	return i.mat.Rows()
}

// MatRef returns a reference to the underlying gocv.Mat.
// The returned Mat is NOT a copy — the caller must not close it independently,
// and must not use it after the ImageSource has been closed.
// Use this for advanced operations that require direct Mat access.
func (i *ImageSource) MatRef() gocv.Mat {
	return i.mat
}
