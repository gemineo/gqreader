package imgprep

import (
	"fmt"
	"image"
	"iter"
	"math"

	"gocv.io/x/gocv"
)

// ============================================================================
// Elementary Operations (building blocks)
// ============================================================================

// Grayscale converts a color image to grayscale.
type Grayscale struct{}

func (g *Grayscale) Name() string { return "grayscale" }

func (g *Grayscale) Apply(src gocv.Mat) (gocv.Mat, error) {
	if src.Empty() {
		return gocv.Mat{}, fmt.Errorf("input image is empty")
	}

	dst := gocv.NewMat()

	if src.Channels() == 1 {
		src.CopyTo(&dst)
		return dst, nil
	}

	gocv.CvtColor(src, &dst, gocv.ColorBGRToGray)
	return dst, nil
}

// GaussianBlur applies Gaussian blur for noise reduction.
type GaussianBlur struct {
	// KernelSize is the blur kernel size. Must be positive and odd.
	// Zero or negative values default to 3; even values are rounded up to the next odd number.
	KernelSize int
	// SigmaX is the Gaussian standard deviation on the X axis. 0 means auto from KernelSize.
	SigmaX float64
}

func (g *GaussianBlur) Name() string {
	return fmt.Sprintf("gaussian_%d", g.kernelSize())
}

func (g *GaussianBlur) kernelSize() int {
	if g.KernelSize <= 0 {
		return 3
	}
	// Ensure odd
	if g.KernelSize%2 == 0 {
		return g.KernelSize + 1
	}
	return g.KernelSize
}

func (g *GaussianBlur) Apply(src gocv.Mat) (gocv.Mat, error) {
	if src.Empty() {
		return gocv.Mat{}, fmt.Errorf("input image is empty")
	}

	dst := gocv.NewMat()
	ksize := g.kernelSize()
	gocv.GaussianBlur(src, &dst, image.Point{X: ksize, Y: ksize}, g.SigmaX, 0, gocv.BorderDefault)
	return dst, nil
}

// MedianBlur applies median filter for salt-and-pepper noise removal.
// Particularly effective for fax documents.
type MedianBlur struct {
	// KernelSize is the median filter aperture. Must be positive and odd.
	// Zero or negative values default to 3; even values are rounded up to the next odd number.
	KernelSize int
}

func (m *MedianBlur) Name() string {
	return fmt.Sprintf("median_%d", m.kernelSize())
}

func (m *MedianBlur) kernelSize() int {
	if m.KernelSize <= 0 {
		return 3
	}
	// Ensure odd
	if m.KernelSize%2 == 0 {
		return m.KernelSize + 1
	}
	return m.KernelSize
}

func (m *MedianBlur) Apply(src gocv.Mat) (gocv.Mat, error) {
	if src.Empty() {
		return gocv.Mat{}, fmt.Errorf("input image is empty")
	}

	dst := gocv.NewMat()
	gocv.MedianBlur(src, &dst, m.kernelSize())
	return dst, nil
}

// FastNlMeansDenoising applies non-local means denoising.
// Excellent for scan noise but computationally expensive.
type FastNlMeansDenoising struct {
	// H is the filter strength. Higher values denoise more but may blur fine details.
	// Zero or negative values default to 10.
	H float32
}

func (f *FastNlMeansDenoising) Name() string {
	return fmt.Sprintf("nlmeans_%.0f", f.h())
}

func (f *FastNlMeansDenoising) h() float32 {
	if f.H <= 0 {
		return 10
	}
	return f.H
}

func (f *FastNlMeansDenoising) Apply(src gocv.Mat) (gocv.Mat, error) {
	if src.Empty() {
		return gocv.Mat{}, fmt.Errorf("input image is empty")
	}

	// Ensure grayscale for FastNlMeansDenoising
	gray := src
	needClose := false
	if src.Channels() > 1 {
		gray = gocv.NewMat()
		gocv.CvtColor(src, &gray, gocv.ColorBGRToGray)
		needClose = true
	}
	defer func() {
		if needClose {
			gray.Close()
		}
	}()

	dst := gocv.NewMat()
	gocv.FastNlMeansDenoisingWithParams(gray, &dst, f.h(), 7, 21)
	return dst, nil
}

// ============================================================================
// Contrast Enhancement
// ============================================================================

// CLAHE applies Contrast Limited Adaptive Histogram Equalization.
// The gold standard for document enhancement with uneven lighting.
type CLAHE struct {
	// ClipLimit controls contrast amplification. Increase toward 4.0 for low-contrast images.
	// Zero or negative values default to 2.0.
	ClipLimit float64
	// TileSize is the grid tile size for local histogram equalisation (NxN tiles).
	// Zero or negative values default to 8.
	TileSize int
}

func (c *CLAHE) Name() string {
	return fmt.Sprintf("clahe_%.1f", c.clipLimit())
}

func (c *CLAHE) clipLimit() float64 {
	if c.ClipLimit <= 0 {
		return 2.0
	}
	return c.ClipLimit
}

func (c *CLAHE) tileSize() int {
	if c.TileSize <= 0 {
		return 8
	}
	return c.TileSize
}

func (c *CLAHE) Apply(src gocv.Mat) (gocv.Mat, error) {
	if src.Empty() {
		return gocv.Mat{}, fmt.Errorf("input image is empty")
	}

	// Ensure grayscale
	gray := src
	needClose := false
	if src.Channels() > 1 {
		gray = gocv.NewMat()
		gocv.CvtColor(src, &gray, gocv.ColorBGRToGray)
		needClose = true
	}
	defer func() {
		if needClose {
			gray.Close()
		}
	}()

	dst := gocv.NewMat()
	tileSize := c.tileSize()
	clahe := gocv.NewCLAHEWithParams(c.clipLimit(), image.Point{X: tileSize, Y: tileSize})
	defer clahe.Close()

	clahe.Apply(gray, &dst)
	return dst, nil
}

// ============================================================================
// Binarization Strategies
// ============================================================================

// OtsuThreshold applies Otsu's automatic thresholding.
// Works best with uniform lighting; may fail on scans.
type OtsuThreshold struct{}

func (o *OtsuThreshold) Name() string { return "otsu" }

func (o *OtsuThreshold) Apply(src gocv.Mat) (gocv.Mat, error) {
	if src.Empty() {
		return gocv.Mat{}, fmt.Errorf("input image is empty")
	}

	// Ensure grayscale
	gray := src
	needClose := false
	if src.Channels() > 1 {
		gray = gocv.NewMat()
		gocv.CvtColor(src, &gray, gocv.ColorBGRToGray)
		needClose = true
	}
	defer func() {
		if needClose {
			gray.Close()
		}
	}()

	dst := gocv.NewMat()
	gocv.Threshold(gray, &dst, 0, 255, gocv.ThresholdBinary|gocv.ThresholdOtsu)
	return dst, nil
}

// AdaptiveThreshold applies local adaptive thresholding.
// Much better than Otsu for documents with uneven lighting.
type AdaptiveThreshold struct {
	// BlockSize is the neighbourhood size. Must be positive and odd.
	// Zero or negative values default to 11; even values are rounded up to the next odd number.
	BlockSize int
	// C is the constant subtracted from the weighted mean. Zero is treated as unset and
	// defaults to 2; use a small non-zero value if you need C near zero.
	C float64
	// Method is the local averaging method. Zero value defaults to AdaptiveThresholdGaussian.
	Method gocv.AdaptiveThresholdType
}

func (a *AdaptiveThreshold) Name() string {
	method := "mean"
	if a.Method == gocv.AdaptiveThresholdGaussian {
		method = "gaussian"
	}
	return fmt.Sprintf("adaptive_%s_%d_%.0f", method, a.blockSize(), a.c())
}

func (a *AdaptiveThreshold) blockSize() int {
	if a.BlockSize <= 0 {
		return 11
	}
	// Ensure odd
	if a.BlockSize%2 == 0 {
		return a.BlockSize + 1
	}
	return a.BlockSize
}

func (a *AdaptiveThreshold) c() float64 {
	if a.C == 0 {
		return 2
	}
	return a.C
}

func (a *AdaptiveThreshold) Apply(src gocv.Mat) (gocv.Mat, error) {
	if src.Empty() {
		return gocv.Mat{}, fmt.Errorf("input image is empty")
	}

	// Ensure grayscale
	gray := src
	needClose := false
	if src.Channels() > 1 {
		gray = gocv.NewMat()
		gocv.CvtColor(src, &gray, gocv.ColorBGRToGray)
		needClose = true
	}
	defer func() {
		if needClose {
			gray.Close()
		}
	}()

	method := a.Method
	if method == 0 {
		method = gocv.AdaptiveThresholdGaussian
	}

	dst := gocv.NewMat()
	gocv.AdaptiveThreshold(gray, &dst, 255, method, gocv.ThresholdBinary, a.blockSize(), float32(a.c()))
	return dst, nil
}

// ============================================================================
// Morphological Operations
// ============================================================================

// MorphOpen applies morphological opening (erosion then dilation).
// Removes small white noise spots.
type MorphOpen struct {
	// KernelSize is the structuring element size. Zero or negative values default to 3.
	KernelSize int
}

func (m *MorphOpen) Name() string {
	return fmt.Sprintf("morph_open_%d", m.kernelSize())
}

func (m *MorphOpen) kernelSize() int {
	if m.KernelSize <= 0 {
		return 3
	}
	return m.KernelSize
}

func (m *MorphOpen) Apply(src gocv.Mat) (gocv.Mat, error) {
	if src.Empty() {
		return gocv.Mat{}, fmt.Errorf("input image is empty")
	}

	kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Point{X: m.kernelSize(), Y: m.kernelSize()})
	defer kernel.Close()

	dst := gocv.NewMat()
	gocv.MorphologyEx(src, &dst, gocv.MorphOpen, kernel)
	return dst, nil
}

// MorphClose applies morphological closing (dilation then erosion).
// Fills small holes in black regions.
type MorphClose struct {
	// KernelSize is the structuring element size. Zero or negative values default to 3.
	KernelSize int
}

func (m *MorphClose) Name() string {
	return fmt.Sprintf("morph_close_%d", m.kernelSize())
}

func (m *MorphClose) kernelSize() int {
	if m.KernelSize <= 0 {
		return 3
	}
	return m.KernelSize
}

func (m *MorphClose) Apply(src gocv.Mat) (gocv.Mat, error) {
	if src.Empty() {
		return gocv.Mat{}, fmt.Errorf("input image is empty")
	}

	kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Point{X: m.kernelSize(), Y: m.kernelSize()})
	defer kernel.Close()

	dst := gocv.NewMat()
	gocv.MorphologyEx(src, &dst, gocv.MorphClose, kernel)
	return dst, nil
}

// ============================================================================
// Rotation Strategies
// ============================================================================

// RotationSearch generates rotated variants of the image.
// Useful for documents that may be slightly skewed.
type RotationSearch struct {
	// Angles lists the rotation angles to try, in degrees. Nil or empty defaults to [-5, -2, 2, 5].
	Angles []float64
}

func (r *RotationSearch) Name() string {
	return "rotation_search"
}

func (r *RotationSearch) angles() []float64 {
	if len(r.Angles) == 0 {
		return []float64{-5, -2, 2, 5}
	}
	return r.Angles
}

func (r *RotationSearch) Apply(src gocv.Mat) (gocv.Mat, error) {
	// Return first rotation for Strategy interface
	for _, mat := range r.Variants(src) {
		return mat, nil
	}
	// Fallback: return copy
	dst := gocv.NewMat()
	src.CopyTo(&dst)
	return dst, nil
}

func (r *RotationSearch) Variants(src gocv.Mat) iter.Seq2[string, gocv.Mat] {
	return func(yield func(string, gocv.Mat) bool) {
		for _, angle := range r.angles() {
			rotated := rotateImage(src, angle)
			name := fmt.Sprintf("rot_%.1f", angle)
			if !yield(name, rotated) {
				return
			}
		}
	}
}

// rotateImage rotates an image by the given angle (in degrees).
func rotateImage(src gocv.Mat, angleDeg float64) gocv.Mat {
	rows := src.Rows()
	cols := src.Cols()
	center := image.Point{X: cols / 2, Y: rows / 2}

	rotMat := gocv.GetRotationMatrix2D(center, angleDeg, 1.0)
	defer rotMat.Close()

	dst := gocv.NewMat()
	gocv.WarpAffine(src, &dst, rotMat, image.Point{X: cols, Y: rows})
	return dst
}

// ============================================================================
// Deskew Strategy (Hough-based)
// ============================================================================

// Deskew attempts to correct document rotation using Hough line detection.
type Deskew struct {
	// MaxAngle is the maximum skew correction angle in degrees. Zero or negative defaults to 15.
	MaxAngle float64
}

func (d *Deskew) Name() string { return "deskew" }

func (d *Deskew) maxAngle() float64 {
	if d.MaxAngle <= 0 {
		return 15
	}
	return d.MaxAngle
}

func (d *Deskew) Apply(src gocv.Mat) (gocv.Mat, error) {
	if src.Empty() {
		return gocv.Mat{}, fmt.Errorf("input image is empty")
	}

	// Convert to grayscale if needed
	gray := src
	needCloseGray := false
	if src.Channels() > 1 {
		gray = gocv.NewMat()
		gocv.CvtColor(src, &gray, gocv.ColorBGRToGray)
		needCloseGray = true
	}
	defer func() {
		if needCloseGray {
			gray.Close()
		}
	}()

	// Edge detection
	edges := gocv.NewMat()
	defer edges.Close()
	gocv.Canny(gray, &edges, 50, 150)

	// Hough line detection
	lines := gocv.NewMat()
	defer lines.Close()
	gocv.HoughLinesP(edges, &lines, 1, math.Pi/180, 100)

	if lines.Empty() || lines.Rows() == 0 {
		// No lines found, return copy
		dst := gocv.NewMat()
		src.CopyTo(&dst)
		return dst, nil
	}

	// Calculate median angle
	angles := make([]float64, 0, lines.Rows())
	for i := 0; i < lines.Rows(); i++ {
		x1 := float64(lines.GetVeciAt(i, 0)[0])
		y1 := float64(lines.GetVeciAt(i, 0)[1])
		x2 := float64(lines.GetVeciAt(i, 0)[2])
		y2 := float64(lines.GetVeciAt(i, 0)[3])

		angle := math.Atan2(y2-y1, x2-x1) * 180 / math.Pi

		// Normalize to [-45, 45] range (assume mostly horizontal/vertical lines)
		if angle > 45 {
			angle -= 90
		} else if angle < -45 {
			angle += 90
		}

		if math.Abs(angle) <= d.maxAngle() {
			angles = append(angles, angle)
		}
	}

	if len(angles) == 0 {
		dst := gocv.NewMat()
		src.CopyTo(&dst)
		return dst, nil
	}

	// Use median angle for robustness
	medianAngle := median(angles)

	// Rotate to correct skew
	return rotateImage(src, -medianAngle), nil
}

// median calculates the median of a slice.
func median(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	// Copy and sort
	sorted := make([]float64, len(values))
	copy(sorted, values)

	// Simple insertion sort (usually small slices)
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j-1] > sorted[j]; j-- {
			sorted[j-1], sorted[j] = sorted[j], sorted[j-1]
		}
	}

	mid := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[mid-1] + sorted[mid]) / 2
	}
	return sorted[mid]
}

// ============================================================================
// Channel Extraction
// ============================================================================

// Ensure RotationSearch implements VariantGenerator.
var _ VariantGenerator = (*RotationSearch)(nil)

// BlueChannel extracts only the blue channel.
// Sometimes better for printed documents with colored backgrounds.
type BlueChannel struct{}

func (b *BlueChannel) Name() string { return "blue_channel" }

func (b *BlueChannel) Apply(src gocv.Mat) (gocv.Mat, error) {
	if src.Empty() {
		return gocv.Mat{}, fmt.Errorf("input image is empty")
	}

	if src.Channels() == 1 {
		dst := gocv.NewMat()
		src.CopyTo(&dst)
		return dst, nil
	}

	channels := gocv.Split(src)
	defer func() {
		for i, ch := range channels {
			if i != 0 { // Keep blue channel
				ch.Close()
			}
		}
	}()

	// Blue is channel 0 in BGR
	dst := gocv.NewMat()
	channels[0].CopyTo(&dst)
	return dst, nil
}
