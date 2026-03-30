package imgprep

import (
	"fmt"
	"image"
	"math"

	"gocv.io/x/gocv"
)

// SauvolaThreshold implements Sauvola's adaptive binarization method.
// This is particularly effective for degraded historical documents and
// images with variable illumination.
//
// The threshold is calculated locally as:
//
//	T(x,y) = mean(x,y) * (1 + k * (stddev(x,y)/R - 1))
//
// where R is typically 128 for 8-bit images.
//
// Reference: Sauvola, J., & Pietikäinen, M. (2000). Adaptive document image binarization.
type SauvolaThreshold struct {
	// WindowSize is the local neighbourhood size. Must be positive and odd.
	// Zero or negative values default to 15; even values are rounded up to the next odd number.
	WindowSize int
	// K is the sensitivity parameter. Increase toward 0.5 for noisy images.
	// Zero or negative values default to 0.2.
	K float64
	// R is the dynamic range of the standard deviation. Zero or negative values default to 128.
	R float64
}

func (s *SauvolaThreshold) Name() string {
	return fmt.Sprintf("sauvola_%d_%.2f", s.windowSize(), s.k())
}

func (s *SauvolaThreshold) windowSize() int {
	if s.WindowSize <= 0 {
		return 15
	}
	if s.WindowSize%2 == 0 {
		return s.WindowSize + 1
	}
	return s.WindowSize
}

func (s *SauvolaThreshold) k() float64 {
	if s.K <= 0 {
		return 0.2
	}
	return s.K
}

func (s *SauvolaThreshold) r() float64 {
	if s.R <= 0 {
		return 128.0
	}
	return s.R
}

func (s *SauvolaThreshold) Apply(src gocv.Mat) (gocv.Mat, error) {
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

	// Convert to float32 for precise calculations
	floatImg := gocv.NewMat()
	defer floatImg.Close()
	gray.ConvertTo(&floatImg, gocv.MatTypeCV32F)

	windowSize := s.windowSize()
	kSize := image.Point{X: windowSize, Y: windowSize}

	// Calculate local mean using box filter
	mean := gocv.NewMat()
	defer mean.Close()
	gocv.BoxFilter(floatImg, &mean, int(gocv.MatTypeCV32F), kSize)

	// Calculate local mean of squares
	squared := gocv.NewMat()
	defer squared.Close()
	gocv.Multiply(floatImg, floatImg, &squared)

	meanSq := gocv.NewMat()
	defer meanSq.Close()
	gocv.BoxFilter(squared, &meanSq, int(gocv.MatTypeCV32F), kSize)

	// Calculate variance: E[X²] - E[X]²
	meanSquared := gocv.NewMat()
	defer meanSquared.Close()
	gocv.Multiply(mean, mean, &meanSquared)

	variance := gocv.NewMat()
	defer variance.Close()
	gocv.Subtract(meanSq, meanSquared, &variance)

	// Calculate standard deviation: sqrt(variance) = variance^0.5
	stddev := gocv.NewMat()
	defer stddev.Close()
	gocv.Pow(variance, 0.5, &stddev)

	// Calculate Sauvola threshold: T = mean * (1 + k * (stddev/R - 1))
	// Rearranged: T = mean + mean * k * (stddev/R - 1)
	//           = mean + k * mean * stddev/R - k * mean
	//           = mean * (1 - k) + k * mean * stddev / R

	k := s.k()
	R := s.r()

	threshold := gocv.NewMat()
	defer threshold.Close()

	// Create threshold matrix by iterating (OpenCV doesn't have element-wise ops we need)
	rows := floatImg.Rows()
	cols := floatImg.Cols()
	threshold = gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV32F)

	meanPtr, _ := mean.DataPtrFloat32()
	stddevPtr, _ := stddev.DataPtrFloat32()
	threshPtr, _ := threshold.DataPtrFloat32()

	for i := 0; i < rows*cols; i++ {
		m := float64(meanPtr[i])
		sd := float64(stddevPtr[i])
		t := m * (1.0 + k*(sd/R-1.0))
		threshPtr[i] = float32(t)
	}

	// Apply threshold: dst = (src > threshold) ? 255 : 0
	dst := gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV8U)
	floatPtr, _ := floatImg.DataPtrFloat32()
	dstPtr, _ := dst.DataPtrUint8()

	for i := 0; i < rows*cols; i++ {
		if floatPtr[i] > threshPtr[i] {
			dstPtr[i] = 255
		} else {
			dstPtr[i] = 0
		}
	}

	return dst, nil
}

// NiblackThreshold implements Niblack's adaptive binarization method.
// A precursor to Sauvola, simpler but can produce noisy results on clean backgrounds.
//
// The threshold is calculated locally as:
//
//	T(x,y) = mean(x,y) + k * stddev(x,y)
//
// where k is typically negative (-0.2) for document binarization.
type NiblackThreshold struct {
	// WindowSize is the local neighbourhood size. Must be positive and odd.
	// Zero or negative values default to 15; even values are rounded up to the next odd number.
	WindowSize int
	// K is the sensitivity parameter. Negative values are normal for document binarization.
	// Zero (the Go zero value) defaults to -0.2.
	K float64
}

func (n *NiblackThreshold) Name() string {
	return fmt.Sprintf("niblack_%d_%.2f", n.windowSize(), n.k())
}

func (n *NiblackThreshold) windowSize() int {
	if n.WindowSize <= 0 {
		return 15
	}
	if n.WindowSize%2 == 0 {
		return n.WindowSize + 1
	}
	return n.WindowSize
}

func (n *NiblackThreshold) k() float64 {
	if n.K == 0 {
		return -0.2
	}
	return n.K
}

func (n *NiblackThreshold) Apply(src gocv.Mat) (gocv.Mat, error) {
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

	// Convert to float32
	floatImg := gocv.NewMat()
	defer floatImg.Close()
	gray.ConvertTo(&floatImg, gocv.MatTypeCV32F)

	windowSize := n.windowSize()
	kSize := image.Point{X: windowSize, Y: windowSize}

	// Calculate local mean
	mean := gocv.NewMat()
	defer mean.Close()
	gocv.BoxFilter(floatImg, &mean, int(gocv.MatTypeCV32F), kSize)

	// Calculate local mean of squares
	squared := gocv.NewMat()
	defer squared.Close()
	gocv.Multiply(floatImg, floatImg, &squared)

	meanSq := gocv.NewMat()
	defer meanSq.Close()
	gocv.BoxFilter(squared, &meanSq, int(gocv.MatTypeCV32F), kSize)

	// Variance and stddev
	meanSquared := gocv.NewMat()
	defer meanSquared.Close()
	gocv.Multiply(mean, mean, &meanSquared)

	variance := gocv.NewMat()
	defer variance.Close()
	gocv.Subtract(meanSq, meanSquared, &variance)

	stddev := gocv.NewMat()
	defer stddev.Close()
	gocv.Pow(variance, 0.5, &stddev)

	// Niblack threshold: T = mean + k * stddev
	rows := floatImg.Rows()
	cols := floatImg.Cols()

	k := n.k()

	meanPtr, _ := mean.DataPtrFloat32()
	stddevPtr, _ := stddev.DataPtrFloat32()
	floatPtr, _ := floatImg.DataPtrFloat32()

	dst := gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV8U)
	dstPtr, _ := dst.DataPtrUint8()

	for i := 0; i < rows*cols; i++ {
		threshold := float64(meanPtr[i]) + k*float64(stddevPtr[i])
		if float64(floatPtr[i]) > threshold {
			dstPtr[i] = 255
		} else {
			dstPtr[i] = 0
		}
	}

	return dst, nil
}

// WolfThreshold implements Wolf's adaptive binarization method.
// An improvement over Sauvola for images with very light or very dark backgrounds.
//
// The threshold is calculated as:
//
//	T(x,y) = mean(x,y) + k * (mean(x,y) - min_gray) * (stddev(x,y) / max_stddev - 1)
type WolfThreshold struct {
	// WindowSize is the local neighbourhood size. Must be positive and odd.
	// Zero or negative values default to 15; even values are rounded up to the next odd number.
	WindowSize int
	// K is the sensitivity parameter. Zero or negative values default to 0.5.
	K float64
}

func (w *WolfThreshold) Name() string {
	return fmt.Sprintf("wolf_%d_%.2f", w.windowSize(), w.k())
}

func (w *WolfThreshold) windowSize() int {
	if w.WindowSize <= 0 {
		return 15
	}
	if w.WindowSize%2 == 0 {
		return w.WindowSize + 1
	}
	return w.WindowSize
}

func (w *WolfThreshold) k() float64 {
	if w.K <= 0 {
		return 0.5
	}
	return w.K
}

func (w *WolfThreshold) Apply(src gocv.Mat) (gocv.Mat, error) {
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

	// Convert to float32
	floatImg := gocv.NewMat()
	defer floatImg.Close()
	gray.ConvertTo(&floatImg, gocv.MatTypeCV32F)

	windowSize := w.windowSize()
	kSize := image.Point{X: windowSize, Y: windowSize}

	// Calculate local mean
	mean := gocv.NewMat()
	defer mean.Close()
	gocv.BoxFilter(floatImg, &mean, int(gocv.MatTypeCV32F), kSize)

	// Calculate local mean of squares
	squared := gocv.NewMat()
	defer squared.Close()
	gocv.Multiply(floatImg, floatImg, &squared)

	meanSq := gocv.NewMat()
	defer meanSq.Close()
	gocv.BoxFilter(squared, &meanSq, int(gocv.MatTypeCV32F), kSize)

	// Variance and stddev
	meanSquared := gocv.NewMat()
	defer meanSquared.Close()
	gocv.Multiply(mean, mean, &meanSquared)

	variance := gocv.NewMat()
	defer variance.Close()
	gocv.Subtract(meanSq, meanSquared, &variance)

	stddev := gocv.NewMat()
	defer stddev.Close()
	gocv.Pow(variance, 0.5, &stddev)

	// Find global min gray and max stddev
	rows := floatImg.Rows()
	cols := floatImg.Cols()

	floatPtr, _ := floatImg.DataPtrFloat32()
	stddevPtr, _ := stddev.DataPtrFloat32()
	meanPtr, _ := mean.DataPtrFloat32()

	minGray := float64(math.MaxFloat32)
	maxStddev := float64(0)

	for i := 0; i < rows*cols; i++ {
		if float64(floatPtr[i]) < minGray {
			minGray = float64(floatPtr[i])
		}
		if float64(stddevPtr[i]) > maxStddev {
			maxStddev = float64(stddevPtr[i])
		}
	}

	if maxStddev < 1e-6 {
		maxStddev = 1.0 // Avoid division by zero
	}

	// Wolf threshold
	k := w.k()
	dst := gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV8U)
	dstPtr, _ := dst.DataPtrUint8()

	for i := 0; i < rows*cols; i++ {
		m := float64(meanPtr[i])
		sd := float64(stddevPtr[i])
		threshold := m + k*(m-minGray)*(sd/maxStddev-1.0)

		if float64(floatPtr[i]) > threshold {
			dstPtr[i] = 255
		} else {
			dstPtr[i] = 0
		}
	}

	return dst, nil
}
