package qrdet

import (
	"fmt"
	"os"
	"runtime"

	ort "github.com/yalue/onnxruntime_go"
)

// session wraps an ONNX Runtime session for YOLOv8 segmentation inference.
// It manages tensor allocation and provides a clean interface for running inference.
type session struct {
	ortSession *ort.AdvancedSession

	// Input tensor: (1, 3, 640, 640) float32
	inputTensor *ort.Tensor[float32]

	// Output tensors
	// output0: (1, 116, 8400) - predictions (bbox + class + mask coefficients)
	// output1: (1, 32, 160, 160) - prototype masks
	output0 *ort.Tensor[float32]
	output1 *ort.Tensor[float32]

	// Shapes for validation
	inputShape   ort.Shape
	output0Shape ort.Shape
	output1Shape ort.Shape

	inputSize int
}

// newSession creates a new ONNX session from model data.
// The model data should be a valid ONNX model exported from YOLOv8-seg.
func newSession(modelData []byte, inputSize int) (*session, error) {
	// Initialize ONNX Runtime if not already done
	if err := initONNXRuntime(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
	}

	s := &session{
		inputSize: inputSize,
	}

	// Define expected shapes
	s.inputShape = ort.Shape{1, 3, int64(inputSize), int64(inputSize)}
	s.output0Shape = ort.Shape{1, modelOutputChannels, modelOutputPredictions}
	s.output1Shape = ort.Shape{1, modelMaskCoefficients, modelPrototypeSize, modelPrototypeSize}

	// Allocate input tensor
	inputData := make([]float32, 1*3*inputSize*inputSize)
	var err error
	s.inputTensor, err = ort.NewTensor(s.inputShape, inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	// Allocate output tensors
	output0Data := make([]float32, 1*modelOutputChannels*modelOutputPredictions)
	s.output0, err = ort.NewTensor(s.output0Shape, output0Data)
	if err != nil {
		s.inputTensor.Destroy()
		return nil, fmt.Errorf("failed to create output0 tensor: %w", err)
	}

	output1Data := make([]float32, 1*modelMaskCoefficients*modelPrototypeSize*modelPrototypeSize)
	s.output1, err = ort.NewTensor(s.output1Shape, output1Data)
	if err != nil {
		s.inputTensor.Destroy()
		s.output0.Destroy()
		return nil, fmt.Errorf("failed to create output1 tensor: %w", err)
	}

	// Create session options
	options, err := ort.NewSessionOptions()
	if err != nil {
		s.cleanup()
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer options.Destroy()

	// Set number of threads for CPU inference
	numThreads := runtime.NumCPU()
	if numThreads > 4 {
		numThreads = 4 // Cap at 4 for reasonable memory usage
	}
	if err := options.SetIntraOpNumThreads(numThreads); err != nil {
		s.cleanup()
		return nil, fmt.Errorf("failed to set intra-op threads: %w", err)
	}

	// Create the ONNX session
	inputs := []ort.ArbitraryTensor{s.inputTensor}
	outputs := []ort.ArbitraryTensor{s.output0, s.output1}

	s.ortSession, err = ort.NewAdvancedSessionWithONNXData(
		modelData,
		[]string{"images"},             // Input names (YOLOv8 default)
		[]string{"output0", "output1"}, // Output names (YOLOv8-seg default)
		inputs,
		outputs,
		options,
	)
	if err != nil {
		s.cleanup()
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return s, nil
}

// newSessionFromFile creates a new ONNX session from a model file.
func newSessionFromFile(modelPath string, inputSize int) (*session, error) {
	// Initialize ONNX Runtime if not already done
	if err := initONNXRuntime(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
	}

	s := &session{
		inputSize: inputSize,
	}

	// Define expected shapes
	s.inputShape = ort.Shape{1, 3, int64(inputSize), int64(inputSize)}
	s.output0Shape = ort.Shape{1, modelOutputChannels, modelOutputPredictions}
	s.output1Shape = ort.Shape{1, modelMaskCoefficients, modelPrototypeSize, modelPrototypeSize}

	// Allocate input tensor
	inputData := make([]float32, 1*3*inputSize*inputSize)
	var err error
	s.inputTensor, err = ort.NewTensor(s.inputShape, inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	// Allocate output tensors
	output0Data := make([]float32, 1*modelOutputChannels*modelOutputPredictions)
	s.output0, err = ort.NewTensor(s.output0Shape, output0Data)
	if err != nil {
		s.inputTensor.Destroy()
		return nil, fmt.Errorf("failed to create output0 tensor: %w", err)
	}

	output1Data := make([]float32, 1*modelMaskCoefficients*modelPrototypeSize*modelPrototypeSize)
	s.output1, err = ort.NewTensor(s.output1Shape, output1Data)
	if err != nil {
		s.inputTensor.Destroy()
		s.output0.Destroy()
		return nil, fmt.Errorf("failed to create output1 tensor: %w", err)
	}

	// Create session options
	options, err := ort.NewSessionOptions()
	if err != nil {
		s.cleanup()
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer options.Destroy()

	// Set number of threads for CPU inference
	numThreads := runtime.NumCPU()
	if numThreads > 4 {
		numThreads = 4
	}
	if err := options.SetIntraOpNumThreads(numThreads); err != nil {
		s.cleanup()
		return nil, fmt.Errorf("failed to set intra-op threads: %w", err)
	}

	// Create the ONNX session from file
	inputs := []ort.ArbitraryTensor{s.inputTensor}
	outputs := []ort.ArbitraryTensor{s.output0, s.output1}

	s.ortSession, err = ort.NewAdvancedSession(
		modelPath,
		[]string{"images"},
		[]string{"output0", "output1"},
		inputs,
		outputs,
		options,
	)
	if err != nil {
		s.cleanup()
		return nil, fmt.Errorf("failed to create ONNX session from %s: %w", modelPath, err)
	}

	return s, nil
}

// setInput copies preprocessed image data into the input tensor.
// The data must be in CHW format (channels, height, width) with shape (3, inputSize, inputSize).
// Values should be normalized to [0, 1] range.
func (s *session) setInput(data []float32) error {
	expectedLen := 3 * s.inputSize * s.inputSize
	if len(data) != expectedLen {
		return fmt.Errorf("input data has wrong length: got %d, expected %d", len(data), expectedLen)
	}

	inputSlice := s.inputTensor.GetData()
	copy(inputSlice, data)
	return nil
}

// run executes the model inference.
func (s *session) run() error {
	return s.ortSession.Run()
}

// getOutput0 returns the prediction output data.
// Shape: (1, 116, 8400) - flattened to []float32.
func (s *session) getOutput0() []float32 {
	return s.output0.GetData()
}

// getOutput1 returns the prototype mask output data.
// Shape: (1, 32, 160, 160) - flattened to []float32.
func (s *session) getOutput1() []float32 {
	return s.output1.GetData()
}

// close releases all resources associated with the session.
func (s *session) close() error {
	s.cleanup()
	return nil
}

// cleanup releases tensors and session resources.
func (s *session) cleanup() {
	if s.ortSession != nil {
		s.ortSession.Destroy()
		s.ortSession = nil
	}
	if s.inputTensor != nil {
		s.inputTensor.Destroy()
		s.inputTensor = nil
	}
	if s.output0 != nil {
		s.output0.Destroy()
		s.output0 = nil
	}
	if s.output1 != nil {
		s.output1.Destroy()
		s.output1 = nil
	}
}

// ONNX Runtime initialization state
var ortInitialized bool

// initONNXRuntime initializes the ONNX Runtime library.
// This is called automatically when creating a session.
func initONNXRuntime() error {
	if ortInitialized {
		return nil
	}

	// Set the shared library path based on the platform
	// On macOS, ONNX Runtime is typically installed via Homebrew
	if runtime.GOOS == "darwin" {
		// Try Homebrew paths (ARM and Intel)
		paths := []string{
			"/opt/homebrew/lib/libonnxruntime.dylib",   // ARM (Apple Silicon)
			"/usr/local/lib/libonnxruntime.dylib",      // Intel
			"/opt/homebrew/lib/libonnxruntime.1.dylib", // Versioned ARM
			"/usr/local/lib/libonnxruntime.1.dylib",    // Versioned Intel
		}
		for _, path := range paths {
			if _, err := os.Stat(path); err == nil {
				ort.SetSharedLibraryPath(path)
				break
			}
		}
	} else if runtime.GOOS == "linux" {
		// On Linux, try standard library paths
		paths := []string{
			"/usr/lib/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
			"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
			"libonnxruntime.so", // Try system path
		}
		for _, path := range paths {
			if _, err := os.Stat(path); err == nil {
				ort.SetSharedLibraryPath(path)
				break
			}
		}
	}
	// On Windows, the default "onnxruntime.dll" should work

	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("failed to initialize ONNX Runtime environment: %w", err)
	}

	ortInitialized = true
	return nil
}

// CleanupONNXRuntime releases ONNX Runtime resources.
// Call this once when your application exits.
func CleanupONNXRuntime() error {
	if !ortInitialized {
		return nil
	}

	if err := ort.DestroyEnvironment(); err != nil {
		return fmt.Errorf("failed to destroy ONNX Runtime environment: %w", err)
	}

	ortInitialized = false
	return nil
}
