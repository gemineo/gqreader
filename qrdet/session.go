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
	// output0: (1, 37, 8400) - predictions (bbox + confidence + mask coefficients)
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
func newSession(modelData []byte, inputSize int, pc ProviderConfig) (*session, error) {
	if err := initONNXRuntime(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
	}

	s := &session{inputSize: inputSize}

	s.inputShape = ort.Shape{1, 3, int64(inputSize), int64(inputSize)}
	s.output0Shape = ort.Shape{1, modelOutputChannels, modelOutputPredictions}
	s.output1Shape = ort.Shape{1, modelMaskCoefficients, modelPrototypeSize, modelPrototypeSize}

	var err error
	s.inputTensor, err = ort.NewTensor(s.inputShape, make([]float32, 1*3*inputSize*inputSize))
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	s.output0, err = ort.NewTensor(s.output0Shape, make([]float32, 1*modelOutputChannels*modelOutputPredictions))
	if err != nil {
		s.inputTensor.Destroy()
		return nil, fmt.Errorf("failed to create output0 tensor: %w", err)
	}

	s.output1, err = ort.NewTensor(s.output1Shape, make([]float32, 1*modelMaskCoefficients*modelPrototypeSize*modelPrototypeSize))
	if err != nil {
		s.inputTensor.Destroy()
		s.output0.Destroy()
		return nil, fmt.Errorf("failed to create output1 tensor: %w", err)
	}

	options, err := buildSessionOptions(pc)
	if err != nil {
		s.cleanup()
		return nil, err
	}
	defer options.Destroy()

	s.ortSession, err = ort.NewAdvancedSessionWithONNXData(
		modelData,
		[]string{"images"},
		[]string{"output0", "output1"},
		[]ort.ArbitraryTensor{s.inputTensor},
		[]ort.ArbitraryTensor{s.output0, s.output1},
		options,
	)
	if err != nil {
		s.cleanup()
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return s, nil
}

// newSessionFromFile creates a new ONNX session from a model file.
func newSessionFromFile(modelPath string, inputSize int, pc ProviderConfig) (*session, error) {
	if err := initONNXRuntime(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
	}

	s := &session{inputSize: inputSize}

	s.inputShape = ort.Shape{1, 3, int64(inputSize), int64(inputSize)}
	s.output0Shape = ort.Shape{1, modelOutputChannels, modelOutputPredictions}
	s.output1Shape = ort.Shape{1, modelMaskCoefficients, modelPrototypeSize, modelPrototypeSize}

	var err error
	s.inputTensor, err = ort.NewTensor(s.inputShape, make([]float32, 1*3*inputSize*inputSize))
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	s.output0, err = ort.NewTensor(s.output0Shape, make([]float32, 1*modelOutputChannels*modelOutputPredictions))
	if err != nil {
		s.inputTensor.Destroy()
		return nil, fmt.Errorf("failed to create output0 tensor: %w", err)
	}

	s.output1, err = ort.NewTensor(s.output1Shape, make([]float32, 1*modelMaskCoefficients*modelPrototypeSize*modelPrototypeSize))
	if err != nil {
		s.inputTensor.Destroy()
		s.output0.Destroy()
		return nil, fmt.Errorf("failed to create output1 tensor: %w", err)
	}

	options, err := buildSessionOptions(pc)
	if err != nil {
		s.cleanup()
		return nil, err
	}
	defer options.Destroy()

	s.ortSession, err = ort.NewAdvancedSession(
		modelPath,
		[]string{"images"},
		[]string{"output0", "output1"},
		[]ort.ArbitraryTensor{s.inputTensor},
		[]ort.ArbitraryTensor{s.output0, s.output1},
		options,
	)
	if err != nil {
		s.cleanup()
		return nil, fmt.Errorf("failed to create ONNX session from %s: %w", modelPath, err)
	}

	return s, nil
}

// buildSessionOptions constructs ORT SessionOptions from a ProviderConfig.
func buildSessionOptions(pc ProviderConfig) (*ort.SessionOptions, error) {
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}

	// Graph optimization: zero value (ORT_DISABLE_ALL) is treated as EnableAll
	// so that the zero-value ProviderConfig is safe to use without explicit setup.
	level := pc.GraphOptimization
	if level == ort.GraphOptimizationLevelDisableAll {
		level = ort.GraphOptimizationLevelEnableAll
	}
	if err := options.SetGraphOptimizationLevel(level); err != nil {
		options.Destroy()
		return nil, fmt.Errorf("failed to set graph optimization level: %w", err)
	}

	// Intra-op threads: 0 means auto-select.
	numThreads := pc.IntraOpThreads
	if numThreads <= 0 {
		numThreads = runtime.NumCPU()
		if numThreads > 4 {
			numThreads = 4
		}
	}
	if err := options.SetIntraOpNumThreads(numThreads); err != nil {
		options.Destroy()
		return nil, fmt.Errorf("failed to set intra-op threads: %w", err)
	}

	// Inter-op threads: only configure when explicitly requested.
	if pc.InterOpThreads > 0 {
		if err := options.SetInterOpNumThreads(pc.InterOpThreads); err != nil {
			options.Destroy()
			return nil, fmt.Errorf("failed to set inter-op threads: %w", err)
		}
	}

	// Memory pattern optimization.
	if err := options.SetMemPattern(pc.EnableMemPattern); err != nil {
		options.Destroy()
		return nil, fmt.Errorf("failed to set memory pattern: %w", err)
	}

	// Execution provider: the selected EP is appended first; ORT falls back to CPU
	// automatically when the EP is unavailable at runtime.
	switch pc.Provider {
	case ProviderCoreML:
		if err := options.AppendExecutionProviderCoreMLV2(pc.CoreMLOptions); err != nil {
			options.Destroy()
			return nil, fmt.Errorf("failed to append CoreML provider: %w", err)
		}

	case ProviderCUDA:
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err != nil {
			options.Destroy()
			return nil, fmt.Errorf("failed to create CUDA provider options: %w", err)
		}
		if len(pc.CUDAOptions) > 0 {
			if err := cudaOpts.Update(pc.CUDAOptions); err != nil {
				cudaOpts.Destroy()
				options.Destroy()
				return nil, fmt.Errorf("failed to update CUDA provider options: %w", err)
			}
		}
		appendErr := options.AppendExecutionProviderCUDA(cudaOpts)
		cudaOpts.Destroy()
		if appendErr != nil {
			options.Destroy()
			return nil, fmt.Errorf("failed to append CUDA provider: %w", appendErr)
		}

	case ProviderTensorRT:
		trtOpts, err := ort.NewTensorRTProviderOptions()
		if err != nil {
			options.Destroy()
			return nil, fmt.Errorf("failed to create TensorRT provider options: %w", err)
		}
		if len(pc.TensorRTOptions) > 0 {
			if err := trtOpts.Update(pc.TensorRTOptions); err != nil {
				trtOpts.Destroy()
				options.Destroy()
				return nil, fmt.Errorf("failed to update TensorRT provider options: %w", err)
			}
		}
		appendErr := options.AppendExecutionProviderTensorRT(trtOpts)
		trtOpts.Destroy()
		if appendErr != nil {
			options.Destroy()
			return nil, fmt.Errorf("failed to append TensorRT provider: %w", appendErr)
		}

	case ProviderDirectML:
		if err := options.AppendExecutionProviderDirectML(pc.DirectMLDeviceID); err != nil {
			options.Destroy()
			return nil, fmt.Errorf("failed to append DirectML provider: %w", err)
		}

	case ProviderOpenVINO:
		if err := options.AppendExecutionProviderOpenVINO(pc.OpenVINOOptions); err != nil {
			options.Destroy()
			return nil, fmt.Errorf("failed to append OpenVINO provider: %w", err)
		}

	case ProviderCPU:
		// CPU is always the ORT fallback; nothing to configure explicitly.

	default:
		options.Destroy()
		return nil, fmt.Errorf("unsupported execution provider: %d", pc.Provider)
	}

	return options, nil
}

// setInput copies preprocessed image data into the input tensor.
// The data must be in CHW format with shape (3, inputSize, inputSize),
// values normalized to [0, 1].
func (s *session) setInput(data []float32) error {
	expectedLen := 3 * s.inputSize * s.inputSize
	if len(data) != expectedLen {
		return fmt.Errorf("input data has wrong length: got %d, expected %d", len(data), expectedLen)
	}
	copy(s.inputTensor.GetData(), data)
	return nil
}

// run executes the model inference.
func (s *session) run() error {
	return s.ortSession.Run()
}

// getOutput0 returns the prediction output data.
// Shape: (1, 37, 8400) - flattened to []float32.
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

	if runtime.GOOS == "darwin" {
		paths := []string{
			"/opt/homebrew/lib/libonnxruntime.dylib",
			"/usr/local/lib/libonnxruntime.dylib",
			"/opt/homebrew/lib/libonnxruntime.1.dylib",
			"/usr/local/lib/libonnxruntime.1.dylib",
		}
		for _, path := range paths {
			if _, err := os.Stat(path); err == nil {
				ort.SetSharedLibraryPath(path)
				break
			}
		}
	} else if runtime.GOOS == "linux" {
		paths := []string{
			"/usr/lib/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
			"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
			"libonnxruntime.so",
		}
		for _, path := range paths {
			if _, err := os.Stat(path); err == nil {
				ort.SetSharedLibraryPath(path)
				break
			}
		}
	}

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
