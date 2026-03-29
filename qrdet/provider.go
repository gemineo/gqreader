package qrdet

import (
	"os"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

// ExecutionProvider selects the hardware backend for ONNX Runtime inference.
type ExecutionProvider int

const (
	// ProviderCPU uses the default multi-threaded CPU execution provider.
	// Portable: works on all platforms. Use as the safe fallback.
	ProviderCPU ExecutionProvider = iota

	// ProviderCoreML uses Apple's CoreML framework (macOS only).
	// Offloads computation to GPU and/or Neural Engine on Apple Silicon.
	// Falls back to CPU when CoreML is unavailable.
	ProviderCoreML

	// ProviderCUDA uses NVIDIA's CUDA execution provider (Linux/Windows).
	// Requires a CUDA-capable GPU and the CUDA-enabled ONNX Runtime build.
	ProviderCUDA

	// ProviderTensorRT uses NVIDIA TensorRT (Linux/Windows).
	// Highest throughput on NVIDIA hardware; compiles the model on first run.
	// Requires TensorRT and the TensorRT-enabled ONNX Runtime build.
	ProviderTensorRT

	// ProviderDirectML uses Microsoft DirectML (Windows).
	// Portable GPU acceleration on Windows via Direct3D 12.
	// Works on NVIDIA, AMD, and Intel GPUs.
	ProviderDirectML

	// ProviderOpenVINO uses Intel's OpenVINO toolkit (Linux/Windows/macOS).
	// Best on Intel CPUs and integrated graphics; also supports discrete Intel GPUs.
	ProviderOpenVINO
)

// ProviderConfig configures the ONNX Runtime session options.
//
// The zero value is valid: selects the CPU execution provider with
// full graph optimization and automatic thread count.
type ProviderConfig struct {
	// Provider selects the hardware execution provider. Default: ProviderCPU.
	Provider ExecutionProvider

	// CoreMLOptions configures the CoreML EP (macOS only).
	// Common keys:
	//   "MLComputeUnits" – "ALL" | "CPUAndGPU" | "CPUOnly" (default: "ALL")
	//   "ModelFormat"    – "MLProgram" | "NeuralNetwork" (default: "MLProgram")
	// Note: "CPUAndNE" (Neural Engine only) requires ORT 1.25+ with ANE support.
	// See: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
	CoreMLOptions map[string]string

	// CUDAOptions configures the CUDA EP (Linux/Windows, NVIDIA).
	// Common keys: "device_id", "gpu_mem_limit", "arena_extend_strategy".
	// See: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
	CUDAOptions map[string]string

	// TensorRTOptions configures the TensorRT EP (Linux/Windows, NVIDIA).
	// Common keys: "trt_max_workspace_size", "trt_fp16_enable", "trt_engine_cache_enable".
	// See: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
	TensorRTOptions map[string]string

	// OpenVINOOptions configures the OpenVINO EP (Linux/Windows/macOS, Intel).
	// Common keys: "device_type" ("CPU_FP32", "GPU_FP32", "GPU_FP16").
	// See: https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html
	OpenVINOOptions map[string]string

	// DirectMLDeviceID selects the DirectML device (Windows). Default 0.
	DirectMLDeviceID int

	// IntraOpThreads sets the number of threads for intra-operator parallelism.
	// 0 auto-selects: min(runtime.NumCPU(), 4).
	IntraOpThreads int

	// InterOpThreads sets the number of threads for inter-operator parallelism.
	// Only relevant in ORT parallel execution mode. 0 uses the ORT default.
	InterOpThreads int

	// GraphOptimization controls the ONNX graph optimization level.
	// The zero value maps to GraphOptimizationLevelEnableAll (best performance).
	GraphOptimization ort.GraphOptimizationLevel

	// EnableMemPattern enables memory pattern optimization.
	// When true ORT pre-allocates memory based on observed usage patterns.
	EnableMemPattern bool
}

// CPUProviderConfig returns a ProviderConfig optimized for CPU inference.
// Enables all graph optimizations and memory pattern for best throughput.
// Portable: works on all platforms.
func CPUProviderConfig() ProviderConfig {
	return ProviderConfig{
		Provider:          ProviderCPU,
		GraphOptimization: ort.GraphOptimizationLevelEnableAll,
		EnableMemPattern:  true,
	}
}

// CoreMLProviderConfig returns a ProviderConfig using Apple's CoreML framework
// with all compute units (CPU + GPU + Neural Engine on Apple Silicon).
// Falls back to CPU automatically when CoreML is not available.
//
// Benchmark on Apple M3 Pro (Dataset-QR-2020, 99 images):
//
//	CPU:    260 ms/image
//	CoreML: 170 ms/image  (-35%, near Python/MPS parity at 159 ms)
func CoreMLProviderConfig() ProviderConfig {
	return ProviderConfig{
		Provider: ProviderCoreML,
		CoreMLOptions: map[string]string{
			"MLComputeUnits": "ALL",
			"ModelFormat":    "MLProgram",
		},
		GraphOptimization: ort.GraphOptimizationLevelEnableAll,
		EnableMemPattern:  true,
	}
}

// CoreMLGPUConfig returns a ProviderConfig using CoreML with CPU and GPU
// compute units. Useful for comparing GPU vs full ALL dispatch on Apple Silicon.
func CoreMLGPUConfig() ProviderConfig {
	return ProviderConfig{
		Provider: ProviderCoreML,
		CoreMLOptions: map[string]string{
			"MLComputeUnits": "CPUAndGPU",
			"ModelFormat":    "MLProgram",
		},
		GraphOptimization: ort.GraphOptimizationLevelEnableAll,
		EnableMemPattern:  true,
	}
}

// CoreMLCPUOnlyConfig returns a ProviderConfig using CoreML with CPU-only compute units.
// Isolation baseline: same CoreML graph compilation overhead as ALL/GPU, but no
// hardware offload. Reveals the split between CoreML graph opt and accelerator gains.
//
// Note: "CPUAndNE" (Neural Engine only) is absent from ONNX Runtime 1.24.x.
func CoreMLCPUOnlyConfig() ProviderConfig {
	return ProviderConfig{
		Provider: ProviderCoreML,
		CoreMLOptions: map[string]string{
			"MLComputeUnits": "CPUOnly",
			"ModelFormat":    "MLProgram",
		},
		GraphOptimization: ort.GraphOptimizationLevelEnableAll,
		EnableMemPattern:  true,
	}
}

// CoreMLNeuralEngineConfig is an alias kept for forward compatibility.
// Falls back to CoreMLGPUConfig since "CPUAndNE" requires ORT 1.25+ with ANE support.
//
// Deprecated: Use CoreMLGPUConfig or CoreMLProviderConfig instead.
func CoreMLNeuralEngineConfig() ProviderConfig {
	return CoreMLGPUConfig()
}

// CUDAProviderConfig returns a ProviderConfig using NVIDIA's CUDA EP.
// Requires a CUDA-capable GPU and a CUDA-enabled ONNX Runtime build.
// Typical speedup vs CPU: 3-10× depending on GPU and model.
func CUDAProviderConfig() ProviderConfig {
	return ProviderConfig{
		Provider:          ProviderCUDA,
		GraphOptimization: ort.GraphOptimizationLevelEnableAll,
		EnableMemPattern:  true,
	}
}

// TensorRTProviderConfig returns a ProviderConfig using NVIDIA TensorRT.
// Requires TensorRT and a TensorRT-enabled ONNX Runtime build.
// Best throughput on NVIDIA hardware; first inference is slow (engine compile).
func TensorRTProviderConfig() ProviderConfig {
	return ProviderConfig{
		Provider: ProviderTensorRT,
		TensorRTOptions: map[string]string{
			"trt_fp16_enable":       "1",
			"trt_engine_cache_enable": "1",
		},
		GraphOptimization: ort.GraphOptimizationLevelEnableAll,
		EnableMemPattern:  true,
	}
}

// DirectMLProviderConfig returns a ProviderConfig using Microsoft DirectML.
// Portable GPU acceleration on Windows via Direct3D 12.
// Works on NVIDIA, AMD, and Intel GPUs.
func DirectMLProviderConfig() ProviderConfig {
	return ProviderConfig{
		Provider:          ProviderDirectML,
		GraphOptimization: ort.GraphOptimizationLevelEnableAll,
		EnableMemPattern:  true,
	}
}

// OpenVINOProviderConfig returns a ProviderConfig using Intel's OpenVINO toolkit.
// Best on Intel CPUs and integrated/discrete Intel graphics.
// The "CPU_FP32" device type works on any Intel CPU without drivers.
func OpenVINOProviderConfig() ProviderConfig {
	return ProviderConfig{
		Provider: ProviderOpenVINO,
		OpenVINOOptions: map[string]string{
			"device_type": "CPU_FP32",
		},
		GraphOptimization: ort.GraphOptimizationLevelEnableAll,
		EnableMemPattern:  true,
	}
}

// ProviderFromEnv returns a ProviderConfig driven by the QREADER_PROVIDER
// environment variable. This enables hardware-adaptive deployments without
// recompilation.
//
// Valid values for QREADER_PROVIDER:
//
//	"cpu"       – CPU (default, all platforms)
//	"coreml"    – Apple CoreML (macOS)
//	"cuda"      – NVIDIA CUDA (Linux/Windows)
//	"tensorrt"  – NVIDIA TensorRT (Linux/Windows)
//	"directml"  – DirectML (Windows)
//	"openvino"  – Intel OpenVINO (Linux/Windows/macOS)
//
// Example:
//
//	QREADER_PROVIDER=coreml ./myapp
//	QREADER_PROVIDER=cuda   ./myapp
func ProviderFromEnv() ProviderConfig {
	switch strings.ToLower(os.Getenv("QREADER_PROVIDER")) {
	case "coreml":
		return CoreMLProviderConfig()
	case "cuda":
		return CUDAProviderConfig()
	case "tensorrt":
		return TensorRTProviderConfig()
	case "directml":
		return DirectMLProviderConfig()
	case "openvino":
		return OpenVINOProviderConfig()
	default:
		return CPUProviderConfig()
	}
}
