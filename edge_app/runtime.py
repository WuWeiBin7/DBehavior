from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class OnnxRuntimeSession:
    def __init__(self, model_path: str, providers: Sequence[str] | None = None) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for .onnx inference") from exc
        available = list(ort.get_available_providers())
        preferred = list(providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
        providers = [name for name in preferred if name in available]
        if not providers:
            providers = available
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input = self.session.get_inputs()[0]
        self.input_name = self.input.name
        self.input_shape = list(self.input.shape)
        self.output_names = [o.name for o in self.session.get_outputs()]

    def infer(self, inputs: np.ndarray) -> list[np.ndarray]:
        outputs = self.session.run(self.output_names, {self.input_name: inputs})
        return [np.asarray(o) for o in outputs]


class TensorRTRuntimeSession:
    def __init__(self, engine_path: str) -> None:
        if not Path(engine_path).exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError("tensorrt and pycuda are required for .engine inference") from exc

        self.cuda = cuda
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings = []
        self.host_inputs = []
        self.host_outputs = []
        self.device_inputs = []
        self.device_outputs = []

        for idx in range(self.engine.num_bindings):
            shape = tuple(self.context.get_binding_shape(idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(idx):
                self.host_inputs.append((idx, host_mem, shape))
                self.device_inputs.append(device_mem)
            else:
                self.host_outputs.append((idx, host_mem, shape))
                self.device_outputs.append(device_mem)

    def infer(self, inputs: np.ndarray) -> list[np.ndarray]:
        if len(self.host_inputs) != 1:
            raise RuntimeError("Only single-input TensorRT engines are supported in this wrapper")
        idx, host_input, expected_shape = self.host_inputs[0]
        if int(np.prod(inputs.shape)) != host_input.size:
            raise ValueError(f"Input shape {inputs.shape} does not match TensorRT binding {expected_shape}")
        np.copyto(host_input, inputs.ravel())
        self.cuda.memcpy_htod_async(self.device_inputs[0], host_input, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        outputs = []
        for out_i, (_, host_output, shape) in enumerate(self.host_outputs):
            self.cuda.memcpy_dtoh_async(host_output, self.device_outputs[out_i], self.stream)
            outputs.append(host_output.reshape(shape).copy())
        self.stream.synchronize()
        return outputs


def create_runtime_session(model_path: str):
    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx":
        return OnnxRuntimeSession(model_path)
    if suffix == ".engine":
        return TensorRTRuntimeSession(model_path)
    raise ValueError(f"Unsupported model format {suffix!r}; expected .onnx or .engine")
