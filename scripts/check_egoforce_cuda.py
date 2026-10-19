"""Read-only CUDA smoke check for the isolated NVIDIA Jetson PyTorch runtime."""
import ctypes
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
library = root / "models/egoforce/runtime/libcusparse_lt-linux-aarch64-0.6.2.3-archive/lib/libcusparseLt.so.0"
handle = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; GPU smoke check failed")
value = torch.randn(512, 512, device="cuda", dtype=torch.float16)
result = value @ value.T
torch.cuda.synchronize()
if not torch.isfinite(result).all().item():
    raise RuntimeError("CUDA FP16 matrix output contains nonfinite values")
print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),
                  "capability":list(torch.cuda.get_device_capability()),"tensor_device":str(result.device),
                  "dtype":str(result.dtype),"fp16_matmul_verified":True},indent=2),flush=True)
