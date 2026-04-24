import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'kernels'))
import torch
import crom_cpu
print("Loaded torch and crom_cpu")

# Test 1
x = torch.randn(10, 128, dtype=torch.float32)
codebook = torch.randn(256, 8, dtype=torch.float32)
indices = torch.randint(0, 256, (10 * 16,), dtype=torch.int16)

print("Running gemv...")
try:
    y = crom_cpu.gemv(x, codebook, indices, 10, 128, 8)
    print("Done! y.shape =", y.shape)
except Exception as e:
    print("Error:", e)
