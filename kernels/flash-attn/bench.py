import os
import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

os.environ['TORCH_CUDA_ARCH_LIST'] = 'Volta'

# Load the CUDA kernel as a python module
flash_attn = load(name="flash_attn", sources=["main.cpp", "flash_attn.cu"], extra_cuda_cflags=['-O2'])

# Use small model parameters
batch_size = 16
n_head = 12
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

print('=== profiling manual attention ===')

# flash attention aims to faster
def manual_attn(q, k, v):
    attn = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
    attn = F.softmax(attn, dim=-1)
    y = attn @ v
    return y

with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
    manual_result = manual_attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling cuda flash attention === ')

with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
    cuda_result = flash_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('attn values sanity check:', torch.allclose(cuda_result, manual_result, rtol=0, atol=1e-02))
