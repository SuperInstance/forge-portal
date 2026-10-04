"""Forge smoke: INSTRUMENT-01 ramp receipt + one hot local generation."""
import sys, time, subprocess
sys.path.insert(0, "/home/eileen/projects/forge-portal")
from server import LOCAL, gpu_state

free0, temp0 = gpu_state()
print(f"pre:  {free0}MB free, {temp0}C")
# INSTRUMENT-01: 0.6s synced load to exit the idle-ramp before measuring
import torch
if torch.cuda.is_available():
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    t0 = time.time()
    while time.time() - t0 < 0.6:
        a = a @ a * 0.01
    torch.cuda.synchronize()
    print(f"ramp: 0.6s synced load done")

class Job: pass
job = {"prompt": "a small brass ship lamp glowing amber on a dark navy rail, minimalist, photorealistic",
       "negative": "blurry, low quality", "width": 512, "height": 768,
       "steps": 8, "cfg": 1.5, "seed": 42}
t0 = time.time()
img = LOCAL.generate(job, lambda n, t: print(f"  step {n}/{t}", flush=True))
dt = time.time() - t0
free1, temp1 = gpu_state()
img.save("/home/eileen/projects/forge-portal/gallery/smoke.png")
print(f"gen:  {dt:.1f}s for 8 steps @ 512x768 (hyper)")
print(f"post: {free1}MB free, {temp1}C")
print("SMOKE-PASS" if dt < 60 else "SMOKE-SLOW")
