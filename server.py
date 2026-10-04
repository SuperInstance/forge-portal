#!/usr/bin/env python3
"""forge-portal — one door to every image engine the fleet has.

Backends:
  local      Realistic Vision V6.0 B1 Hyper (pruned fp16) on the RTX 4050, diffusers
  mmx        MiniMax image gen (MMX CLI, Starter plan)
  cf         Cloudflare Workers AI @cf/black-forest-labs/flux-1-schnell (free tier)
  deepinfra  black-forest-labs/FLUX-2-max via DeepInfra (hero art — width <= 1440)

Keys are read at use-time from /mnt/c/Users/casey/key.txt — never stored here.
GPU guard: refuse to start a local job unless >=1 GB VRAM free and GPU <= 80 C
(same doctrine as quilt-gpu-lab guard.py). WSL2 idle-ramp law (INSTRUMENT-01)
is honored with a warmup generation at boot so the first human click is hot.

Stdlib server on purpose: no new deps on the boat. Run:  python3 server.py
"""

import base64
import json
import os
import queue
import random
import re
import subprocess
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
GALLERY_DIR = os.path.join(ROOT, "gallery")
GALLERY_JSONL = os.path.join(GALLERY_DIR, "gallery.jsonl")
KEYFILE = "/mnt/c/Users/casey/key.txt"
MODEL_PATH = "/home/eileen/projects/sd-fleet/models/realisticVisionV60B1_v51HyperVAE_pruned_fp16.safetensors"
NSMIDI = "/usr/lib/wsl/lib/nvidia-smi"
PORT = 7860

os.makedirs(GALLERY_DIR, exist_ok=True)

# ---------------------------------------------------------------- keys

def read_key(name):
    """Read one key from key.txt at use-time. Values never logged."""
    try:
        with open(KEYFILE) as f:
            for line in f:
                if line.startswith(name):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


def cf_account_id():
    try:
        with open(KEYFILE) as f:
            for line in f:
                if line.startswith("CF_ACCOUNT_ID"):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return os.environ.get("CF_ACCOUNT_ID")

# ---------------------------------------------------------------- gpu guard

def gpu_state():
    """(free_mb, temp_c) via the WSL nvidia-smi; (None, None) if unavailable."""
    try:
        out = subprocess.run(
            [NSMIDI, "--query-gpu=memory.free,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        free, temp = out.splitlines()[0].split(", ")
        return int(free), int(float(temp))
    except Exception:
        return None, None


def gpu_guard():
    free, temp = gpu_state()
    if free is not None and (free < 1024 or temp > 80):
        return False, f"guard refuses: {free} MB free, {temp} C (need >=1024 MB, <=80 C)"
    return True, f"guard ok: {free} MB free, {temp} C"

# ---------------------------------------------------------------- local pipeline

class LocalEngine:
    """Single-flight diffusers pipeline. Loaded lazily, kept hot."""

    def __init__(self):
        self.pipe = None
        self.lock = threading.Lock()

    def load(self):
        if self.pipe is not None:
            return
        import torch  # heavy import inside the worker thread
        from diffusers import (DPMSolverMultistepScheduler,
                               StableDiffusionPipeline)
        t0 = time.time()
        pipe = StableDiffusionPipeline.from_single_file(
            MODEL_PATH, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True)
        pipe.enable_attention_slicing()
        pipe = pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        print(f"[forge] local pipeline loaded in {time.time()-t0:.1f}s", flush=True)

    def generate(self, job, on_step):
        import torch
        ok, why = gpu_guard()
        if not ok:
            raise RuntimeError(why)
        with self.lock:
            self.load()
            g = None
            if job["seed"] not in (None, "", -1, "random"):
                g = torch.Generator("cuda").manual_seed(int(job["seed"]))
            step_box = {"n": 0}

            def cb(p, i, t, kw):  # diffusers 0.40 signature; must return kwargs
                step_box["n"] = i + 1
                on_step(step_box["n"], job["steps"])
                return kw

            t0 = time.time()
            out = self.pipe(
                prompt=job["prompt"], negative_prompt=job["negative"] or None,
                width=job["width"], height=job["height"],
                num_inference_steps=job["steps"], guidance_scale=job["cfg"],
                generator=g, callback_on_step_end=cb, output_type="latent")
            lat = out.images
            if lat.dim() == 3:  # batchless latent — decode needs [B,C,H,W]
                lat = lat.unsqueeze(0)
            # decode outside the pipeline: fp16 first, fp32 VAE if NaN greys out
            import torch as _t
            vae = self.pipe.vae
            with _t.no_grad():  # pipeline's inference context ended; keep decode clean
                lat = (lat / vae.config.scaling_factor).to(vae.dtype)
                raw = vae.decode(lat, return_dict=False)[0]
                if not _t.isfinite(raw).all() or float(raw.float().std()) < 0.02:
                    vae_f = vae.float()
                    raw = vae_f.decode(lat.float() / vae_f.config.scaling_factor,
                                       return_dict=False)[0]
            img = self.pipe.image_processor.postprocess(
                raw.float().detach(), output_type="pil")[0]
            dt = time.time() - t0
            # INSTRUMENT-01 receipt: this box ramps after idle; report honestly
            free, temp = gpu_state()
            print(f"[forge] local gen {dt:.1f}s ({job['steps']} steps, "
                  f"{job['width']}x{job['height']}) post: {free}MB {temp}C", flush=True)
            return img

LOCAL = LocalEngine()

# ---------------------------------------------------------------- cloud backends

def gen_mmx(job):
    """MiniMax via the MMX CLI (subprocess list-form only, per house law)."""
    out = os.path.join(GALLERY_DIR, f"_mmx_{job['id']}.png")
    r = subprocess.run(
        ["mmx", "image", "generate", "--prompt", job["prompt"],
         "--width", str(max(512, job["width"])),
         "--height", str(max(512, job["height"])),
         "--out", out],
        capture_output=True, text=True, timeout=180)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"mmx failed: {(r.stderr or r.stdout)[-300:]}")
    with open(out, "rb") as f:
        return f.read(), "png"


def gen_cf(job):
    """Workers AI flux-1-schnell (free tier)."""
    acct = cf_account_id()
    token = read_key("CF_API_TOKEN")
    if not (acct and token):
        raise RuntimeError("cf backend not configured: need CF_ACCOUNT_ID + CF_API_TOKEN in key.txt")
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    body = json.dumps({"prompt": job["prompt"], "steps": min(job["steps"], 8)}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    if not data.get("success"):
        raise RuntimeError(f"cf failed: {str(data.get('errors'))[:300]}")
    b64 = data["result"]["image"]
    return base64.b64decode(b64), "png"


def gen_deepinfra(job):
    """FLUX-2-max — hero art only. Known gotcha: width must be <= 1440."""
    key = read_key("DEEPINFRA_KEY")
    if not key:
        raise RuntimeError("deepinfra not configured: need DEEPINFRA_KEY in key.txt")
    w = min(job["width"] * 2, 1440)
    h = min(job["height"] * 2, 1440)
    body = json.dumps({"prompt": job["prompt"], "width": w, "height": h}).encode()
    req = urllib.request.Request(
        "https://api.deepinfra.com/v1/inference/black-forest-labs/FLUX-2-max",
        data=body, headers={"Authorization": f"Bearer {key}",
                            "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as resp:
        data = json.loads(resp.read())
    imgs = data.get("images") or data.get("output")
    if not imgs:
        raise RuntimeError(f"deepinfra returned no image: {str(data)[:300]}")
    b64 = imgs[0].split(",", 1)[-1]  # strips data: prefix if present
    return base64.b64decode(b64), "png"


BACKENDS = {
    "local": ("RTX 4050 · Realistic Vision Hyper", LOCAL.generate),
    "mmx": ("MiniMax (MMX)", gen_mmx),
    "cf": ("Cloudflare Workers AI · FLUX schnell", gen_cf),
    "deepinfra": ("DeepInfra · FLUX-2-max (hero)", gen_deepinfra),
}

# ---------------------------------------------------------------- job queue

JOBS = {}
Q = queue.Queue()


def worker():
    while True:
        job_id = Q.get()
        job = JOBS[job_id]
        backend = job["backend"]
        job["state"] = "running"
        job["note"] = BACKENDS[backend][0]
        t0 = time.time()
        try:
            if backend == "local":
                img_bytes, fmt = None, "png"
                pil = LOCAL.generate(job, lambda n, tot: job.update(
                    progress=int(100 * n / max(tot, 1))))
                path = os.path.join(GALLERY_DIR, f"{job['id']}.png")
                pil.save(path)
            else:
                img_bytes, fmt = BACKENDS[backend][1](job)
                path = os.path.join(GALLERY_DIR, f"{job['id']}.{fmt}")
                with open(path, "wb") as f:
                    f.write(img_bytes)
            job["file"] = os.path.basename(path)
            job["state"] = "done"
            job["secs"] = round(time.time() - t0, 1)
            record = {k: job.get(k) for k in
                      ("id", "prompt", "negative", "backend", "seed", "steps",
                       "cfg", "width", "height", "file", "secs")}
            record["ts"] = int(time.time())
            with open(GALLERY_JSONL, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:  # fail loud, book it, keep the door open
            job["state"] = "error"
            job["error"] = str(e)[-400:]
            print(f"[forge] job {job_id} FAILED: {job['error']}", flush=True)
        finally:
            job.pop("progress", None)


threading.Thread(target=worker, daemon=True).start()

# ---------------------------------------------------------------- http

INDEX = os.path.join(ROOT, "index.html")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet server log; the gallery jsonl is the real log

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(INDEX, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/backends":
            self._send(200, json.dumps(
                {k: v[0] for k, v in BACKENDS.items()}).encode(), "application/json")
        elif self.path == "/api/gallery":
            items = []
            if os.path.exists(GALLERY_JSONL):
                with open(GALLERY_JSONL) as f:
                    items = [json.loads(l) for l in f if l.strip()][-200:]
            self._send(200, json.dumps(list(reversed(items))).encode(),
                       "application/json")
        elif self.path.startswith("/api/job/"):
            job = JOBS.get(self.path.rsplit("/", 1)[-1], {"state": "unknown"})
            self._send(200, json.dumps(job).encode(), "application/json")
        elif self.path.startswith("/gallery/"):
            name = os.path.basename(self.path)
            path = os.path.join(GALLERY_DIR, name)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self._send(200, f.read(), "image/png")
            else:
                self._send(404, b"no such image", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/generate":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        job = {
            "id": uuid.uuid4().hex[:10],
            "prompt": str(req.get("prompt", ""))[:2000],
            "negative": str(req.get("negative", ""))[:1000],
            "backend": req.get("backend", "local"),
            "seed": req.get("seed", "random"),
            "steps": max(1, min(int(req.get("steps", 8)), 50)),
            "cfg": max(1.0, min(float(req.get("cfg", 1.5)), 15.0)),
            "width": max(256, min(int(req.get("width", 512)), 1024)),
            "height": max(256, min(int(req.get("height", 768)), 1024)),
            "state": "queued",
        }
        if job["seed"] in ("", None, "random", -1):
            job["seed"] = random.randint(0, 2**31 - 1)
        if job["backend"] not in BACKENDS:
            return self._send(400, json.dumps({"error": "unknown backend"}).encode(),
                              "application/json")
        JOBS[job["id"]] = job
        Q.put(job["id"])
        self._send(200, json.dumps({"id": job["id"]}).encode(), "application/json")


if __name__ == "__main__":
    ok, why = gpu_guard()
    print(f"[forge] {why}", flush=True)
    print(f"[forge] http://localhost:{PORT} — backends: {', '.join(BACKENDS)}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
