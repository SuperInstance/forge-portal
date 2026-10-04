# forge-portal — one door to every image engine the fleet has

A single dark portal in front of **every image-making technology on the boat**:

| Engine | What | Cost |
|---|---|---|
| `local` | Realistic Vision V6.0 B1 **Hyper** (pruned fp16) on the RTX 4050 via diffusers — 4–8 step generation | free, ~5–10 s hot |
| `mmx` | MiniMax image-01 via the MMX CLI | Starter plan |
| `cf` | Cloudflare Workers AI `@cf/black-forest-labs/flux-1-schnell` | free tier |
| `deepinfra` | `black-forest-labs/FLUX-2-max` — hero art only (width capped at 1440, the known gotcha) | metered, sparingly |

## Run it

```bash
/home/eileen/venvs/elephant-gpu/bin/python server.py
# → http://localhost:7860
```

Keys are **read at use-time** from `/mnt/c/Users/casey/key.txt`
(`DEEPINFRA_KEY`, `CF_API_TOKEN`, `CF_ACCOUNT_ID`) — never stored in this repo,
never logged.

## Doctrine

- **GPU guard before every local job**: refuses to fire unless ≥1 GB VRAM is
  free and the GPU is ≤80 °C — same law as quilt-gpu-lab's `guard.py`. This
  laptop has crash-looped before; the guard exists so the portal never joins in.
- **Warm start**: the pipeline loads lazily on the first local job and stays
  hot. The box's WSL2 idle-ramp behavior (INSTRUMENT-01) means the *first*
  generation after idle is the slow one — that's physics, booked honestly in
  the per-job log line (`post: …MB …C` receipt after every gen).
- **Fail loud**: backend errors land in the UI verbatim, no silent re-rolls.
- **Gallery is a receipt trail**: every landing appends to
  `gallery/gallery.jsonl` (prompt, params, backend, seed, seconds) next to its
  PNG. Append-only, nothing deleted.
- **Stdlib server on purpose** — zero new dependencies on the 6 GB boat.

## Verify

```bash
python3 smoke_local.py
# ramp receipt (0.6 s synced load) + one 8-step 512×768 generation
```

## Notes for the next engineer

- The checkpoint lives at
  `sd-fleet/models/realisticVisionV60B1_v51HyperVAE_pruned_fp16.safetensors`
  (ext4, not /mnt/c — model loads from NTFS are miserable).
- Decode is done **outside the pipeline** (`output_type="latent"` + manual
  `vae.decode`) with an fp32-VAE fallback if fp16 NaNs grey the image out —
  SD1.5 pipelines don't guard the Half/float boundary themselves (learned the
  hard way, twice, in one afternoon).
- Hyper presets: 8 steps / CFG 1.5 is the daily driver; 4 steps / CFG 1.2 for
  sprint; 24 steps / CFG 5 for when you want the detail pass.
- Backends are a dict — adding one is `async def gen_x(job) -> (bytes, fmt)`
  plus a row in `BACKENDS`. Keys via `read_key("NAME")`.
