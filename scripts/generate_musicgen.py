#!/usr/bin/env python3
"""Generate paired MusicGen-Chord outputs from a JSON manifest."""
from __future__ import annotations
import argparse, hashlib, json, os, time, random
from pathlib import Path
import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_sha(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t5", type=Path, required=True)
    ap.add_argument("--encodec", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2027])
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument(
        "--device", default=None, help="cuda device accepted by AudioCraft, e.g. cuda:1"
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--repeat-check", action="store_true")
    args = ap.parse_args()
    args.repo = args.repo.resolve()
    args.weights = args.weights.resolve()
    args.manifest = args.manifest.resolve()
    args.out = args.out.resolve()
    args.t5 = args.t5.resolve()
    args.encodec = args.encodec.resolve()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")
    required = {
        "repository": args.repo,
        "weights": args.weights,
        "manifest": args.manifest,
        "T5 assets": args.t5,
        "EnCodec assets": args.encodec,
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise SystemExit("missing required local assets: " + "; ".join(missing))
    cache = args.out.parent / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache)
    os.environ["TRANSFORMERS_CACHE"] = str(cache)
    os.chdir(args.repo)
    import sys, types

    torch.hub.set_dir(str(cache / "torchhub"))
    sys.path.insert(0, str(args.repo))
    # The public repository wraps inference in Cog; local smoke only needs its
    # checkpoint loader, so provide the minimal import-time symbols.
    cog = types.ModuleType("cog")
    cog.BasePredictor = object
    cog.Input = lambda *a, **k: None
    cog.Path = Path
    sys.modules["cog"] = cog
    import predict
    from omegaconf import OmegaConf

    original_builder = predict.get_lm_model

    def device_builder(config):
        # Upstream load_ckpt builds from the unmodified training config.
        config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        config.device = device
        config.dtype = "float16" if device.startswith("cuda") else "float32"
        for field in (
            "conditioners.self_wav.chroma_chord.cache_path",
            "conditioners.self_wav.chroma_stem.cache_path",
            "conditioners.args.merge_text_conditions_p",
            "conditioners.args.drop_desc_p",
        ):
            predict._delete_param(config, field)
        return original_builder(config)

    predict.get_lm_model = device_builder
    from audiocraft.solvers.compression import CompressionSolver
    from audiocraft.models.encodec import HFEncodecModel

    _hf_encodec_loader = HFEncodecModel.from_pretrained
    HFEncodecModel.from_pretrained = classmethod(
        lambda cls, name, *a, **k: _hf_encodec_loader(str(args.encodec), *a, **k)
    )
    _compression_loader = CompressionSolver.model_from_checkpoint

    def _local_compression(checkpoint_path, device="cpu"):
        if "facebook/encodec_32khz" in str(checkpoint_path):
            from audiocraft.models.encodec import CompressionModel

            return CompressionModel.get_pretrained(
                "facebook/encodec_32khz", device=device
            )
        return _compression_loader(checkpoint_path, device)

    CompressionSolver.model_from_checkpoint = staticmethod(_local_compression)
    from transformers import T5Tokenizer, T5EncoderModel

    # Preserve the original encoder architecture while resolving it from the
    # locally mirrored official T5-base assets (the server cannot reach HF).
    _tok_loader = T5Tokenizer.from_pretrained.__func__
    _enc_loader = T5EncoderModel.from_pretrained.__func__
    T5Tokenizer.from_pretrained = classmethod(
        lambda cls, name, *a, **k: _tok_loader(cls, str(args.t5), *a, **k)
    )
    T5EncoderModel.from_pretrained = classmethod(
        lambda cls, name, *a, **k: _enc_loader(cls, str(args.t5), *a, **k)
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"{device} requested but CUDA is unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch 2.0.1's deterministic advanced-index assignment crashes in the
    # upstream T5 tokenizer. Pin math SDPA and verify repetitions directly.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    seed_all(0)
    model = predict.load_ckpt(str(args.weights), device, False)
    model.lm.eval()
    model.compression_model.eval()
    model.lm.condition_provider.conditioners["self_wav"].match_len_on_eval = True
    for conditioner in model.lm.condition_provider.conditioners.values():
        if hasattr(conditioner, "t5"):
            conditioner.t5.eval()
    jobs = json.loads(args.manifest.read_text(encoding="utf-8"))
    tracks = []
    [tracks.append(j["track_key"]) for j in jobs if j["track_key"] not in tracks]
    tracks = tracks[: args.limit]
    tracks = [
        track
        for i, track in enumerate(tracks)
        if i % args.num_shards == args.shard_index
    ]
    if not tracks:
        raise SystemExit("this shard has no tracks")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for track in tracks:
        for j in [x for x in jobs if x["track_key"] == track]:
            for seed in args.seeds:
                seed_all(seed)
                model.set_generation_params(
                    duration=int(j["duration"]),
                    use_sampling=args.temperature > 0,
                    top_k=250 if args.temperature > 0 else 0,
                    top_p=0.0,
                    temperature=args.temperature,
                    cfg_coef=3,
                )
                t = time.time()
                with torch.inference_mode():
                    wav, tok = model.generate_with_text_chroma(
                        [j["prompt"]],
                        [j["chord_text"]],
                        bpm=[j["bpm"]],
                        meter=[j["meter"]],
                        progress=False,
                        return_tokens=True,
                    )
                elapsed = time.time() - t
                arr = wav.detach().float().cpu().numpy()
                finite = bool(np.isfinite(arr).all())
                peak = float(np.max(np.abs(arr)))
                duration = float(arr.shape[-1] / model.sample_rate)
                rms = float(np.sqrt(np.mean(arr**2)))
                if not finite or rms < 1e-5 or abs(duration - j["duration"]) > 0.01:
                    raise RuntimeError(
                        f'audio integrity failed: {track} {j["condition"]} {seed}'
                    )
                repeat = None
                if args.repeat_check and not results:
                    seed_all(seed)
                    with torch.inference_mode():
                        repeated, repeated_tokens = model.generate_with_text_chroma(
                            [j["prompt"]],
                            [j["chord_text"]],
                            bpm=[j["bpm"]],
                            meter=[j["meter"]],
                            progress=False,
                            return_tokens=True,
                        )
                    repeat = {
                        "tokens_equal": bool(torch.equal(tok, repeated_tokens)),
                        "token_equal_fraction": float(
                            (tok == repeated_tokens).float().mean()
                        ),
                        "waveform_equal": bool(torch.equal(wav, repeated)),
                        "waveform_max_abs_error": float((wav - repeated).abs().max()),
                    }
                    if not repeat["tokens_equal"]:
                        raise RuntimeError(
                            f"same-condition token reproducibility failed: {repeat}"
                        )
                path = args.out / f"{track}__{j['condition']}__S{seed}.pt"
                torch.save(wav.detach().cpu(), path)
                results.append(
                    {
                        "track_key": track,
                        "condition": j["condition"],
                        "seed": seed,
                        "path": str(path),
                        "sha256": sha(path),
                        "tensor_sha256": tensor_sha(wav),
                        "token_sha256": tensor_sha(tok),
                        "repeat_check": repeat,
                        "chord_text": j["chord_text"],
                        "shape": list(arr.shape),
                        "sample_rate": model.sample_rate,
                        "duration_sec": duration,
                        "peak": peak,
                        "rms": rms,
                        "finite": finite,
                        "wall_sec": elapsed,
                        "gpu_peak_alloc_mb": (
                            torch.cuda.max_memory_allocated() / 2**20
                            if torch.cuda.is_available()
                            else 0.0
                        ),
                    }
                )
                print(
                    f'{track} {j["condition"]} S{seed}: {elapsed:.1f}s repeat={repeat}',
                    flush=True,
                )
                torch.cuda.empty_cache()
    result_path = args.out / f"results_shard{args.shard_index:02d}.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "device": device,
                "weights_sha256": sha(args.weights),
                "manifest": str(args.manifest),
                "tracks": tracks,
                "seeds": args.seeds,
                "manifest_sha256": sha(args.manifest),
                "runner_sha256": sha(Path(__file__).resolve()),
                "temperature": args.temperature,
                "top_k": 250 if args.temperature > 0 else 0,
                "model_initialization_seed": 0,
                "lm_device": str(next(model.lm.parameters()).device),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "outputs": len(results),
                "device": device,
                "result_path": str(result_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
