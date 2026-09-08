# Model environments

Use separate environments. MusicGen-Chord runs on Python 3.9 and PyTorch 2.0.1;
MIDI-SAG runs on PyTorch 2.5.1. Installing both into one environment changes the
dependencies used for the experiments.

## MusicGen-Chord (Linux, NVIDIA GPU)

```bash
conda create -n chord-musicgen python=3.9 -y
conda activate chord-musicgen
python -m pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r environments/musicgen.txt
python -m pip install xformers==0.0.22 --no-deps
```

The last step keeps the existing PyTorch version. Inference uses PyTorch math
SDPA, not xFormers attention kernels. These are the versions in the experiment
environment; the installation recipe has not yet been tested in a new environment.

Clone the chord-aware AudioCraft fork, rather than installing stock AudioCraft:

```bash
git clone https://github.com/sakemin/cog-musicgen-chord external/musicgen-chord
git -C external/musicgen-chord checkout f5ad238ed9c891656d39eba2b46af449d623612a
```

The runner imports AudioCraft directly from that checkout. Download the
`musicgen-stereo-chord.th` checkpoint from the upstream MusicGen-Chord weight
distribution and the Hugging Face `t5-base` and `facebook/encodec_32khz` models.
Keep each Hugging Face model's tokenizer/configuration files beside its weights.

```bash
mkdir -p checkpoints
curl -L https://weights.replicate.delivery/default/musicgen-chord/musicgen-stereo-chord.th -o checkpoints/musicgen-stereo-chord.th
huggingface-cli download t5-base --local-dir checkpoints/t5-base
huggingface-cli download facebook/encodec_32khz --local-dir checkpoints/encodec_32khz
```

An upstream auxiliary checkpoint may be fetched on first use. For offline runs,
copy the populated Torch Hub cache from a machine with network access into
`outputs/cache/torchhub/`; the experiment used `955717e8-8726e21a.th` there.
Model weights retain their upstream licenses and are not included in this repository.

## MIDI-SAG

The experiment used PyTorch 2.5.1 with CUDA 12.1. Start its environment separately:

```bash
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environments/midi-sag.txt
```

This file pins the main inference dependencies, not the full GAME/AccoMontage2
input-preparation stack. The portable MIDI-SAG renderer and baseline preparation
are not included yet. MIDI-SAG's released measurements can already be recomputed
with `scripts/run_reproduction.py`.
