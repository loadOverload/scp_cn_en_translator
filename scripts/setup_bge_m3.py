#!/usr/bin/env python
"""Convert the cached ``BAAI/bge-m3`` checkpoint to a local safetensors directory.

Why this is needed
------------------
bge-m3 publishes ``pytorch_model.bin`` (there is no ``model.safetensors`` in the
repo), and transformers 5.x refuses to load a ``.bin`` with torch < 2.6 because
of CVE-2025-32434::

    ValueError: Due to a serious vulnerability issue in torch.load ...
                we now require users to upgrade torch to at least v2.6

Upgrading torch would drag the whole CUDA stack with it, so instead the trusted
checkpoint is converted once into the format transformers is happy to load. The
conversion also makes loading faster (safetensors is memory-mapped).

    ./py scripts/setup_bge_m3.py                 # -> models/bge-m3
    ./py scripts/setup_bge_m3.py --out DIR
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "models" / "bge-m3"
# files the encoder needs; weights are handled separately
COPY_FILES = [
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "sentencepiece.bpe.model",
    "sentence_bert_config.json", "modules.json",
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert bge-m3 to safetensors")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    target = out_dir / "model.safetensors"
    if target.exists() and not args.force:
        print(f"already converted: {target} ({target.stat().st_size / 2**30:.2f} GB)")
        return 0

    from huggingface_hub import snapshot_download

    print(f"resolving {args.model} …", flush=True)
    snapshot = Path(snapshot_download(
        args.model,
        allow_patterns=COPY_FILES + ["pytorch_model.bin", "1_Pooling/*"],
    ))

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in COPY_FILES:
        source = snapshot / name
        if source.exists():
            shutil.copy2(source, out_dir / name)
    pooling = snapshot / "1_Pooling"
    if pooling.is_dir():
        shutil.copytree(pooling, out_dir / "1_Pooling", dirs_exist_ok=True)

    weights = snapshot / "pytorch_model.bin"
    if not weights.exists():
        # a safetensors file already exists in the snapshot: just use it
        existing = list(snapshot.glob("*.safetensors"))
        if existing:
            shutil.copy2(existing[0], target)
            print(f"copied {existing[0].name} -> {target}")
            return 0
        raise SystemExit(f"no weights found in {snapshot}")

    import torch
    from safetensors.torch import save_file

    print(f"loading {weights} ({weights.stat().st_size / 2**30:.2f} GB) …", flush=True)
    # weights_only=True is the safe mode; the checkpoint comes from the official
    # BAAI repository, which is why converting it here is acceptable at all
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise SystemExit(f"unexpected checkpoint type: {type(state)}")
    # some exports wrap the model under "model."; unwrap so the keys match the
    # transformers module names
    if all(key.startswith("model.") for key in state):
        state = {key[len("model."):]: value for key, value in state.items()}
    state = {key: value.contiguous() for key, value in state.items()}

    print(f"writing {target} …", flush=True)
    save_file(state, str(target), metadata={"format": "pt", "converted_from": "pytorch_model.bin"})
    print(f"done: {target} ({target.stat().st_size / 2**30:.2f} GB), "
          f"{len(state)} tensors, dir {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
