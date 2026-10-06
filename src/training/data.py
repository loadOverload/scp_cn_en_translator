"""Chat SFT dataset + collator.

The dataset consumes the JSONL produced by ``src.data.to_chat``:

    {"id": ..., "task": "translate"|"term", "source": ..., "target": ...,
     "messages": [{"role": ..., "content": ...}, ...]}

Key behaviours
--------------
* the tokenizer's own chat template is used, so training and inference prompts
  are identical by construction
* ``train_on_assistant_only`` masks everything except the assistant answer, so
  the loss is computed on the translation, not on the prompt
* samples longer than ``max_seq_length`` are dropped (default) or truncated,
  and the count is reported -- never silently
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..utils.io import read_jsonl


class ChatSFTDataset:
    """Lazy, map-style dataset over chat-format JSONL."""

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        max_seq_length: int = 4096,
        overlong_policy: str = "drop",
        train_on_assistant_only: bool = True,
        limit: int = 0,
        name: str = "train",
        packing: bool = False,
        cache_dir: str | Path | None = None,
        tokenize_workers: int = 0,
        logger=None,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_length = int(max_seq_length)
        self.overlong_policy = overlong_policy
        self.train_on_assistant_only = bool(train_on_assistant_only)
        self.name = name
        self.packing = bool(packing)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        # 0 means "decide from the machine"; negative disables parallelism.
        self.tokenize_workers = int(tokenize_workers)
        self.limit = int(limit)
        self.logger = logger
        self.rows: List[Dict[str, Any]] = []
        self.skipped_overlong = 0
        self.skipped_empty = 0
        self.truncated = 0
        self._load(limit)
        self._select_valid()
        if self.packing:
            self._pack()

    # -- loading -----------------------------------------------------------
    def _load(self, limit: int) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.name} data file not found: {self.path}\n"
                "run `python scripts/prepare_data.py` first"
            )
        rows = list(read_jsonl(self.path))
        if limit:
            rows = rows[:limit]
        self.rows = rows

    def _messages(self, row: Mapping[str, Any]) -> List[Dict[str, str]]:
        messages = row.get("messages")
        if isinstance(messages, list) and messages:
            return [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]
        raise ValueError(f"{self.path}: sample {row.get('id')!r} has no 'messages' field")

    # -- tokenization ------------------------------------------------------
    def _render(self, messages: Sequence[Mapping[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=False
        )

    def encode(self, row: Mapping[str, Any]) -> Dict[str, List[int]]:
        messages = self._messages(row)
        text = self._render(messages)
        enc = self.tokenizer(text, add_special_tokens=False, truncation=False)
        input_ids: List[int] = list(enc["input_ids"])

        labels = list(input_ids)
        if self.train_on_assistant_only:
            prompt_len = self._prompt_length(messages)
            if 0 < prompt_len <= len(labels):
                labels = [-100] * prompt_len + labels[prompt_len:]
            # prompt_len == 0 means the template could not be split; keep full loss

        if len(input_ids) > self.max_seq_length:
            if self.overlong_policy == "truncate":
                input_ids = input_ids[: self.max_seq_length]
                labels = labels[: self.max_seq_length]
            else:
                raise _OverlongError()

        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

    def _prompt_length(self, messages: Sequence[Mapping[str, str]]) -> int:
        """Tokens that belong to system+user (i.e. must not contribute to loss)."""
        prefix = list(messages[:-1]) if messages and messages[-1]["role"] == "assistant" else list(messages)
        if not prefix:
            return 0
        try:
            prompt_text = self.tokenizer.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True)
        except Exception:
            return 0
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False, truncation=False)["input_ids"]
        return len(prompt_ids)

    # -- index selection ---------------------------------------------------
    def _cache_key(self) -> str:
        """Identity of everything that can change the tokenized output.

        The file's size and mtime are part of the key, so regenerating the
        dataset invalidates the cache without anyone remembering to clear it.
        """
        stat = self.path.stat()
        try:
            vocab = int(len(self.tokenizer))
        except Exception:
            vocab = -1
        parts = [
            str(self.path.resolve()), str(stat.st_size), str(stat.st_mtime_ns),
            str(self.max_seq_length), str(self.overlong_policy),
            str(self.train_on_assistant_only), str(self.limit),
            str(getattr(self.tokenizer, "name_or_path", "") or ""), str(vocab),
        ]
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]

    def _cache_file(self) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self.path.stem}.{self.name}.{self._cache_key()}.npz"

    def _encode_one(self, item):
        """Tokenize one row and never raise.

        Returns ``(status, index, encoded)`` where status is one of ``ok``,
        ``overlong``, ``truncated``, ``empty`` or ``malformed``.
        """
        index, row = item
        try:
            encoded = self.encode(row)
        except _OverlongError:
            if self.overlong_policy == "drop":
                return ("overlong", index, None)
            try:
                messages = self._messages(row)
                text = self._render(messages)
                ids = list(self.tokenizer(text, add_special_tokens=False,
                                          truncation=False)["input_ids"])[: self.max_seq_length]
            except Exception:
                return ("malformed", index, None)
            encoded = {"input_ids": ids, "labels": ids, "attention_mask": [1] * len(ids)}
            status = "truncated"
        except Exception as exc:  # malformed row
            return ("malformed", index, exc)
        ids = encoded["input_ids"]
        if not ids:
            return ("empty", index, None)
        return ("ok", index, encoded)

    def _tokenize_all(self) -> None:
        """Tokenize every row, in parallel across threads.

        The tokenizer is a Rust ``tokenizers`` object which releases the GIL
        while encoding, so threads give near-linear speedup here -- no process
        pool, no pickling of the tokenizer, no copying rows between processes.
        On the long set (8,184 samples, 49M tokens) this is the difference
        between roughly four minutes of a completely idle GPU and under one.
        """
        import os

        rows = list(enumerate(self.rows))
        workers = self.tokenize_workers
        if workers == 0:
            workers = min(16, max(1, (os.cpu_count() or 4)))
        if workers < 0:
            workers = 1

        started = time.time()
        if workers <= 1 or len(rows) < 32:
            results = [self._encode_one(item) for item in rows]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(self._encode_one, rows))

        self.valid_indices = []
        self.encoded = []
        for status, index, payload in results:
            if status == "overlong":
                self.skipped_overlong += 1
                continue
            if status == "malformed":
                self.skipped_empty += 1
                if self.logger and isinstance(payload, Exception):
                    self.logger.warning("skipping malformed sample %s: %s",
                                        self.rows[index].get("id"), payload)
                continue
            if status == "empty":
                self.skipped_empty += 1
                continue
            if status == "truncated":
                self.truncated += 1
            encoded = payload
            ids = encoded["input_ids"]
            self.valid_indices.append(index)
            self.encoded.append({
                "input_ids": np.asarray(ids, dtype=np.int32),
                "labels": np.asarray(encoded["labels"], dtype=np.int32),
                "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int32),
                "id": self.rows[index].get("id", f"row-{index}"),
            })
        self._tokenize_seconds = time.time() - started
        self._log_selection(tokenized=True, cache_state="built")

    def _log_selection(self, tokenized: bool, cache_state: str) -> None:
        if not self.logger:
            return
        share = 100 * self.skipped_overlong / max(len(self.rows), 1)
        self.logger.info(
            "%s: %d/%d samples usable (dropped %d over %d tokens = %.1f%%, "
            "%d truncated, %d malformed) [cache %s%s]",
            self.name, len(self.valid_indices), len(self.rows),
            self.skipped_overlong, self.max_seq_length, share,
            self.truncated, self.skipped_empty, cache_state,
            f", tokenized in {self._tokenize_seconds:.1f}s" if tokenized else "",
        )

    # -- tokenized cache ---------------------------------------------------
    def _save_cache(self) -> None:
        target = self._cache_file()
        if target is None:
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            offsets = np.zeros(len(self.encoded) + 1, dtype=np.int64)
            for i, item in enumerate(self.encoded):
                offsets[i + 1] = offsets[i] + len(item["input_ids"])
            flat_ids = np.concatenate([i["input_ids"] for i in self.encoded]) \
                if self.encoded else np.zeros(0, dtype=np.int32)
            flat_labels = np.concatenate([i["labels"] for i in self.encoded]) \
                if self.encoded else np.zeros(0, dtype=np.int32)
            flat_mask = np.concatenate([i["attention_mask"] for i in self.encoded]) \
                if self.encoded else np.zeros(0, dtype=np.int32)
            tmp = target.with_name(target.name + ".part")
            # np.savez appends ".npz" when given a path, which silently made the
            # atomic rename look for a file that was never created. Writing
            # through an open handle keeps the name exactly as asked.
            with open(tmp, "wb") as handle:
                np.savez(handle,
                         flat_ids=flat_ids, flat_labels=flat_labels, flat_mask=flat_mask,
                         offsets=offsets,
                         sample_ids=np.asarray([i["id"] for i in self.encoded], dtype=object),
                         valid_indices=np.asarray(self.valid_indices, dtype=np.int64),
                         counters=np.asarray(
                             [self.skipped_overlong, self.skipped_empty, self.truncated],
                             dtype=np.int64))
            tmp.replace(target)
            if self.logger:
                self.logger.info("%s: tokenized cache written -> %s (%.1f MB)",
                                 self.name, target.name, target.stat().st_size / 1e6)
        except Exception as exc:  # a cache that cannot be written is not fatal
            if self.logger:
                self.logger.warning("%s: could not write tokenized cache: %s", self.name, exc)

    def _load_cache(self) -> bool:
        target = self._cache_file()
        if target is None or not target.exists():
            return False
        try:
            with np.load(target, allow_pickle=True) as data:
                # materialise once: slicing an NpzFile member re-reads the
                # archive, and there is one slice per sample.
                flat_ids = np.array(data["flat_ids"])
                flat_labels = np.array(data["flat_labels"])
                flat_mask = np.array(data["flat_mask"])
                offsets = np.array(data["offsets"])
                sample_ids = np.array(data["sample_ids"], dtype=object)
                valid_indices = np.array(data["valid_indices"])
                counters = np.array(data["counters"])
            self.encoded = []
            for i in range(len(offsets) - 1):
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                self.encoded.append({
                    "input_ids": flat_ids[lo:hi].astype(np.int32, copy=False),
                    "labels": flat_labels[lo:hi].astype(np.int32, copy=False),
                    "attention_mask": flat_mask[lo:hi].astype(np.int32, copy=False),
                    "id": str(sample_ids[i]),
                })
            self.valid_indices = [int(x) for x in valid_indices]
            self.skipped_overlong, self.skipped_empty, self.truncated = \
                (int(x) for x in counters)
            self._tokenize_seconds = 0.0
            self._log_selection(tokenized=False, cache_state="hit")
            return True
        except Exception as exc:
            if self.logger:
                self.logger.warning("%s: tokenized cache unreadable (%s); rebuilding",
                                    self.name, exc)
            return False

    def _select_valid(self) -> None:
        """Tokenize every sample once, keeping the result (parallel + cached).

        Tokenizing here rather than in ``__getitem__`` matters: the old version
        tokenized twice per sample, inside the dataloader, while the GPU waited
        (two chat-template renders and two tokenizations, because
        ``_prompt_length`` rendered and tokenized the prompt separately). On the
        short set that was the difference between a two-hour run and a
        twenty-hour one.

        The result is cached on disk keyed by file size+mtime, sequence budget,
        overlong policy and tokenizer, so restarting a run no longer pays the
        tokenization cost again. Samples are held as int32 arrays: 49M tokens
        costs ~600 MB against ~3 GB as Python int lists.
        """
        self.valid_indices: List[int] = []
        self.encoded: List[Dict[str, Any]] = []
        self._tokenize_seconds = 0.0
        if self._load_cache():
            return
        self._tokenize_all()
        self._save_cache()

    @property
    def lengths(self) -> List[int]:
        """Token length of every usable sample, for the length-grouped sampler."""
        return [int(len(item["input_ids"])) for item in self.encoded]

    # -- packing -----------------------------------------------------------
    def _pack(self) -> None:
        """Concatenate samples into fixed-length sequences.

        Short samples are individually wasteful: a batch of them pads to its
        longest member (measured at 47% padding on this data) and each forward
        pass is too small to use the GPU well -- batch 8 cost 0.3 s per sample
        against 0.07 s for a 2,000-token sample, i.e. 1,880 padded tokens/s
        instead of 6,940. Packing fixes both at once: every sequence is exactly
        ``max_seq_length``, so there is no padding and no short-sequence
        inefficiency.

        The assistant-only mask survives packing, because it is a per-token
        label: a ``-100`` run in the middle of a packed sequence still means
        "this was the prompt", which is what the loss reads.
        """
        budget = self.max_seq_length
        packed: List[Dict[str, Any]] = []
        ids: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        current = 0
        for item in self.encoded:
            length = len(item["input_ids"])
            if current + length > budget and ids:
                packed.append({
                    "input_ids": np.concatenate(ids),
                    "labels": np.concatenate(labels),
                    "attention_mask": np.ones(current, dtype=np.int32),
                    "id": f"packed-{len(packed)}",
                })
                ids, labels, current = [], [], 0
            ids.append(item["input_ids"])
            labels.append(item["labels"])
            current += length
        if ids:
            packed.append({
                "input_ids": np.concatenate(ids),
                "labels": np.concatenate(labels),
                "attention_mask": np.ones(current, dtype=np.int32),
                "id": f"packed-{len(packed)}",
            })
        if self.logger:
            self.logger.info("%s: packed %d samples into %d sequences of <= %d tokens",
                             self.name, len(self.encoded), len(packed), budget)
        self.encoded = packed
        self.valid_indices = list(range(len(packed)))

    # -- Dataset protocol --------------------------------------------------
    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # tokenized once at load time; nothing to redo per epoch
        return self.encoded[index]

    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "rows": len(self.rows),
            "usable": len(self.valid_indices),
            "skipped_overlong": self.skipped_overlong,
            "truncated": self.truncated,
            "malformed": self.skipped_empty,
            "max_seq_length": self.max_seq_length,
            "overlong_policy": self.overlong_policy,
        }

    def to_hf_dataset(self):
        """Materialise into a ``datasets.Dataset`` (required by TRL's SFTTrainer).

        Only used when ``training.trainer: trl`` is explicitly requested: the
        tokenised samples are held in memory, which is fine for a bounded corpus
        but pointless overhead for the default native trainer.
        """
        from datasets import Dataset

        records = []
        for index in range(len(self.valid_indices)):
            item = self[index]
            records.append(
                {
                    "input_ids": list(item["input_ids"]),
                    "labels": list(item["labels"]),
                    "attention_mask": list(item["attention_mask"]),
                }
            )
        return Dataset.from_list(records)


class _OverlongError(Exception):
    pass


class DataCollatorForChatSFT:
    """Pad a batch of tokenized chat samples (dynamic padding)."""

    def __init__(self, tokenizer, pad_to_multiple_of: int = 8, label_pad_token_id: int = -100) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        input_ids, labels, attention_mask = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            lbl = list(f["labels"])
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append(lbl + [self.label_pad_token_id] * pad)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        # dense (non-nested) batch: let plain tensors through, the DataLoader
        # handles pin_memory when dataloader_pin_memory is enabled
        return batch


class LengthBucketedBatchSampler(Sampler):
    """Batches of near-identical length, with the batch order shuffled.

    The collator pads every member of a batch to the longest one, so a batch that
    mixes a 6,000-token sample with seven 300-token samples spends most of its
    forward pass on padding. Measured on the short set: **47% of every batch was
    padding** when samples were taken in file order.

    ``transformers``' own ``LengthGroupedSampler`` sorts inside a window of
    ``mega_batch_mult * batch_size`` (100 by default) and shuffles within it, and
    that was not enough here -- batches still mixed very different lengths and the
    achieved rate stayed at ~1,600 real tokens/s against ~7,000 for a
    length-homogeneous run, a factor of four.

    This sorts the whole index list by length and cuts it into consecutive
    batches, so every batch is as homogeneous as the data allows; only the order
    of the batches is shuffled, once per epoch, which is all the randomness
    training needs. The cost is that samples are not i.i.d. across a step, which
    is the standard trade for padding efficiency.
    """

    def __init__(self, lengths, batch_size: int, drop_last: bool = False,
                 seed: int = 42, shuffle: bool = True) -> None:
        self.lengths = [int(x) for x in lengths]
        self.batch_size = max(1, int(batch_size))
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self._batches: List[List[int]] = []

    def _build(self) -> None:
        order = sorted(range(len(self.lengths)), key=lambda i: (self.lengths[i], i))
        batches = [order[i:i + self.batch_size]
                   for i in range(0, len(order), self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        if self.shuffle:
            import random
            random.Random(self.seed + self.epoch).shuffle(batches)
        self._batches = batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._build()

    def __iter__(self):
        if not self._batches:
            self._build()
        return iter(self._batches)

    def __len__(self) -> int:
        if not self._batches:
            self._build()
        return len(self._batches)
