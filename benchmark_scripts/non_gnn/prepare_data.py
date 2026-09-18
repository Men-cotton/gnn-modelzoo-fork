"""Reusable, verified WikiText -> BERT CSV / Llama token-corpus preparation."""

import csv
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import tempfile
import uuid

FORMAT_VERSION = 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_spec(args, profiles, vocab):
    lengths = sorted(
        {
            profiles[name]["sequence_length"]
            for name in args.only
            if name.startswith("bert_")
        }
    )
    needs_llama = any(name.startswith("llama") for name in args.only)
    local = args.raw_text.resolve() if args.raw_text else None
    tokenizer = Path(args.llama_tokenizer)
    tokenizer_files = None
    if needs_llama and tokenizer.is_dir():
        tokenizer_files = {
            p.name: sha256(p)
            for p in sorted(tokenizer.iterdir())
            if p.is_file() and p.suffix in (".json", ".model", ".txt")
        }
    return {
        "format_version": FORMAT_VERSION,
        "source": (
            {"raw_text": str(local), "sha256": sha256(local)}
            if local
            else {
                "dataset": args.dataset,
                "config": args.dataset_config,
                "split": "train",
                "revision": args.dataset_revision,
            }
        ),
        "max_documents": args.max_documents,
        "seed": args.seed,
        "bert_lengths": lengths,
        "bert_vocab_sha256": sha256(vocab) if lengths else None,
        "llama_tokenizer": args.llama_tokenizer if needs_llama else None,
        "tokenizer_revision": args.tokenizer_revision if needs_llama else None,
        "tokenizer_files": tokenizer_files,
    }


def cache_path(data_root, spec):
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    return data_root / key


def inspect_cache(directory, spec):
    """No network or tokenizer imports on the cache hit path."""
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["request"] != spec or not manifest["artifacts"]:
            return None
        for name, expected in manifest["artifacts"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                return None
            path = directory / relative
            if (
                not path.is_file()
                or path.stat().st_size != expected["bytes"]
                or sha256(path) != expected["sha256"]
            ):
                return None
        required = {f"bert_msl{n}/meta.dat" for n in spec["bert_lengths"]}
        if spec["bert_lengths"]:
            required.add("bert_vocab.txt")
        for length in spec["bert_lengths"]:
            folder = f"bert_msl{length}"
            entries = [
                line.split()
                for line in (directory / folder / "meta.dat").read_text().splitlines()
            ]
            samples = manifest["stats"]["bert"][str(length)]["samples"]
            if samples <= 0 or sum(int(count) for _, count in entries) != samples:
                return None
            required.update(f"{folder}/{name}" for name, _ in entries)
        if spec["llama_tokenizer"]:
            required.add("llama_corpus/tokens.h5")
            if manifest["stats"]["llama"]["tokens"] <= 1:
                return None
        if not required.issubset(manifest["artifacts"]):
            return None
        return manifest
    except (OSError, ValueError, KeyError, TypeError):
        return None


def acquire_documents(args):
    if args.raw_text:
        text = args.raw_text.read_text(encoding="utf-8")
        docs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        return docs[: args.max_documents], {
            "kind": "local_text",
            "sha256": sha256(args.raw_text),
        }
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(args.dataset, revision=args.dataset_revision).sha
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split="train",
        streaming=True,
        revision=revision,
    )
    documents = []
    for row in dataset:
        text = row["text"].strip()
        # WikiText paragraphs are documents here; omit empty rows and headings.
        if len(text) < 20 or text.startswith("="):
            continue
        documents.append(text)
        if len(documents) >= args.max_documents:
            break
    return documents, {
        "kind": "huggingface_dataset",
        "dataset": args.dataset,
        "config": args.dataset_config,
        "split": "train",
        "revision": revision,
    }


def load_llama_tokenizer(args):
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi

    local = Path(args.llama_tokenizer).is_dir()
    try:
        revision = (
            None
            if local
            else HfApi()
            .model_info(args.llama_tokenizer, revision=args.tokenizer_revision)
            .sha
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.llama_tokenizer, revision=revision, trust_remote_code=False
        )
    except OSError as exc:
        raise ValueError(
            "Cannot load the Llama tokenizer. For meta-llama/Llama-3.2-1B, grant access "
            "on Hugging Face and log in (or set HF_TOKEN), or pass --llama-tokenizer "
            "/path/to/local/tokenizer. No jobs have been submitted."
        ) from exc
    if tokenizer.eos_token_id is None or len(tokenizer) != 128256:
        raise ValueError(
            "Expected a Llama 3 tokenizer with 128256 tokens and an EOS token"
        )
    return tokenizer, {"source": args.llama_tokenizer, "revision": revision}


def bert_examples(documents, length, seed):
    """Adjacent sentence groups / random-other-document NSP with length capping."""
    rng = random.Random(seed)
    budget = length - 3
    for index, sentences in enumerate(documents):
        position = 0
        while position < len(sentences):
            group, count = [], 0
            while position < len(sentences) and count < budget:
                sentence = sentences[position]
                group.append(sentence)
                count += len(sentence)
                position += 1
            cut = rng.randrange(1, len(group)) if len(group) > 1 else 1
            first = [token for sentence in group[:cut] for token in sentence]
            random_next = len(group) == 1 or rng.random() < 0.5
            if random_next:
                other = rng.randrange(len(documents) - 1)
                other += other >= index
                candidates = documents[other]
                start = rng.randrange(len(candidates))
                second = []
                for sentence in candidates[start:]:
                    second.extend(sentence)
                    if len(second) >= max(1, budget - len(first)):
                        break
                position -= len(group) - cut
            else:
                second = [token for sentence in group[cut:] for token in sentence]
            while len(first) + len(second) > budget:
                target = first if len(first) > len(second) else second
                if rng.random() < 0.5:
                    del target[0]
                else:
                    target.pop()
            tokens = ["[CLS]", *first, "[SEP]", *second, "[SEP]"]
            segments = [0] * (len(first) + 2) + [1] * (len(second) + 1)
            yield {
                "tokens": tokens,
                "segment_ids": segments,
                "is_random_next": int(random_next),
            }


def write_bert(documents, vocab, lengths, destination, seed):
    from transformers import BertTokenizer

    tokenizer = BertTokenizer(vocab_file=str(vocab), do_lower_case=True)
    tokenized = []
    for text in documents:
        sentences = [
            tokenizer.tokenize(part) for part in re.split(r"(?<=[.!?])\s+", text)
        ]
        sentences = [sentence for sentence in sentences if sentence]
        if sentences:
            tokenized.append(sentences)
    if len(tokenized) < 2:
        raise ValueError(
            "BERT NSP preparation requires at least two nonempty documents"
        )
    shutil.copyfile(vocab, destination / "bert_vocab.txt")
    results = {}
    for length in lengths:
        folder = destination / f"bert_msl{length}"
        folder.mkdir()
        handles, writers = [], []
        counts = [0] * 32
        try:
            for index in range(32):
                handle = (folder / f"part-{index:03d}.csv").open(
                    "w", newline="", encoding="utf-8"
                )
                writer = csv.DictWriter(
                    handle, fieldnames=["tokens", "segment_ids", "is_random_next"]
                )
                writer.writeheader()
                handles.append(handle)
                writers.append(writer)
            tokens = 0
            for index, example in enumerate(bert_examples(tokenized, length, seed)):
                shard = index % len(writers)
                writers[shard].writerow(example)
                counts[shard] += 1
                tokens += len(example["tokens"])
        finally:
            for handle in handles:
                handle.close()
        (folder / "meta.dat").write_text(
            "".join(
                f"part-{i:03d}.csv {count}\n" for i, count in enumerate(counts) if count
            )
        )
        results[str(length)] = {
            "samples": sum(counts),
            "input_tokens": tokens,
            "shards": sum(c > 0 for c in counts),
        }
    return results


def write_llama(documents, tokenizer, destination):
    import h5py
    import numpy as np

    folder = destination / "llama_corpus"
    folder.mkdir()
    with h5py.File(folder / "tokens.h5", "w") as handle:
        data = handle.create_dataset(
            "data", shape=(0,), maxshape=(None,), chunks=(65536,), dtype="i4"
        )
        for offset in range(0, len(documents), 64):
            encoded = tokenizer(
                documents[offset : offset + 64], add_special_tokens=False
            )["input_ids"]
            values = []
            for ids in encoded:
                if tokenizer.bos_token_id is not None:
                    values.append(tokenizer.bos_token_id)
                values.extend(ids)
                values.append(tokenizer.eos_token_id)
            array = np.asarray(values, dtype=np.int32)
            if array.size and (array.min() < 0 or array.max() >= 128256):
                raise ValueError("Tokenizer emitted an ID outside the model vocabulary")
            size = len(data)
            data.resize((size + len(array),))
            data[size:] = array
        count = len(data)
    return {
        "tokens": count,
        "format": "corpus",
        "document_boundary": "BOS/text/EOS; cross-document packing",
    }


def prepare(args, profiles, vocab):
    from filelock import FileLock

    spec = request_spec(args, profiles, vocab)
    directory = cache_path(args.data_root, spec)
    args.data_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory) + ".lock"):
        manifest = inspect_cache(directory, spec)
        if manifest is not None:
            print(
                f"[data] verified cache: {directory}; skipping download and preprocessing",
                flush=True,
            )
            return directory, manifest
        print(f"[data] preparing: {directory}", flush=True)
        tokenizer, tokenizer_info = (
            load_llama_tokenizer(args) if spec["llama_tokenizer"] else (None, None)
        )
        documents, source = acquire_documents(args)
        if len(documents) < 2:
            raise ValueError("The corpus must contain at least two nonempty documents")
        stage = Path(
            tempfile.mkdtemp(prefix=directory.name + ".partial-", dir=args.data_root)
        )
        try:
            stats = {}
            if spec["bert_lengths"]:
                stats["bert"] = write_bert(
                    documents, vocab, spec["bert_lengths"], stage, args.seed
                )
            if tokenizer is not None:
                stats["llama"] = write_llama(documents, tokenizer, stage)
                saved = stage / "llama_tokenizer"
                tokenizer.save_pretrained(saved)
            artifacts = {
                str(p.relative_to(stage)): {
                    "bytes": p.stat().st_size,
                    "sha256": sha256(p),
                }
                for p in sorted(stage.rglob("*"))
                if p.is_file()
            }
            manifest = {
                "request": spec,
                "source": source,
                "tokenizer": tokenizer_info,
                "documents": len(documents),
                "stats": stats,
                "artifacts": artifacts,
            }
            (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if directory.exists():
                backup = directory.with_name(
                    directory.name + ".invalid-" + uuid.uuid4().hex[:8]
                )
                directory.rename(backup)
                print(f"[data] preserved invalid cache: {backup}", flush=True)
            stage.rename(directory)
        except BaseException:
            # Only this invocation's incomplete directory is removed.
            shutil.rmtree(stage)
            raise
        return directory, manifest
