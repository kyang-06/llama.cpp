#!/usr/bin/env python3
"""
Convert a pseudo-quantized fp16 Qwen3 model to GGUF BPT1_0 format.

Weight tensors (linear layers):   BPT1_0
Embedding / output (lm_head):     Q6_K  or  F16  (--embd-type)
1-D tensors and norms:             F32

Usage:
    python convert_qwen3_bpt1_0.py ../../models/Qwen3-4B/ \
        --output ../../models/Qwen3-4B-BPT1_0.gguf \
        [--embd-type {f16,q6_k}]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Locate gguf-py relative to this script
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "gguf-py"))

import gguf
from gguf import GGMLQuantizationType as GT
from gguf.quants import quantize as _quantize
from gguf.constants import GGML_QUANT_SIZES, QK_K

# ---------------------------------------------------------------------------
# Tensor name mapping  (HuggingFace → GGUF/llama.cpp)
# ---------------------------------------------------------------------------

_GLOBAL = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight":         "output_norm.weight",
    "lm_head.weight":            "output.weight",
}

_BLOCK_RE = re.compile(r"model\.layers\.(\d+)\.(.*)")

_BLOCK_MAP = {
    "input_layernorm.weight":            "attn_norm.weight",
    "post_attention_layernorm.weight":   "ffn_norm.weight",
    "self_attn.q_proj.weight":          "attn_q.weight",
    "self_attn.k_proj.weight":          "attn_k.weight",
    "self_attn.v_proj.weight":          "attn_v.weight",
    "self_attn.o_proj.weight":          "attn_output.weight",
    "self_attn.q_norm.weight":          "attn_q_norm.weight",   # Qwen3 QK-RMSNorm
    "self_attn.k_norm.weight":          "attn_k_norm.weight",
    "mlp.gate_proj.weight":             "ffn_gate.weight",
    "mlp.up_proj.weight":               "ffn_up.weight",
    "mlp.down_proj.weight":             "ffn_down.weight",
}

# Biases (Qwen3 uses no biases in linear layers; include for completeness)
_BLOCK_MAP.update({
    "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.o_proj.bias": "attn_output.bias",
})

# These tensors are always kept in F32 regardless of dimensionality
_ALWAYS_F32 = {
    "attn_norm.weight", "ffn_norm.weight",
    "attn_q_norm.weight", "attn_k_norm.weight",
    "output_norm.weight",
}

# Tensors that map to embedding / output (subject to --embd-type)
_EMBD_NAMES = {"token_embd.weight", "output.weight"}


def hf_to_gguf_name(hf_name: str) -> str | None:
    if hf_name in _GLOBAL:
        return _GLOBAL[hf_name]
    m = _BLOCK_RE.match(hf_name)
    if m:
        blk, rest = m.group(1), m.group(2)
        suffix = _BLOCK_MAP.get(rest)
        if suffix:
            return f"blk.{blk}.{suffix}"
    return None


# ---------------------------------------------------------------------------
# Quantization type selection
# ---------------------------------------------------------------------------

def select_qtype(gguf_name: str, shape: tuple, embd_qtype: GT) -> GT:
    # Extract the base suffix (after "blk.N." or the full name)
    suffix = gguf_name.split(".")[-2] + "." + gguf_name.split(".")[-1]
    base = gguf_name.rsplit(".", 1)[-1] + ""   # not used directly

    if gguf_name in _EMBD_NAMES:
        return embd_qtype

    # 1-D tensors and known norm tensors → F32
    if len(shape) <= 1:
        return GT.F32

    for f32_suffix in _ALWAYS_F32:
        if gguf_name.endswith(f32_suffix):
            return GT.F32

    # Biases → F32
    if gguf_name.endswith(".bias"):
        return GT.F32

    # Inner dimension must be multiple of QK_K=256 for block quantization
    if shape[-1] % QK_K != 0:
        print(f"  WARN: {gguf_name} inner dim {shape[-1]} not ×{QK_K} → F16")
        return GT.F16

    return GT.BPT1_0


# ---------------------------------------------------------------------------
# Vocabulary (Qwen3 uses tiktoken / BPE via transformers)
# ---------------------------------------------------------------------------

def write_qwen_vocab(writer: gguf.GGUFWriter, model_dir: Path, config: dict) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  WARN: transformers not installed, skipping tokenizer")
        return

    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    vocab_size = config.get("vocab_size", tokenizer.vocab_size)

    tokens:    list[bytes] = []
    scores:    list[float] = []
    toktypes:  list[int]   = []

    def token_bytes_to_string(b: bytes) -> str:
        return b.decode("utf-8", errors="replace")

    mergeable_ranks: dict[bytes, int] = getattr(tokenizer, "mergeable_ranks", {})
    special_tokens:  dict[str, int]   = getattr(tokenizer, "special_tokens", {})

    # Build rank → token mapping
    reverse: dict[int, str] = {}
    for tok_bytes, rank in mergeable_ranks.items():
        reverse[rank] = token_bytes_to_string(tok_bytes)
    for tok_str, rank in special_tokens.items():
        reverse[rank] = tok_str

    # Build BPE merges
    merges: list[str] = []
    if mergeable_ranks:
        def bpe(ranks: dict, token: bytes, max_rank: int) -> list[bytes]:
            parts = [bytes([b]) for b in token]
            while True:
                min_idx, min_rank = None, None
                for i, pair in enumerate(zip(parts, parts[1:])):
                    rank = ranks.get(pair[0] + pair[1])
                    if rank is not None and rank < max_rank and (min_rank is None or rank < min_rank):
                        min_idx, min_rank = i, rank
                if min_idx is None:
                    break
                parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
            return parts

        for tok_bytes, rank in mergeable_ranks.items():
            if len(tok_bytes) == 1:
                continue
            merged = bpe(mergeable_ranks, tok_bytes, max_rank=rank)
            if len(merged) == 2:
                merges.append(" ".join(token_bytes_to_string(p) for p in merged))

    for i in range(vocab_size):
        if i not in reverse:
            tokens.append(f"[PAD{i}]".encode())
            scores.append(-10000.0)
            toktypes.append(gguf.TokenType.UNUSED)
        elif reverse[i] in (special_tokens or {}):
            tokens.append(reverse[i].encode())
            scores.append(-10000.0)
            toktypes.append(gguf.TokenType.CONTROL)
        else:
            tokens.append(reverse[i].encode())
            scores.append(-float(i))
            toktypes.append(gguf.TokenType.NORMAL)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)
    if merges:
        writer.add_token_merges(merges)
    writer.add_bos_token_id(tokenizer.bos_token_id or 151643)
    writer.add_eos_token_id(tokenizer.eos_token_id or 151645)
    writer.add_unk_token_id(tokenizer.unk_token_id or 0)
    writer.add_pad_token_id(tokenizer.pad_token_id or 151643)

    # Special vocab (chat tokens, etc.)
    try:
        special_vocab = gguf.SpecialVocab(str(model_dir), n_vocab=vocab_size)
        special_vocab.add_to_gguf(writer)
    except Exception:
        pass

    print(f"  vocab size = {vocab_size}, merges = {len(merges)}")


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(args: argparse.Namespace) -> None:
    model_dir  = args.model_dir.resolve()
    output     = args.output
    embd_qtype = GT.Q6_K if args.embd_type == "q6_k" else GT.F16

    # ── Config ──────────────────────────────────────────────────────────────
    with open(model_dir / "config.json") as f:
        config: dict = json.load(f)

    n_layer   = config["num_hidden_layers"]
    n_head    = config["num_attention_heads"]
    n_kv_head = config.get("num_key_value_heads", n_head)
    n_embd    = config["hidden_size"]
    n_ff      = config["intermediate_size"]
    n_ctx     = config.get("max_position_embeddings", 32768)
    vocab_sz  = config.get("vocab_size", 0)
    head_dim  = config.get("head_dim", n_embd // n_head)
    rope_base = config.get("rope_theta", 10000.0)
    rms_eps   = config.get("rms_norm_eps", 1e-6)

    print(f"Model: {config.get('model_type', '?')}  layers={n_layer}  "
          f"embd={n_embd}  heads={n_head}/{n_kv_head}  ff={n_ff}")
    print(f"Embedding quantization: {embd_qtype.name}")

    # ── Discover shard files ────────────────────────────────────────────────
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index_data = json.load(f)
        shard_files = sorted(set(index_data["weight_map"].values()))
    elif (model_dir / "model.safetensors").exists():
        shard_files = ["model.safetensors"]
    else:
        sys.exit("ERROR: no safetensors files found in model directory")

    # ── GGUF writer ──────────────────────────────────────────────────────────
    writer = gguf.GGUFWriter(str(output), arch="qwen3")

    writer.add_context_length(n_ctx)
    writer.add_embedding_length(n_embd)
    writer.add_feed_forward_length(n_ff)
    writer.add_block_count(n_layer)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_kv_head)
    writer.add_rope_freq_base(float(rope_base))
    writer.add_rope_dimension_count(head_dim)
    writer.add_layer_norm_rms_eps(float(rms_eps))
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_BPT1_0)
    if vocab_sz:
        writer.add_vocab_size(vocab_sz)

    # Qwen3 uses sliding-window attention
    if "sliding_window" in config and config["sliding_window"]:
        writer.add_sliding_window(config["sliding_window"])

    # ── Tokenizer ────────────────────────────────────────────────────────────
    write_qwen_vocab(writer, model_dir, config)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()

    # ── Load and quantize tensors ─────────────────────────────────────────────
    try:
        from safetensors.numpy import load_file as st_load_file
        def load_shard(path: Path) -> dict[str, np.ndarray]:
            return st_load_file(str(path))
    except ImportError:
        try:
            from safetensors import safe_open
            def load_shard(path: Path) -> dict[str, np.ndarray]:  # type: ignore[misc]
                out = {}
                with safe_open(str(path), framework="numpy") as f:
                    for k in f.keys():
                        out[k] = f.get_tensor(k)
                return out
        except ImportError:
            sys.exit("ERROR: safetensors package required (pip install safetensors)")

    seen: set[str] = set()
    total_orig  = 0
    total_quant = 0

    for shard_file in shard_files:
        shard_path = model_dir / shard_file
        print(f"\nShard: {shard_file}")
        tensors = load_shard(shard_path)

        for hf_name, raw in sorted(tensors.items()):
            gguf_name = hf_to_gguf_name(hf_name)
            if gguf_name is None:
                print(f"  SKIP (unmapped): {hf_name}")
                continue
            if gguf_name in seen:
                continue
            seen.add(gguf_name)

            data = raw.astype(np.float32)
            qtype = select_qtype(gguf_name, data.shape, embd_qtype)

            orig_bytes  = data.nbytes
            total_orig += orig_bytes

            if qtype == GT.F32:
                writer.add_tensor(gguf_name, data, raw_dtype=GT.F32)
                quant_bytes = data.nbytes
            elif qtype == GT.F16:
                d16 = data.astype(np.float16)
                writer.add_tensor(gguf_name, d16, raw_dtype=GT.F16)
                quant_bytes = d16.nbytes
            else:
                try:
                    qdata = _quantize(data, qtype)
                    writer.add_tensor(gguf_name, qdata, raw_dtype=qtype)
                    quant_bytes = qdata.nbytes
                except Exception as e:
                    print(f"  WARN: {gguf_name} quantize failed ({e}), using F16")
                    d16 = data.astype(np.float16)
                    writer.add_tensor(gguf_name, d16, raw_dtype=GT.F16)
                    quant_bytes = d16.nbytes
                    qtype = GT.F16

            ratio = orig_bytes / quant_bytes if quant_bytes else 0
            total_quant += quant_bytes
            shape_str = "×".join(str(d) for d in data.shape)
            print(f"  {gguf_name:<45s} {shape_str:>20s}  {qtype.name:<8s}  "
                  f"({ratio:.2f}×)")

    writer.write_tensors_to_file(progress=True)
    writer.close()

    orig_gb  = total_orig  / 1e9
    quant_gb = total_quant / 1e9
    ratio    = total_orig / total_quant if total_quant else 0
    print(f"\nDone → {output}")
    print(f"  Original:   {orig_gb:.2f} GB")
    print(f"  Quantized:  {quant_gb:.2f} GB  ({ratio:.2f}× compression)")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert fp16 Qwen3 model to GGUF BPT1_0 format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_dir", type=Path,
        help="Path to the HuggingFace Qwen3 model directory",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output GGUF file path (default: <model_dir_name>-BPT1_0.gguf)",
    )
    parser.add_argument(
        "--embd-type", choices=["f16", "q6_k"], default="f16",
        help="Quantization for token_embd / lm_head tensors",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(args.model_dir.name + "-BPT1_0.gguf")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    convert(args)


if __name__ == "__main__":
    main()
