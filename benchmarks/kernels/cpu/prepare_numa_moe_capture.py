# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare exact Qwen3.6 FP8 MoE weights and fixed Sonnet prompts for replay."""

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import regex as re
import torch
import transformers
from safetensors import safe_open
from transformers import AutoTokenizer

DEFAULT_CHECKPOINT = Path(
    "/home/tianmuli/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/"
    "95a723d08a9490559dae23d0cff1d9466213d989"
)
DEFAULT_SONNET = Path(
    "/home/tianmuli/vllm_test/vllm_baseline_serving/benchmarks/sonnet.txt"
)
NUM_LAYERS = 40
NUM_EXPERTS = 256
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512
INPUT_TOKENS = 1024
NUM_PROMPTS = 256
SONNET_PREFIX_TOKENS = 200
PROMPT_SEED = 0
BLOCK_SIZE = 128

TOKENIZER_ASSETS = (
    "chat_template.jinja",
    "config.json",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_expert_keys(layer: int, expert: int) -> dict[str, str]:
    root = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
    return {
        "gate_weight": f"{root}gate_proj.weight",
        "up_weight": f"{root}up_proj.weight",
        "down_weight": f"{root}down_proj.weight",
        "gate_scale": f"{root}gate_proj.weight_scale_inv",
        "up_scale": f"{root}up_proj.weight_scale_inv",
        "down_scale": f"{root}down_proj.weight_scale_inv",
    }


def _validate_checkpoint(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{40}", checkpoint.name):
        raise ValueError(
            "Checkpoint directory name must be an immutable 40-character "
            f"revision, got {checkpoint.name!r}"
        )
    config_path = checkpoint / "config.json"
    index_path = checkpoint / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", {})
    expected_config = {
        "hidden_size": HIDDEN_SIZE,
        "moe_intermediate_size": INTERMEDIATE_SIZE,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": 8,
        "num_hidden_layers": NUM_LAYERS,
    }
    actual_config = {key: text_config.get(key) for key in expected_config}
    if actual_config != expected_config:
        raise ValueError(
            f"Checkpoint text geometry differs from the experiment: {actual_config}"
        )
    quant_config = config.get("quantization_config", {})
    if quant_config.get("quant_method") != "fp8":
        raise ValueError(
            f"Expected FP8 checkpoint, got {quant_config.get('quant_method')!r}"
        )
    if quant_config.get("fmt") != "e4m3":
        raise ValueError(f"Expected E4M3 weights, got {quant_config.get('fmt')!r}")
    if quant_config.get("weight_block_size") != [BLOCK_SIZE, BLOCK_SIZE]:
        raise ValueError(
            "Expected 128x128 FP8 weight blocks, got "
            f"{quant_config.get('weight_block_size')!r}"
        )

    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Checkpoint index has no weight_map")
    shards = sorted(set(weight_map.values()))
    missing_shards = [shard for shard in shards if not (checkpoint / shard).is_file()]
    if missing_shards:
        raise FileNotFoundError(
            f"Checkpoint index references missing shards: {missing_shards}"
        )

    layer_shards: dict[int, str] = {}
    for layer in range(NUM_LAYERS):
        keys = [
            weight_map.get(_expected_expert_keys(layer, expert)["gate_weight"])
            for expert in range(NUM_EXPERTS)
        ]
        if any(key is None for key in keys):
            missing = [
                f"layer {layer} expert {expert} gate weight"
                for expert, shard in enumerate(keys)
                if shard is None
            ]
            raise ValueError(f"Checkpoint index is missing MoE weights: {missing[:8]}")
        mapped_shards = set(keys)
        if len(mapped_shards) != 1:
            raise ValueError(
                f"Layer {layer} experts span unexpected checkpoint shards: "
                f"{sorted(mapped_shards)}"
            )
        layer_shards[layer] = mapped_shards.pop()

    mtp_shards = sorted(shard for shard in shards if "mtp" in Path(shard).name.lower())
    base_shards = sorted(set(shards) - set(mtp_shards))
    if not any(Path(shard).name == "outside.safetensors" for shard in base_shards):
        raise ValueError(
            "Checkpoint index does not include outside.safetensors; refusing to "
            "claim complete base-checkpoint provenance"
        )

    shapes_and_dtypes = {
        "gate_weight": ([INTERMEDIATE_SIZE, HIDDEN_SIZE], "F8_E4M3"),
        "up_weight": ([INTERMEDIATE_SIZE, HIDDEN_SIZE], "F8_E4M3"),
        "down_weight": ([HIDDEN_SIZE, INTERMEDIATE_SIZE], "F8_E4M3"),
        "gate_scale": (
            [INTERMEDIATE_SIZE // BLOCK_SIZE, HIDDEN_SIZE // BLOCK_SIZE],
            "BF16",
        ),
        "up_scale": (
            [INTERMEDIATE_SIZE // BLOCK_SIZE, HIDDEN_SIZE // BLOCK_SIZE],
            "BF16",
        ),
        "down_scale": (
            [HIDDEN_SIZE // BLOCK_SIZE, INTERMEDIATE_SIZE // BLOCK_SIZE],
            "BF16",
        ),
    }
    for layer, shard in layer_shards.items():
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as source:
            source_keys = set(source.keys())
            for expert in range(NUM_EXPERTS):
                for role, key in _expected_expert_keys(layer, expert).items():
                    if key not in source_keys:
                        raise ValueError(f"Checkpoint shard {shard} is missing {key}")
                    if weight_map.get(key) != shard:
                        raise ValueError(
                            f"Checkpoint index maps {key} to {weight_map.get(key)!r}, "
                            f"but {shard} contains the layer"
                        )
                    expected_shape, expected_dtype = shapes_and_dtypes[role]
                    tensor_slice = source.get_slice(key)
                    actual_shape = tensor_slice.get_shape()
                    actual_dtype = tensor_slice.get_dtype()
                    if actual_shape != expected_shape or actual_dtype != expected_dtype:
                        raise ValueError(
                            f"{key}: expected {expected_dtype} {expected_shape}, got "
                            f"{actual_dtype} {actual_shape}"
                        )

    for asset in TOKENIZER_ASSETS:
        if not (checkpoint / asset).is_file():
            raise FileNotFoundError(f"Checkpoint is missing required asset {asset}")

    source_paths = sorted(
        set(base_shards) | set(TOKENIZER_ASSETS),
    )
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_revision": checkpoint.name,
        "config": config,
        "weight_map": weight_map,
        "layer_shards": layer_shards,
        "base_parameter_shards": base_shards,
        "excluded_mtp_shards": mtp_shards,
        "source_paths": source_paths,
        "indexed_tensor_count": len(weight_map),
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "block_size": [BLOCK_SIZE, BLOCK_SIZE],
        "source_weight_layout": {
            "gate_proj.weight": [INTERMEDIATE_SIZE, HIDDEN_SIZE],
            "up_proj.weight": [INTERMEDIATE_SIZE, HIDDEN_SIZE],
            "down_proj.weight": [HIDDEN_SIZE, INTERMEDIATE_SIZE],
            "weight_scale_inv_dtype": "BF16",
        },
        "exported_weight_layout": {
            "w1": [NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE],
            "w2": [NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE],
            "s1": [
                NUM_EXPERTS,
                2 * INTERMEDIATE_SIZE // BLOCK_SIZE,
                HIDDEN_SIZE // BLOCK_SIZE,
            ],
            "s2": [
                NUM_EXPERTS,
                HIDDEN_SIZE // BLOCK_SIZE,
                INTERMEDIATE_SIZE // BLOCK_SIZE,
            ],
        },
    }


def _chat_template_ids(tokenizer: Any, prompt_text: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True,
        add_generation_prompt=True,
    )
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _fit_prompt_to_token_count(
    tokenizer: Any,
    candidate: str,
    target_tokens: int = INPUT_TOKENS,
) -> tuple[str, list[int]]:
    empty_template = _chat_template_ids(tokenizer, "")
    candidate_template = _chat_template_ids(tokenizer, candidate)
    end_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    def find_user_end(ids: list[int]) -> int:
        matches = [
            index for index, token_id in enumerate(ids) if token_id == end_token_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one user-message closure in the rendered template"
            )
        return matches[0]

    empty_user_end = find_user_end(empty_template)
    candidate_user_end = find_user_end(candidate_template)
    prefix = empty_template[:empty_user_end]
    suffix = empty_template[empty_user_end:]
    candidate_prefix = candidate_template[: len(prefix)]
    if candidate_prefix != prefix:
        raise ValueError("Prompt content changed the chat-template prefix")
    body_ids = candidate_template[len(prefix) : candidate_user_end]
    body_token_count = target_tokens - len(prefix) - len(suffix)
    if body_token_count <= 0 or body_token_count > len(body_ids):
        raise ValueError(
            f"Cannot fit a {target_tokens}-token prompt while preserving its "
            "chat-template prefix and suffix"
        )

    selected_body_ids = body_ids[:body_token_count]
    input_ids = [*prefix, *selected_body_ids, *suffix]
    if len(input_ids) != target_tokens:
        raise AssertionError("Prompt token-count construction lost its invariant")
    prompt_text = tokenizer.decode(
        selected_body_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return prompt_text, input_ids


def _make_prompts(
    sonnet_path: Path,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sonnet_path = sonnet_path.resolve(strict=True)
    lines = sonnet_path.read_text().splitlines(keepends=True)
    if not lines:
        raise ValueError(f"No poem lines were found in {sonnet_path}")
    rng = random.Random(PROMPT_SEED)
    instruction = "Pick as many lines as you can from these poem lines:\n"
    average_line_tokens = sum(len(tokenizer(line).input_ids) for line in lines) / len(
        lines
    )
    base_tokens = len(_chat_template_ids(tokenizer, instruction))
    input_lines = max(round((INPUT_TOKENS - base_tokens) / average_line_tokens), 1)
    prefix_lines = max(
        round((SONNET_PREFIX_TOKENS - base_tokens) / average_line_tokens), 0
    )
    prompts = []
    for prompt_index in range(NUM_PROMPTS):
        selected_lines = lines[:prefix_lines] + rng.choices(
            lines, k=input_lines - prefix_lines
        )
        candidate = instruction + "".join(selected_lines)
        while len(_chat_template_ids(tokenizer, candidate)) < INPUT_TOKENS:
            candidate += "".join(rng.choices(lines, k=4))
        prompt_text, prompt_token_ids = _fit_prompt_to_token_count(
            tokenizer,
            candidate,
        )
        prompts.append(
            {
                "index": prompt_index,
                "prompt_text": prompt_text,
                "messages": [{"role": "user", "content": prompt_text}],
                "prompt_token_ids": prompt_token_ids,
            }
        )

    prompt_hash_input = [
        {
            "prompt_text": prompt["prompt_text"],
            "prompt_token_ids": prompt["prompt_token_ids"],
        }
        for prompt in prompts
    ]
    metadata = {
        "source_sonnet_path": str(sonnet_path),
        "source_sonnet_sha256": _sha256_file(sonnet_path),
        "num_prompts": NUM_PROMPTS,
        "prompts_per_concurrency": NUM_PROMPTS,
        "concurrency": [8, 32, 128],
        "input_tokens": INPUT_TOKENS,
        "generated_tokens": 128,
        "mtp_enabled": False,
        "seed": PROMPT_SEED,
        "sampling": "canonical Sonnet prefix and random.choices over poem lines",
        "estimated_lines_per_prompt": input_lines,
        "shared_prefix_lines": prefix_lines,
        "shared_prefix_target_tokens": SONNET_PREFIX_TOKENS,
        "exact_length_extension": "append four sampled lines until at least 1024 IDs",
        "user_prefix": instruction,
        "tokenizer_path": "checkpoint-local tokenizer, local_files_only=True",
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_eos_token": tokenizer.eos_token,
        "transformers_version": transformers.__version__,
        "chat_template": "one user message plus add_generation_prompt=True",
        "chat_template_includes": [
            "user message closure",
            "assistant generation prefix",
            "thinking prefix",
        ],
        "token_ids_are_authoritative": True,
        "prompt_text_must_not_be_retokenized": True,
        "token_truncation": (
            "truncate the rendered user-body token IDs and preserve the "
            "chat-template prefix and suffix"
        ),
        "prompt_set_sha256": _canonical_sha256(prompt_hash_input),
        "token_lengths": sorted(
            {len(prompt["prompt_token_ids"]) for prompt in prompts}
        ),
    }
    if metadata["token_lengths"] != [INPUT_TOKENS]:
        raise ValueError(
            f"Prompt token length validation failed: {metadata['token_lengths']}"
        )
    return prompts, metadata


def _checkpoint_hashes(
    checkpoint_info: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    checkpoint = Path(checkpoint_info["checkpoint_path"])
    files = []
    for relative_path in checkpoint_info["source_paths"]:
        path = checkpoint / relative_path
        files.append(
            {
                "path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    digest_input = {
        "format": "qwen36-base-checkpoint-sha256-v1",
        "scope": (
            "all indexed non-MTP parameter shards, including outside.safetensors, "
            "plus model configuration and tokenizer assets; MTP is excluded because "
            "the replay disables MTP"
        ),
        "files": files,
    }
    return files, _canonical_sha256(digest_input)


def _mapped_tensor(
    path: Path,
    dtype: torch.dtype,
    numel: int,
) -> torch.Tensor:
    path.touch()
    with path.open("r+b") as file:
        file.truncate(numel * torch.empty((), dtype=dtype).element_size())
    return torch.from_file(str(path), shared=True, size=numel, dtype=dtype)


def _export_weights(
    checkpoint_info: dict[str, Any],
    output_path: Path,
) -> dict[str, int]:
    checkpoint = Path(checkpoint_info["checkpoint_path"])
    w1_shape = (NUM_LAYERS, NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
    w2_shape = (NUM_LAYERS, NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    s1_shape = (
        NUM_LAYERS,
        NUM_EXPERTS,
        2 * INTERMEDIATE_SIZE // BLOCK_SIZE,
        HIDDEN_SIZE // BLOCK_SIZE,
    )
    s2_shape = (
        NUM_LAYERS,
        NUM_EXPERTS,
        HIDDEN_SIZE // BLOCK_SIZE,
        INTERMEDIATE_SIZE // BLOCK_SIZE,
    )
    element_counts = {
        "w1": torch.Size(w1_shape).numel(),
        "w2": torch.Size(w2_shape).numel(),
        "s1": torch.Size(s1_shape).numel(),
        "s2": torch.Size(s2_shape).numel(),
    }
    total_bytes = element_counts["w1"] + element_counts["w2"]
    total_bytes += 4 * (element_counts["s1"] + element_counts["s2"])
    required_free = 2 * total_bytes + 1024**3
    free_bytes = shutil.disk_usage(output_path.parent).free
    if free_bytes < required_free:
        raise OSError(
            "Insufficient free space for the temporary memory-mapped tensors and "
            f"ZIP archive: need {required_free} bytes, have {free_bytes}"
        )

    with tempfile.TemporaryDirectory(
        prefix=".numa-moe-weights-",
        dir=output_path.parent,
    ) as temp_dir:
        temp_dir_path = Path(temp_dir)
        weights = {
            "w1": _mapped_tensor(
                temp_dir_path / "w1.raw",
                torch.float8_e4m3fn,
                element_counts["w1"],
            ).view(w1_shape),
            "w2": _mapped_tensor(
                temp_dir_path / "w2.raw",
                torch.float8_e4m3fn,
                element_counts["w2"],
            ).view(w2_shape),
            "s1": _mapped_tensor(
                temp_dir_path / "s1.raw",
                torch.float32,
                element_counts["s1"],
            ).view(s1_shape),
            "s2": _mapped_tensor(
                temp_dir_path / "s2.raw",
                torch.float32,
                element_counts["s2"],
            ).view(s2_shape),
        }
        for layer, shard in sorted(checkpoint_info["layer_shards"].items()):
            with safe_open(checkpoint / shard, framework="pt", device="cpu") as source:
                for expert in range(NUM_EXPERTS):
                    keys = _expected_expert_keys(layer, expert)
                    gate = source.get_tensor(keys["gate_weight"])
                    up = source.get_tensor(keys["up_weight"])
                    down = source.get_tensor(keys["down_weight"])
                    gate_scale = source.get_tensor(keys["gate_scale"])
                    up_scale = source.get_tensor(keys["up_scale"])
                    down_scale = source.get_tensor(keys["down_scale"])
                    weights["w1"][layer, expert, :INTERMEDIATE_SIZE].copy_(gate)
                    weights["w1"][layer, expert, INTERMEDIATE_SIZE:].copy_(up)
                    weights["w2"][layer, expert].copy_(down)
                    weights["s1"][
                        layer, expert, : INTERMEDIATE_SIZE // BLOCK_SIZE
                    ].copy_(gate_scale.float())
                    weights["s1"][
                        layer, expert, INTERMEDIATE_SIZE // BLOCK_SIZE :
                    ].copy_(up_scale.float())
                    weights["s2"][layer, expert].copy_(down_scale.float())

        layers = {
            layer: {
                "w1": weights["w1"][layer],
                "w2": weights["w2"][layer],
                "s1": weights["s1"][layer],
                "s2": weights["s2"][layer],
            }
            for layer in range(NUM_LAYERS)
        }
        temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(
            {
                "layers": layers,
                "metadata": {
                    "source_checkpoint_revision": checkpoint_info[
                        "checkpoint_revision"
                    ],
                    "source_layout": checkpoint_info["source_weight_layout"],
                    "export_layout": checkpoint_info["exported_weight_layout"],
                    "scale_conversion": "checkpoint BF16 weight_scale_inv to FP32",
                    "gate_up_order": ["gate_proj", "up_proj"],
                },
            },
            temporary_output,
        )
        temporary_output.replace(output_path)
    return {
        "layers": NUM_LAYERS,
        "experts_per_layer": NUM_EXPERTS,
        "raw_tensor_bytes": total_bytes,
        "archive_bytes": output_path.stat().st_size,
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sonnet", type=Path, default=DEFAULT_SONNET)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Create weights.pt, prompts.pt, and checkpoint_manifest.json here",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Validate source tensor headers and generate prompts without writing files"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint_info = _validate_checkpoint(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_info["checkpoint_path"],
        local_files_only=True,
        trust_remote_code=True,
    )
    prompts, prompt_metadata = _make_prompts(args.sonnet, tokenizer)

    if args.check_only:
        print(
            json.dumps(
                {
                    "checkpoint_revision": checkpoint_info["checkpoint_revision"],
                    "checkpoint_header_validation": "passed",
                    "indexed_tensor_count": checkpoint_info["indexed_tensor_count"],
                    "base_parameter_shards": checkpoint_info["base_parameter_shards"],
                    "excluded_mtp_shards": checkpoint_info["excluded_mtp_shards"],
                    "exported_weight_layout": checkpoint_info["exported_weight_layout"],
                    "prompt_metadata": prompt_metadata,
                    "first_prompt_token_count": len(prompts[0]["prompt_token_ids"]),
                    "last_prompt_token_count": len(prompts[-1]["prompt_token_ids"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --check-only is used")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    checkpoint_files, checkpoint_sha256 = _checkpoint_hashes(checkpoint_info)
    weight_summary = _export_weights(
        checkpoint_info,
        output_dir / "weights.pt",
    )
    prompt_payload = {
        "metadata": {
            **prompt_metadata,
            "checkpoint_revision": checkpoint_info["checkpoint_revision"],
            "checkpoint_sha256": checkpoint_sha256,
        },
        "prompts": prompts,
    }
    torch.save(prompt_payload, output_dir / "prompts.pt")
    manifest = {
        "checkpoint_revision": checkpoint_info["checkpoint_revision"],
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_hash_format": "qwen36-base-checkpoint-sha256-v1",
        "checkpoint_hash_scope": (
            "all indexed non-MTP parameter shards including outside.safetensors, "
            "plus model configuration and tokenizer assets; MTP excluded because "
            "the replay disables MTP"
        ),
        "checkpoint_files": checkpoint_files,
        "indexed_tensor_count": checkpoint_info["indexed_tensor_count"],
        "base_parameter_shards": checkpoint_info["base_parameter_shards"],
        "excluded_mtp_shards": checkpoint_info["excluded_mtp_shards"],
        "moe_layer_shards": {
            str(layer): shard
            for layer, shard in sorted(checkpoint_info["layer_shards"].items())
        },
        "source_weight_layout": checkpoint_info["source_weight_layout"],
        "exported_weight_layout": checkpoint_info["exported_weight_layout"],
        "weight_archive": {
            "filename": "weights.pt",
            "format": "PyTorch ZIP serialization",
            "sha256": _sha256_file(output_dir / "weights.pt"),
            "file_bytes": (output_dir / "weights.pt").stat().st_size,
            **weight_summary,
        },
        "prompt_set": {
            "filename": "prompts.pt",
            "sha256": _sha256_file(output_dir / "prompts.pt"),
            "file_bytes": (output_dir / "prompts.pt").stat().st_size,
            **prompt_metadata,
        },
    }
    _write_json(output_dir / "checkpoint_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
