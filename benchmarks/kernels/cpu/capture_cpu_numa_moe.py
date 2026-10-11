# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture ordered CPU MoE replay inputs from one TP1 reference runtime."""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import regex as re
import torch

from benchmarks.kernels.cpu import benchmark_cpu_numa_moe as replay
from vllm import EngineArgs, SamplingParams
from vllm.model_executor.layers.fused_moe.experts.cpu_moe import CPUExpertsFp8
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.worker.cpu.model_runner import CPUModelRunner as CPUModelRunnerV2
from vllm.v1.worker.cpu_model_runner import CPUModelRunner


class Capture:
    def __init__(
        self,
        output: Path,
        expected_layers: int,
        reference_cpus: tuple[int, ...] = tuple(range(32, 128)),
    ):
        self.output = output
        self.reference_cpus = reference_cpus
        self.expected_layers = list(range(expected_layers))
        self.step = 0
        self.concurrency = None
        self.current = None
        self.records = []
        self.files = []
        self.windows = {}
        self.layer_by_weight = {}
        self.runner = None
        self.prompt_lengths = {}
        self.expert_calls = 0

    def event(self, **fields):
        fields["time_unix"] = time.time()
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(fields) + "\n")
        print(json.dumps(fields), flush=True)

    def register(self, runner):
        self.runner = runner
        for module in runner.model.modules():
            if isinstance(module, RoutedExperts):
                match = re.search(r"\.layers\.(\d+)\.", module.layer_name)
                if match:
                    self.layer_by_weight[module.w13_weight.data_ptr()] = int(
                        match.group(1)
                    )
        if sorted(self.layer_by_weight.values()) != self.expected_layers:
            raise RuntimeError(
                f"Routed layers differ: {sorted(self.layer_by_weight.values())}"
            )
        if runner.speculative_config is not None:
            raise RuntimeError("MTP must be disabled")
        affinity = replay._thread_affinity(self.reference_cpus)
        bound_cpus = {mask[0] for mask in affinity.values() if len(mask) == 1}
        if bound_cpus != set(self.reference_cpus) or torch.get_num_threads() != len(
            self.reference_cpus
        ):
            raise RuntimeError(
                "Reference OpenMP threads must match the selected physical CPUs; "
                f"singleton thread masks cover {sorted(bound_cpus)}"
            )
        self.event(
            event="model_loaded",
            layers=self.expected_layers,
            extension=replay._extension_identity(),
            torch_threads=torch.get_num_threads(),
            thread_affinity_by_tid=affinity,
            omp_affinity=os.environ.get("KMP_AFFINITY"),
            speculative_config=None,
            runner_class=f"{type(runner).__module__}.{type(runner).__qualname__}",
        )

    def begin(self, runner, scheduler_output):
        self.current = None
        if not scheduler_output.total_num_scheduled_tokens:
            return
        self.step += 1
        new = {
            request.req_id: request for request in scheduler_output.scheduled_new_reqs
        }
        cached = scheduler_output.scheduled_cached_reqs
        computed = dict(zip(cached.req_ids, cached.num_computed_tokens))
        for request_id in scheduler_output.finished_req_ids:
            self.prompt_lengths.pop(request_id, None)
        for request_id, request in new.items():
            self.prompt_lengths[request_id] = len(request.prompt_token_ids)
        prefill_tokens = decode_tokens = 0
        for request_id, count in scheduler_output.num_scheduled_tokens.items():
            if request_id in new:
                request = new[request_id]
                before = request.num_computed_tokens
                prompt_tokens = len(request.prompt_token_ids)
            else:
                before = computed[request_id]
                prompt_tokens = self.prompt_lengths[request_id]
            if before < prompt_tokens:
                prefill_tokens += count
            else:
                decode_tokens += count
        phase = (
            "mixed"
            if prefill_tokens and decode_tokens
            else ("prefill" if prefill_tokens else "decode")
        )
        if self.concurrency is None:
            return
        windows = self.windows[self.concurrency]
        window = None
        if len(windows["prefill"]) < 8 and prefill_tokens:
            window = "prefill"
        elif len(windows["decode"]) < 8 and not prefill_tokens:
            window = "decode"
        if window is None:
            return
        previous = windows[window]
        if previous and self.step != previous[-1] + 1:
            discarded = set(previous)
            self.files = [
                path
                for path in self.files
                if not (
                    path.parent.name == str(self.concurrency)
                    and int(path.name.split("_", 1)[0]) in discarded
                )
            ]
            self.event(
                event="capture_window_restarted",
                concurrency=self.concurrency,
                capture_window=window,
                discarded_steps=previous.copy(),
                source_step=self.step,
            )
            previous.clear()
        self.current = {
            "layer": None,
            "phase": phase,
            "concurrency": self.concurrency,
            "step": self.step,
            "capture_window": window,
            "scheduled_prefill_tokens": prefill_tokens,
            "scheduled_decode_tokens": decode_tokens,
            "scheduled_requests": len(scheduler_output.num_scheduled_tokens),
        }
        self.records = []

    def expert(self, hidden, w1, ids, weights):
        if self.current is None:
            return
        layer = self.layer_by_weight[w1.data_ptr()]
        record = {
            **self.current,
            "layer": layer,
            "hidden": hidden.detach().clone().contiguous(),
            "ids": ids.detach().to(torch.int32).clone().contiguous(),
            "weights": weights.detach().float().clone().contiguous(),
        }
        self.records.append(record)

    def end(self):
        if self.current is None:
            return
        actual_layers = [record["layer"] for record in self.records]
        if actual_layers != self.expected_layers:
            raise RuntimeError(f"Captured routed layer order differs: {actual_layers}")
        current = self.current
        directory = self.output / "steps" / str(self.concurrency)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            f"{self.step:08d}_{current['capture_window']}_{current['phase']}.pt"
        )
        if path.exists():
            raise FileExistsError(path)
        torch.save({"records": self.records}, path)
        self.files.append(path)
        self.windows[self.concurrency][current["capture_window"]].append(self.step)
        self.event(
            event="captured_step",
            **{key: value for key, value in current.items() if key != "layer"},
            token_rows=int(self.records[0]["hidden"].shape[0]),
            layers=len(self.records),
            path=str(path),
        )
        self.records = []
        self.current = None

    def install(self):
        original_apply = CPUExpertsFp8.apply
        capture = self

        def install_runner(runner_class):
            original_load = runner_class.load_model
            original_execute = runner_class.execute_model

            def load(runner, *args, **kwargs):
                result = original_load(runner, *args, **kwargs)
                capture.register(runner)
                return result

            def execute(runner, scheduler_output, *args, **kwargs):
                if kwargs.get("dummy_run", False):
                    return original_execute(runner, scheduler_output, *args, **kwargs)
                capture.begin(runner, scheduler_output)
                result = original_execute(runner, scheduler_output, *args, **kwargs)
                capture.end()
                return result

            runner_class.load_model = load
            runner_class.execute_model = execute

        for runner_class in (CPUModelRunner, CPUModelRunnerV2):
            install_runner(runner_class)

        def apply(
            experts, output, hidden_states, w1, w2, topk_weights, topk_ids, **kwargs
        ):
            capture.expert_calls += 1
            if capture.current is not None:
                capture.expert(hidden_states, w1, topk_ids, topk_weights)
            result = original_apply(
                experts,
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                **kwargs,
            )
            if capture.current is not None:
                capture.records[-1]["reference_output"] = output.detach().clone()
            return result

        CPUExpertsFp8.apply = apply


def run_requests(engine, prompts, concurrency, output_tokens, label, capture):
    params = SamplingParams(
        temperature=0,
        seed=0,
        max_tokens=output_tokens,
        min_tokens=output_tokens,
        ignore_eos=True,
        detokenize=False,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    submitted = completed = 0
    generated_hash = hashlib.sha256()
    start = time.monotonic()
    while completed < len(prompts):
        while submitted < len(prompts) and submitted - completed < concurrency:
            engine.add_request(
                f"{label}-{submitted:04d}",
                {"prompt_token_ids": prompts[submitted]["prompt_token_ids"]},
                params,
            )
            submitted += 1
        outputs = engine.step()
        for result in outputs:
            if result.finished:
                tokens = result.outputs[0].token_ids
                if len(tokens) != output_tokens:
                    raise RuntimeError(
                        f"{result.request_id} generated {len(tokens)} tokens"
                    )
                generated_hash.update(
                    json.dumps([result.request_id, list(tokens)]).encode()
                )
                completed += 1
        if outputs or completed == len(prompts):
            capture.event(
                event="request_progress",
                label=label,
                submitted=submitted,
                completed=completed,
                total=len(prompts),
                source_step=capture.step,
            )
    capture.event(
        event="requests_complete",
        label=label,
        concurrency=concurrency,
        requests=completed,
        output_tokens=output_tokens,
        generated_token_digest=generated_hash.hexdigest(),
        elapsed_seconds=time.monotonic() - start,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-batched-tokens", type=int, default=1152)
    parser.add_argument(
        "--reference-node",
        type=int,
        choices=(1, 2, 3),
        help="use one 32-core node for TP1 capture; default uses all 96 cores",
    )
    args = parser.parse_args()
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise RuntimeError("Capture hooks require the in-process TP1 executor")
    replay._validate_source_context()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    input_manifest = json.loads((args.inputs / "checkpoint_manifest.json").read_text())
    payload = torch.load(args.inputs / "prompts.pt", weights_only=False)
    prompts = payload["prompts"]
    if len(prompts) != 256 or any(
        len(prompt["prompt_token_ids"]) != 1024 for prompt in prompts
    ):
        raise RuntimeError("Expected exactly 256 fixed 1024-token prompts")
    reference_cpus = tuple(
        range(32, 128)
        if args.reference_node is None
        else range(32 * args.reference_node, 32 * (args.reference_node + 1))
    )
    capture = Capture(args.output_dir, 40, reference_cpus)
    capture.install()
    engine = None
    try:
        engine_args = EngineArgs(
            model=args.checkpoint,
            tokenizer=args.checkpoint,
            dtype="bfloat16",
            tensor_parallel_size=1,
            distributed_executor_backend="uni",
            max_model_len=2048,
            max_num_seqs=128,
            max_num_batched_tokens=args.max_batched_tokens,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            enforce_eager=True,
            kv_cache_memory_bytes=16 * 1024**3,
            language_model_only=True,
            seed=0,
            disable_log_stats=True,
        )
        capture.event(
            event="capture_start",
            engine_args=repr(engine_args),
            prompt_file_sha256=replay._sha256_file(args.inputs / "prompts.pt"),
        )
        engine = LLMEngine.from_engine_args(engine_args, enable_multiprocessing=False)
        if capture.runner is None:
            raise RuntimeError("Capture hooks did not register the active model runner")
        if engine.vllm_config.parallel_config.tensor_parallel_size != 1:
            raise RuntimeError("Reference model must use TP1")
        for concurrency in (8, 32, 128):
            previous_step = capture.step
            previous_expert_calls = capture.expert_calls
            run_requests(
                engine,
                prompts[:concurrency],
                concurrency,
                8,
                f"warmup-{concurrency}",
                capture,
            )
            if (
                capture.step == previous_step
                or capture.expert_calls == previous_expert_calls
            ):
                raise RuntimeError("Capture hooks missed warmup forward execution")
        cutoff = capture.step
        for concurrency in (8, 32, 128):
            capture.concurrency = concurrency
            capture.windows[concurrency] = {"prefill": [], "decode": []}
            run_requests(
                engine,
                prompts,
                concurrency,
                128,
                f"primary-{concurrency}",
                capture,
            )
            if any(len(steps) != 8 for steps in capture.windows[concurrency].values()):
                raise RuntimeError(f"Capture incomplete at concurrency {concurrency}")
        capture.concurrency = None
        records = []
        step_hashes = {}
        for path in capture.files:
            step_hashes[str(path.relative_to(args.output_dir))] = replay._sha256_file(
                path
            )
            records.extend(torch.load(path, weights_only=False, mmap=True)["records"])
        metadata = {
            "model": "Qwen/Qwen3.6-35B-A3B-FP8",
            "fixed_sonnet_prompts": True,
            "input_tokens": 1024,
            "generated_tokens": 128,
            "prompts_per_concurrency": 256,
            "concurrency": [8, 32, 128],
            "mtp_enabled": False,
            "deterministic": True,
            "captured_prefill_steps": 8,
            "captured_decode_steps": 8,
            "capture_after_warmup": True,
            "consecutive_steps": True,
            "preserves_mixed_steps": True,
            "warmup_step_cutoff": cutoff,
            "routed_layer_order": list(range(40)),
            "checkpoint_revision": input_manifest["checkpoint_revision"],
            "checkpoint_sha256": input_manifest["checkpoint_sha256"],
            "prompt_set_sha256": replay._sha256_file(args.inputs / "prompts.pt"),
            "max_num_batched_tokens": args.max_batched_tokens,
            "reference_cpus": list(reference_cpus),
            "reference_threads": len(reference_cpus),
            "openmp_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_WAIT_POLICY",
                    "KMP_BLOCKTIME",
                    "KMP_TPAUSE",
                    "KMP_AFFINITY",
                )
            },
            "source": replay._source_identity(Path(replay.__file__).resolve()),
            "capture_script_sha256": replay._sha256_file(Path(__file__).resolve()),
            "extension": replay._extension_identity(),
            "preload": replay._preload_identity(),
            "step_files_sha256": step_hashes,
            "capture_windows": capture.windows,
        }
        torch.save(
            {"metadata": metadata, "records": records},
            args.output_dir / "corpus.pt",
        )
        capture.event(
            event="capture_complete",
            corpus=str(args.output_dir / "corpus.pt"),
            corpus_sha256=replay._sha256_file(args.output_dir / "corpus.pt"),
            records=len(records),
            windows=capture.windows,
        )
        (args.output_dir / "complete.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    except BaseException:
        failure = {
            "source_step": capture.step,
            "concurrency": capture.concurrency,
            "error": traceback.format_exc(),
        }
        (args.output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2) + "\n"
        )
        capture.event(event="capture_failed", **failure)
        raise
    finally:
        if engine is not None:
            engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
