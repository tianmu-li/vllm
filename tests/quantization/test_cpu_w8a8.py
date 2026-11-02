# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from vllm.model_executor.layers.utils import check_cpu_sgl_kernel
from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

MODELS = [
    "RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w8a8",  # INT8 W8A8 MoE
]
DTYPE = ["bfloat16"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", DTYPE)
def test_cpu_w8a8(vllm_runner, model, dtype):
    with vllm_runner(model, dtype=dtype) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=32)
    assert output
    print(output)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.skipif(
    not check_cpu_sgl_kernel(32, 32, torch.int8)
    or not hasattr(torch.ops._C, "int8_scaled_mm_with_quant"),
    reason="requires SGL INT8 AMX kernels",
)
def test_cpu_w8a8_sgl_modes_preserve_large_prefill_generation(model, dtype, tmp_path):
    """Direct AMX execution preserves greedy output on a large prefill."""
    script = textwrap.dedent(
        """
        import json
        import sys
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=sys.argv[1],
            dtype=sys.argv[2],
            max_model_len=1536,
            max_num_seqs=2,
            max_num_batched_tokens=1536,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            enforce_eager=True,
            seed=0,
            kernel_config={"enable_jit_warmup": False},
        )
        tokenizer = llm.get_tokenizer()

        def repeated_prompt(prefix):
            prompt = prefix
            while len(tokenizer.encode(prompt, add_special_tokens=False)) < 768:
                prompt += prefix
            return prompt

        prompts = [
            repeated_prompt("The capital of France is Paris. "),
            repeated_prompt("A CPU cache helps matrix multiplication. "),
        ]
        assert min(
            len(tokenizer.encode(prompt, add_special_tokens=False))
            for prompt in prompts
        ) >= 768
        llm.sleep(level=0)
        llm.enqueue(
            prompts,
            SamplingParams(temperature=0, max_tokens=32),
        )
        llm.wake_up(tags=["scheduling"])
        outputs = llm.wait_for_completion()
        with open(sys.argv[3], "w") as stream:
            json.dump([output.outputs[0].token_ids for output in outputs], stream)
        """
    )
    physical_cores: dict[tuple[str, str], int] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        core = (
            (topology / "physical_package_id").read_text().strip(),
            (topology / "core_id").read_text().strip(),
        )
        physical_cores.setdefault(core, cpu)
    cpus = list(physical_cores.values())[:24]
    env = os.environ.copy()
    env.pop("KMP_AFFINITY", None)
    env["VLLM_CPU_OMP_THREADS_BIND"] = ",".join(map(str, cpus))
    env["OMP_NUM_THREADS"] = str(len(cpus))
    env["OMP_THREAD_LIMIT"] = str(len(cpus))
    env["OMP_DYNAMIC"] = "FALSE"
    env["VLLM_CPU_KVCACHE_SPACE"] = "1"
    results = []
    for mode in ("legacy", "generic"):
        env["VLLM_CPU_SGL_SCHEDULER"] = mode
        env["VLLM_CPU_SGL_DISPATCH"] = "0" if mode == "legacy" else "1"
        result_path = tmp_path / f"{mode}.json"
        completed = subprocess.run(
            [sys.executable, "-c", script, model, dtype, str(result_path)],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
        if mode != "legacy":
            dispatches = [
                json.loads(line.removeprefix("SGL_DISPATCH "))
                for line in completed.stderr.splitlines()
                if line.startswith("SGL_DISPATCH ")
            ]
            assert dispatches, "generation did not execute a dense INT8 call"
            assert all(
                record["kernel"]
                in ("generic", "amx_cohort", "amx_local", "legacy_team")
                for record in dispatches
            )
            assert all(record["actual_team_size"] == len(cpus) for record in dispatches)
            l2 = Path(f"/sys/devices/system/cpu/cpu{cpus[0]}/cache/index2/size")
            if len(cpus) == 24 and l2.read_text().strip() == "2048K":
                assert any(
                    record["kernel"] in ("amx_cohort", "amx_local")
                    for record in dispatches
                ), "large prefill did not execute direct AMX"
            for record in dispatches:
                if record["kernel"] in ("amx_cohort", "legacy_team"):
                    assert (record["kernel"] == "legacy_team") == (
                        record["replicas"] == 1
                    )
                    assert record["activation_bytes"] > 0
                    assert (record["scratch_bytes"] == 0) == (record["replicas"] == 1)
        results.append(json.loads(result_path.read_text()))

    assert all(tokens for result in results for tokens in result)
    assert all(result == results[0] for result in results[1:])
