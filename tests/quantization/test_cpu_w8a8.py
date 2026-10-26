# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import subprocess
import sys
import textwrap

import pytest

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
def test_cpu_w8a8_tile_scheduler_preserves_generation(model, dtype, tmp_path):
    """Tile scheduling preserves greedy generation through the real LLM path."""
    script = textwrap.dedent(
        """
        import json
        import sys
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=sys.argv[1],
            dtype=sys.argv[2],
            max_model_len=256,
            max_num_seqs=2,
            enforce_eager=True,
            seed=0,
            kernel_config={"enable_jit_warmup": False},
        )
        outputs = llm.generate(
            [
                "The capital of France is",
                "Explain why a CPU cache helps matrix multiplication:",
            ],
            SamplingParams(temperature=0, max_tokens=16),
        )
        with open(sys.argv[3], "w") as stream:
            json.dump([output.outputs[0].token_ids for output in outputs], stream)
        """
    )
    cpus = sorted(os.sched_getaffinity(0))[:6]
    groups = [len(cpus) // 3 + (i < len(cpus) % 3) for i in range(3)]
    env = os.environ.copy()
    env["VLLM_CPU_OMP_THREADS_BIND"] = ",".join(map(str, cpus))
    env["VLLM_CPU_SGL_THREAD_GROUPS"] = ",".join(str(size) for size in groups if size)
    env["OMP_NUM_THREADS"] = str(len(cpus))
    env["OMP_THREAD_LIMIT"] = str(len(cpus))
    env["OMP_DYNAMIC"] = "FALSE"
    env["VLLM_CPU_KVCACHE_SPACE"] = "1"
    results = []
    for mode in ("legacy", "tile_all"):
        env["VLLM_CPU_SGL_SCHEDULER"] = mode
        result_path = tmp_path / f"{mode}.json"
        subprocess.run(
            [sys.executable, "-c", script, model, dtype, str(result_path)],
            check=True,
            env=env,
        )
        results.append(json.loads(result_path.read_text()))

    assert all(tokens for tokens in results[0])
    assert results[0] == results[1]
