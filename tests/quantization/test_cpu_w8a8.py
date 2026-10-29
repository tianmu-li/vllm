# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import subprocess
import sys
import textwrap

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
    """SGL scheduler modes preserve greedy output on an eligible large prefill."""
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
    cpus = sorted(os.sched_getaffinity(0))[:6]
    env = os.environ.copy()
    env["VLLM_CPU_OMP_THREADS_BIND"] = ",".join(map(str, cpus))
    env["OMP_NUM_THREADS"] = str(len(cpus))
    env["OMP_THREAD_LIMIT"] = str(len(cpus))
    env["OMP_DYNAMIC"] = "FALSE"
    env["VLLM_CPU_KVCACHE_SPACE"] = "1"
    results = []
    for mode in (
        "legacy",
        "logical96_dense",
        "flat_32x32",
        "flat_128x64",
        "flat_128x128",
        "flat_auto",
    ):
        env["VLLM_CPU_SGL_SCHEDULER"] = mode
        result_path = tmp_path / f"{mode}.json"
        subprocess.run(
            [sys.executable, "-c", script, model, dtype, str(result_path)],
            check=True,
            env=env,
        )
        results.append(json.loads(result_path.read_text()))

    assert all(tokens for result in results for tokens in result)
    assert all(result == results[0] for result in results[1:])
