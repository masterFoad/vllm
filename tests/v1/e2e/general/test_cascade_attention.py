# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

from ....utils import create_new_process_for_each_test


@create_new_process_for_each_test()
@pytest.mark.parametrize("attn_backend", ["FLASH_ATTN", "FLASHINFER"])
def test_cascade_attention(example_system_message, attn_backend):
    prompt = "\n<User>: Implement fibonacci sequence in Python.\n<Claude>:"

    if attn_backend == "FLASHINFER":
        pytest.skip(
            "This test is failing with FlashInfer backend and "
            "needs investigation. See issue #25679."
        )

    llm = LLM(
        model="Qwen/Qwen2-1.5B-Instruct", attention_config={"backend": attn_backend}
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

    # No cascade attention.
    single_prompt = [example_system_message + prompt]
    responses = llm.generate(single_prompt, sampling_params)
    ref_output = responses[0].outputs[0].text

    # (Probably) Use cascade attention.
    prompts = [example_system_message + prompt] * 64
    responses = llm.generate(prompts, sampling_params)
    for response in responses:
        assert response.outputs[0].text == ref_output


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="FlashInfer cascade debug repro is CUDA-only.",
)
@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
    ],
)
def test_flashinfer_repeated_batch_mismatch_repro(
    example_system_message, model_name
):
    prompt = "\n<User>: Implement fibonacci sequence in Python.\n<Claude>:"

    llm = LLM(
        model=model_name,
        attention_config={"backend": "FLASHINFER"},
        enforce_eager=True,
        disable_log_stats=True,
        disable_cascade_attn=False,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64, seed=42)

    single_prompt = [example_system_message + prompt]
    responses = llm.generate(single_prompt, sampling_params)
    ref_output = responses[0].outputs[0].text

    prompts = [example_system_message + prompt] * 16
    responses = llm.generate(prompts, sampling_params)

    mismatches = [
        (idx, response.outputs[0].text)
        for idx, response in enumerate(responses)
        if response.outputs[0].text != ref_output
    ]
    assert not mismatches, (
        f"FlashInfer repeated-batch output diverged on model={model_name!r} "
        f"for {len(mismatches)}/{len(prompts)} identical shared-prefix prompts. "
        f"First mismatch: {mismatches[0]!r}; reference={ref_output!r}"
    )
