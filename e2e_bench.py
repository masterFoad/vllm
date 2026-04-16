import time
from vllm import LLM, SamplingParams


def run_e2e():
    # ~19,000 tokens to simulate the OCR document
    prompt = "document text " * 9500

    print("Initializing LLM...")
    llm = LLM(
        model="mistralai/Ministral-3-3B-Instruct-2512",
        download_dir="/dccstor/video-ai/work/foadad/foads_vault",
        enable_prefix_caching=True,
        enforce_eager=True,  # Isolate host overhead, avoid cuda graph compilation time
        gpu_memory_utilization=0.4,
        max_model_len=32768,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

    print("Run 1: Fill the prefix cache...")
    llm.generate([prompt + " Question 1"], SamplingParams(max_tokens=1))

    print("Run 2: Trigger Prefix Cache (Issue 4) and Decode (Issue 2)...")
    # Batch of 4 requests to amplify the host overhead slightly and show batching
    prompts = [prompt + f" Question {i}" for i in range(2, 6)]

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end = time.perf_counter()

    # Calculate average TTFT
    ttfts = []
    for out in outputs:
        if out.metrics.first_token_time and out.metrics.first_scheduled_time:
            ttfts.append(
                out.metrics.first_token_time - out.metrics.first_scheduled_time
            )

    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0

    print("\n" + "=" * 40)
    print(f"Average TTFT: {avg_ttft * 1000:.2f} ms")
    print(f"Total Time for Batch (100 tokens each): {end - start:.2f} s")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    run_e2e()
