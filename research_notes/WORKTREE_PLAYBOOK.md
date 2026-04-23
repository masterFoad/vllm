# vLLM Shared Environment & Worktree Playbook

**Context:** This server has strict disk quotas. A single vLLM virtual environment with PyTorch, CUDA, and compiled C++ extensions (`.so` files) takes up ~10GB. If you create a new `.venv` and run `pip install -e .` for every git worktree, you will instantly hit the `[Errno 122] Disk quota exceeded` error and your builds will fail.

To solve this, we use a **Single Shared Virtual Environment** and a **C-Extension Copy Trick**.

---

## 1. The Shared Virtual Environment

We are using the virtual environment located at:
👉 `/dccstor/video-ai/work/foadad/vllm-research/vllm/.venv`

This environment already has:
- `torch`, `torchvision`, `torchaudio` (cu129)
- All vLLM build dependencies
- vLLM installed in editable mode (`__editable__.vllm...pth`) pointing to the main `vllm` directory.

**DO NOT create new `.venv` folders in your worktrees.**
**DO NOT run `uv pip install -e .` in your worktrees.**

---

## 2. How to Create and Use a New Worktree

When you need to work on a new issue or feature, follow these exact steps:

### Step 1: Create the Worktree
Run this from inside the main `vllm` repo:
```bash
git worktree add ../wt-my-new-feature -b my-new-feature
cd ../wt-my-new-feature
```

### Step 2: Activate the Shared Venv
Activate the master environment from the main repo:
```bash
source ../vllm/.venv/bin/activate
```

### Step 3: The C-Extension Copy Trick (CRITICAL)
Because you didn't run `pip install -e .` in this worktree, Python doesn't know where the compiled C++ kernels (`vllm._C.abi3.so`, etc.) are for this specific folder. If you try to run code now, it will fail with `ImportError: cannot import name '_C'`.

To fix this without recompiling (which takes 10 minutes and blows up the disk quota), simply copy the pre-compiled binaries from the main repo into your worktree:
```bash
cp ../vllm/vllm/*.so vllm/
```
*(Note: These files are ~460MB. When you are completely done with a worktree, you should delete these `.so` files to reclaim space).*

### Step 4: Run Tests and Benchmarks with `PYTHONPATH=.`
Because the shared `.venv` has vLLM installed in editable mode pointing to the *main* `vllm` directory, running `pytest` normally will test the code in the main repo, NOT your worktree.

To force Python to use your local worktree edits, you **MUST** prefix every execution command with `PYTHONPATH=.`:

```bash
# Correct way to run a test in your worktree:
PYTHONPATH=. pytest tests/v1/worker/test_my_feature.py

# Correct way to run a benchmark in your worktree:
PYTHONPATH=. python benchmarks/my_benchmark.py
```

---

## 3. Cleanup Routine (Reclaiming Disk Space)

When you hit a disk quota error, do not panic. Run these cleanup steps:

1. **Delete accidental venvs:**
   ```bash
   rm -rf ../wt-*/.venv
   ```
2. **Delete copied C-extensions from INACTIVE worktrees:**
   If you are done with `wt-issue1`, delete its `.so` files to save ~460MB:
   ```bash
   rm ../wt-issue1/vllm/*.so
   ```
3. **Remove old worktrees entirely:**
   If the PR is merged or pushed, nuke the worktree:
   ```bash
   git worktree remove ../wt-issue1
   ```