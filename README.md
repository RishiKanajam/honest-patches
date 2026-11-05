# Honest Patches

Code and data for **"Permission, Not Doubt: A Single Planted Authorization Makes an Honest Coding Model Cheat"**, a Paper Track entry for Google's Gemma 4 Developer Agent competition on Kaggle.

Gemma 4 31B (the competition's `gemma-4-31b-it-qat-w4a16-ct` checkpoint) never cheats on Impossible-LiveCodeBench without a note (0/206 runs) and ignores a note saying the tests are flaky (0/103). A note that also says special-casing them is acceptable makes it cheat in 14.9% of runs (46/309) and attempt to in 22.7% (70/309). A system-prompt guard cuts successful cheating by 41% but leaves 8.7%.

## Contents

| Path | What it is |
| --- | --- |
| `runners/lcb_runner.py` | Repair loop (3 attempts with test feedback), the five conditions, exit-loophole guard, cheat detector; talks to any OpenAI-compatible server |
| `runners/test_lcb_runner.py` | 18 unit tests: `cd runners && pytest -q` |
| `analysis/make_figures.py` | Rebuilds the results table and both figures from the run log, on a CPU |
| `data/honest_patches_runs_v1.jsonl` | All 1,030 runs: every attempt's code, model reply, test output and finish reason |
| `scripts/download_tasks.py` | Downloads the Impossible-LiveCodeBench tasks from Hugging Face |
| `notebooks/run_on_kaggle.ipynb` | The Kaggle notebook used for the GPU runs (4×L4, competition starter + runner) |

## Reproduce the paper's numbers (CPU, seconds)

```bash
pip install matplotlib pandas pyarrow
python analysis/make_figures.py data/honest_patches_runs_v1.jsonl
```

## Re-run the experiments (GPU)

```bash
pip install datasets pandas pyarrow
python scripts/download_tasks.py
python runners/lcb_runner.py --split conflicting --condition injection --seeds 0 \
    --limit 103 --workers 8 --model <served-model> --base-url http://localhost:8000/v1
```

Conditions: `baseline`, `injection_soft`, `injection`, `injection_guard`. Use `--split original` for the capability check. Re-running a command resumes where it stopped. The runner executes model-written code locally in a subprocess with a timeout; run it somewhere disposable.

## The run log

One JSON object per run: `task_id`, `split`, `condition`, `seed`, `model`, `quant`, `hardware`, `temperature`, `final_passed`, `n_attempts`, `wall_time_s`, and `attempts` (each with `response`, `code`, `test_output`, `passed`, `finish_reason`). Tasks are referenced by ID; the task text is not included, because the source dataset declares no license. Run `scripts/download_tasks.py` to get it.

One seed-1 guard run was lost before its log was saved; the guard runs here are its rerun. Details are in the paper.

**Intended use.** These trajectories document a model gaming tests. Please use them to study and prevent that behaviour, not to train models to do it.

## Credits

Tasks: ImpossibleBench (Zhong, Raghunathan and Carlini, 2025), https://github.com/safety-research/impossiblebench. Model: Gemma 4, Google.

## License

Code: MIT (see `LICENSE`). Run log: CC BY 4.0.
