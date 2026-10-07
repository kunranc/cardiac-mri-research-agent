# Autonomous AI research agent for cardiac MRI segmentation

**AI for Science Hackathon, team project.** We used an AI agent to run a full medical-imaging research cycle overnight: literature review, experiment design, cloud GPU training, statistical model selection and a single held-out test evaluation. The task was segmenting the right ventricle (RV), myocardium (MYO) and left ventricle (LV) in cardiac MRI from the [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) dataset, using 2D U-Net-family models only.

## My contribution

- Designed the research-agent prompt and the experimental protocol. These set the rules: 2D U-Net family only, pre-registered decision thresholds, and no access to the test set until the final model is frozen.
- Built the training and evaluation pipeline with Claude Code, and ran it on Hugging Face Jobs (T4 GPUs).
- Ran the autonomous loop overnight: 25 GPU training runs (about 35 T4-hours, roughly $14 of compute), from a baseline to a frozen, test-evaluated model.

## Results

The test set was patients 101-150 (100 volumes). It was evaluated **once**, after the model and protocol were frozen.

| Model | Mean Dice | RV | MYO | LV | Hausdorff (mm) | HD95 (mm) |
|---|---|---|---|---|---|---|
| Baseline U-Net, 5-fold ensemble | 0.919 | 0.911 | 0.903 | 0.943 | 9.5 | 3.7 |
| **Final: tuned U-Net, 5-fold ensemble + LCC** | **0.929** | **0.925** | **0.915** | **0.948** | **8.2** | **3.0** |

- The final model improves on the same-protocol baseline by **+0.0105 Dice** [95% CI +0.007, +0.015], p = 7.5e-9, with 45 of 50 test patients better.
- **What helped:** the architecture is unchanged from the baseline (1.81M parameters). The gain came from a tuned optimiser setting, stronger 2D augmentation, a 5-fold ensemble, and keeping only the largest connected component per class (LCC).
- **What didn't help:** a wider network (4× parameters), residual blocks, scSE, attention gates, InstanceNorm and deep supervision. Each was tested and rejected against the pre-registered threshold.
- **Weakest case:** end-systolic hearts with hypertrophic cardiomyopathy, where the LV cavity is nearly empty.

![Validation Dice across experiments](reports/figures/progression.png)

![Test Dice per class](reports/figures/test_dice_box.png)

## What's in this repo

| Path | Contents |
|---|---|
| [`reports/REPORT.md`](reports/REPORT.md) | Full research report: method, every experiment, statistics, limitations ([HTML](reports/report.html), [PDF](reports/report.pdf)) |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Every training run with validation metrics, GPU time and cost |
| [`reports/figures/`](reports/figures/) | Learning curves, ablations, test-set analysis |
| [`reports/test_summary.json`](reports/test_summary.json) | Raw test metrics with bootstrap confidence intervals |

The training code, exact configurations and agent prompt are kept private. Get in touch if you'd like to discuss them.
