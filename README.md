# 🎓 Course Project: Uncertainty-Aware Segmentation with SAM3

**61.502 Deep Learning for Enterprise | Y2026 | SUTD**

> Inference-Time Uncertainty Estimation for Instance Segmentation using Monte Carlo Dropout on SAM3

## Team Members

| Name | Student ID | Contribution |
|---|---:|---|
| Shiva Prasad | 1009737 | Model integration, MC-Dropout pipeline, GitHub |
| Duan Xu | 1010728 | Inference script, debugging, COCO evaluation, entropy map |
| Lee Kai Boo | 1011130 | Visualisation, report writing, analysis |

## Project Overview

This project studies inference-time uncertainty estimation for SAM3 using Monte Carlo (MC) Dropout, a zero-retraining Bayesian approximation method. By selectively enabling dropout in decoder and mask-head layers during inference, the pipeline generates stochastic predictions from which mean masks, variance maps, entropy maps, and per-object uncertainty scores can be computed.

The goal is to determine whether uncertainty estimates from a pretrained SAM3 model correlate meaningfully with segmentation quality, without retraining or changing the model architecture.

## Final Results

**Quantitative results on a reproducible 1000-image COCO val2017 subset (seed=42):**
- Spearman ρ = **0.4498**
- p-value = **0.0000**
- N = **559**
- Deterministic baseline mean best-IoU = **0.2307**
- MC-Dropout mean best-IoU = **0.2401**
- Simplified 10-bin ECE proxy = **0.3264**

**Runtime:**
- T=3: **0.81s/image**
- T=20: **~5.4s/image**

**Main finding:**  
Boundary-region entropy is a more useful uncertainty metric than full-mask standard deviation for predicting segmentation quality.

## Repository Structure

| Path | Description |
|---|---|
| `inference_script.py` | Main MC-Dropout inference pipeline |
| `coco_eval.py` | Evaluation script for uncertainty-error correlation |
| `create_stitched_images.py` | Utility script for qualitative stitched comparison figures |
| `spearman_results.json` | Final quantitative results |
| `assets/figures/` | Selected qualitative figures used in the report |
| `report/` | Final project report PDF |
| `examples/` | Original SAM3 example notebooks |
| `sam3/` | Core SAM3 package code |

## Selected Qualitative Results

| Groceries scene | Truck scene |
|---|---|
| ![groceries](assets/figures/stitched_comparison_3x3_1_groceries.png) | ![truck](assets/figures/stitched_comparison_3x3_2_truck.png) |

## Environment Setup

```bash
# Create conda environment in the project directory
conda create --prefix ./env python=3.12 -y
source /opt/conda/bin/activate ./env

# Install PyTorch with CUDA
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# Install the repository and notebook dependencies
pip install -e .
pip install -e ".[notebooks]"

# Extra dependencies used in this project
pip install pycocotools scipy
```

## Hugging Face Authentication

SAM3 checkpoints require Hugging Face authentication.

```bash
hf auth login
```

Make sure you have access to the SAM3 / SAM3.1 checkpoints before running inference.

## Running Inference

Place test images in:

```bash
assets/uncertainImages/
```

Run the inference pipeline:

```bash
python inference_script.py
```

Outputs are written to:

```bash
inference_results_uncertainty/
```

Typical outputs include:
- segmentation visualisations
- uncertainty maps
- entropy maps
- stitched comparison figures

## Reproducing Quantitative Results

First download COCO 2017 annotations:

```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
python -c "import zipfile; zipfile.ZipFile('annotations_trainval2017.zip').extractall('.')"
```

Then download the reproducible 1000-image subset:

```bash
python coco/download_1000.py
```

Move the downloaded images into:

```bash
assets/uncertainImages/
```

Then run evaluation:

```bash
python coco_eval.py
```

Expected output:

```bash
spearman_results.json
```

Expected result summary:
- Spearman rho = **0.4498**
- p-value = **0.0000**
- N = **559**

## Reproducing Report Figures

### Qualitative stitched comparisons

```bash
python inference_script.py
```

Expected outputs include:
- `inference_results_uncertainty/stitched_comparison_3x3_1_groceries.png`
- `inference_results_uncertainty/stitched_comparison_3x3_2_truck.png`

### Per-image uncertainty and entropy maps

```bash
python inference_script.py
```

Expected outputs include:
- `inference_results_uncertainty/result_*_uncertainty_map.png`
- `inference_results_uncertainty/result_*_entropy_map.png`

## Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| MC samples `T` | 3, 20 | Both reported in final evaluation |
| Confidence threshold | 0.1 | Low threshold to retain uncertain predictions |
| Active dropout layers | 30 | Decoder / mask-head only |
| IoU filter threshold | 0.05 | Filters cases with no meaningful target |
| Random seed | 42 | Reproducible subset selection |

## Hardware Notes

- NVIDIA GPU with CUDA 12.6+
- Python 3.12
- PyTorch 2.10.0
- Tested on SUTD AI Mega Cluster

## Reproducibility Notes

This repository includes:
- documented scripts
- selected qualitative figures
- final quantitative results
- exact reproduction commands
- final report PDF

> Note: COCO images are not uploaded to GitHub because of space constraints. Download them separately before running the pipeline.

📄 **Final Report:** `report/SAM3_Final_Report.pdf`

---

# SAM 3: Segment Anything with Concepts

The section below preserves the base SAM3 project context and installation references for upstream usage.

## Project Links

- [SAM 3 Paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
- [SAM 3 Project Page](https://ai.meta.com/sam3)
- [SAM 3 Demo](https://segment-anything.com/)
- [SAM 3 Blog](https://ai.meta.com/blog/segment-anything-model-3/)

![SAM 3 architecture](assets/model_diagram.png?raw=true)

SAM 3 is a unified foundation model for promptable segmentation in images and videos. It supports text and visual prompts including points, boxes, and masks. Compared to earlier SAM-family systems, SAM3 extends open-vocabulary promptable segmentation and broader concept coverage.

## Original Installation Notes

### Prerequisites

- Python 3.12 or higher
- PyTorch 2.7 or higher
- CUDA-compatible GPU with CUDA 12.6 or higher

```bash
conda create -n sam3 python=3.12
conda activate sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install -e ".[notebooks]"
```

### Optional dependencies

```bash
pip install einops ninja
pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/ronghanghu/cc_torch.git
```

## Examples

The `examples/` directory contains notebooks for:
- image prompting
- video prompting
- batched inference
- agent-based prompting
- SA-Co evaluation and visualisation

To run an example notebook:

```bash
jupyter notebook examples/sam3_image_predictor_example.ipynb
```

## License

This repository uses the SAM License. See `LICENSE` for details.

## Citation

If you use SAM3 in academic work, please cite the original SAM3 paper.

```bibtex
@misc{carion2025sam3segmentconcepts,
      title={SAM 3: Segment Anything with Concepts},
      author={Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and Ronghang Hu and Didac Suris and Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo and Arpit Kalla and Markus Marks and Joseph Greer and Meng Wang and Peize Sun and Roman Rädle and Triantafyllos Afouras and Effrosyni Mavroudi and Katherine Xu and Tsung-Han Wu and Yu Zhou and Liliane Momeni and Rishi Hazra and Shuangrui Ding and Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár and Nikhila Ravi and Kate Saenko and Pengchuan Zhang and Christoph Feichtenhofer},
      year={2025},
      eprint={2511.16719},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16719},
}
```