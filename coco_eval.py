import os
import json
import random
import numpy as np
import torch
from PIL import Image
from scipy import stats
from collections import defaultdict
from pycocotools import mask as coco_mask
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from scipy.ndimage import binary_erosion

COCO_IMAGES_DIR = '/home/jovyan/DuanXu_1010728/coco/val2017'
COCO_ANN_FILE   = '/home/jovyan/DuanXu_1010728/coco/annotations/instances_val2017.json'
SAM3_BPE_PATH   = '/home/jovyan/DuanXu_1010728/SAM3/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'
T = 20
CONFIDENCE_THRESHOLD = 0.1

print("Loading COCO annotations...")
with open(COCO_ANN_FILE) as f:
    coco = json.load(f)

id2img = {img['id']: img for img in coco['images']}
id2anns = defaultdict(list)
for ann in coco['annotations']:
    id2anns[ann['image_id']].append(ann)

cat_names = {c['id']: c['name'] for c in coco['categories']}
downloaded = set(os.listdir(COCO_IMAGES_DIR))
valid_imgs = [img for img in coco['images'] if img['file_name'] in downloaded]
print(f"Found {len(valid_imgs)} images")

def ann_to_binary_mask(ann, height, width):
    if isinstance(ann['segmentation'], list):
        rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
        rle = coco_mask.merge(rle)
    elif isinstance(ann['segmentation'], dict):
        rle = ann['segmentation']
        if isinstance(rle.get('counts'), list):
            rle = coco_mask.frPyObjects(rle, height, width)
    else:
        return np.zeros((height, width), dtype=bool)
    return coco_mask.decode(rle).astype(bool)

def compute_best_iou(pred_mask, gt_masks):
    if len(gt_masks) == 0:
        return 0.0
    pred = pred_mask.astype(bool)
    best = 0.0
    for gt in gt_masks:
        intersection = (pred & gt).sum()
        union = (pred | gt).sum()
        iou = intersection / union if union > 0 else 0.0
        best = max(best, iou)
    return best

def compute_ece(confidences, accuracies, n_bins=10):
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    if len(confidences) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)

        if np.sum(mask) == 0:
            continue

        bin_conf = np.mean(confidences[mask])
        bin_acc = np.mean(accuracies[mask])
        bin_weight = np.sum(mask) / len(confidences)

        ece += np.abs(bin_acc - bin_conf) * bin_weight

    return float(ece)

def enable_mc_dropout(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            if 'decoder' in name.lower() or 'mask' in name.lower():
                module.train()

print("Building SAM3 model...")
model = build_sam3_image_model(bpe_path=SAM3_BPE_PATH, device="cuda")
model = model.to(dtype=torch.float32)

uncertainty_scores = []
iou_errors = []
baseline_ious = []
mc_ious = []
mc_confidences = []
mc_accuracies = []

random.seed(42)
eval_imgs = random.sample(valid_imgs, min(1000, len(valid_imgs)))

for idx, img_info in enumerate(eval_imgs):
    if idx % 20 == 0:
        print(f"Progress: {idx}/{len(eval_imgs)} images started", flush=True)
    img_path = os.path.join(COCO_IMAGES_DIR, img_info['file_name'])
    if not os.path.exists(img_path):
        continue
    image = Image.open(img_path).convert('RGB')
    H, W = img_info['height'], img_info['width']
    anns = id2anns[img_info['id']]
    if len(anns) == 0:
        continue
    cat_ids = list(set(a['category_id'] for a in anns))
    text_prompt = cat_names.get(cat_ids[0], 'object')
    gt_masks = [ann_to_binary_mask(a, H, W) for a in anns]
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD, device="cuda")
    #1. Run Deterministic Baseline (No Dropout)
    model.eval()
    baseline_iou = 0.0
    try:
        with torch.no_grad():
            state_base = processor.set_image(image)
            state_base = processor.set_text_prompt(prompt=text_prompt, state=state_base)
            logits_base = state_base.get("masks_logits")
            
            if logits_base is not None and len(logits_base) > 0:
                logits_base_np = logits_base.detach().cpu().numpy()
                logits_base_np = np.max(logits_base_np, axis=0)
                prob_base = 1 / (1 + np.exp(-logits_base_np))
                pred_binary_base = (prob_base > 0.5).astype(bool)
                
                if pred_binary_base.sum() > 0:
                    baseline_iou = compute_best_iou(pred_binary_base, gt_masks)
    except Exception as e:
        print(f"[{idx+1}] Baseline Error: {e}")
        continue

  

    # ---------------------------------------------------------
    # 2. Run MC-Dropout Pipeline (T passes)
    # ---------------------------------------------------------
    enable_mc_dropout(model) # Turn ON decoder dropouts
    
    logits_list = []
    try:
        for _ in range(T):
            with torch.no_grad():
                state = processor.set_image(image)
                state = processor.set_text_prompt(prompt=text_prompt, state=state)
                logits = state.get("masks_logits")
                if logits is not None and len(logits) > 0:
                    logits_list.append(logits.detach().cpu().numpy())
    except Exception as e:
        print(f"[{idx+1}] MC Error: {e}")
        continue
    if len(logits_list) == 0:
        continue
    logits_stack = np.array([np.max(l, axis=0) for l in logits_list])
    probs = 1 / (1 + np.exp(-logits_stack))

    mean_mask = np.mean(probs, axis=0)
    std_mask  = np.std(probs, axis=0)
    entropy_map = -mean_mask * np.log(mean_mask + 1e-8) - (1 - mean_mask) * np.log(1 - mean_mask + 1e-8)
    
    pred_binary = (mean_mask > 0.5).astype(bool)
    if pred_binary.sum() == 0:
        continue
    
    # NEW: confidence of final MC prediction
    mean_confidence = float(np.mean(mean_mask[pred_binary]))
    
    eroded = binary_erosion(pred_binary, iterations=3)
    boundary = pred_binary & ~eroded
    
    if boundary.sum() == 0:
        uncertainty = float(np.mean(entropy_map[pred_binary]))
    else:
        uncertainty = float(np.mean(entropy_map[boundary]))
    
    iou = compute_best_iou(pred_binary, gt_masks)
    iou_error = 1.0 - iou
    
    if iou > 0.05:
            baseline_ious.append(baseline_iou)
            mc_ious.append(iou)
            uncertainty_scores.append(uncertainty)
            iou_errors.append(iou_error)
            mc_confidences.append(mean_confidence)
            mc_accuracies.append(iou)
    
    print(f"[{idx+1}/{len(eval_imgs)}] {img_info['file_name']} | prompt={text_prompt} | uncertainty={uncertainty:.4f} | conf={mean_confidence:.4f} | IoU={iou:.4f}", flush=True)

if len(uncertainty_scores) > 5:
    rho, pval = stats.spearmanr(uncertainty_scores, iou_errors)

    mean_baseline_iou = np.mean(baseline_ious)
    mean_mc_iou = np.mean(mc_ious)
    iou_delta = mean_baseline_iou - mean_mc_iou
    iou_drop = max(0.0, iou_delta)
    ece = compute_ece(mc_confidences, mc_accuracies, n_bins=10)

    print("-" * 50)
    print(f"N = {len(uncertainty_scores)} valid images")
    print(f"Spearman rho: {rho:.4f} (p-value: {pval:.4f})")
    print(f"Proposal target rho > 0.3 -> {'PASSED' if rho > 0.3 else 'FAILED'}")
    print("-" * 50)
    print("=== PERFORMANCE EVALUATION (Rubric Requirement) ===")
    print(f"Deterministic SAM3 (Baseline) Mean IoU : {mean_baseline_iou:.4f}")
    print(f"MC-Dropout SAM3 (T={T}) Mean IoU       : {mean_mc_iou:.4f}")
    print(f"IoU Delta (Baseline - MC)              : {iou_delta:.4f}")
    print(f"Effective IoU Drop                     : {iou_drop:.4f}")
    print(f"Proposal target Drop < 0.02 (2%)       : {'PASSED' if iou_drop < 0.02 else 'FAILED'}")
    print(f"ECE (10 bins)                          : {ece:.4f}")
    print("-" * 50)

    results = {
        "n_valid_images": len(uncertainty_scores),
        "spearman_rho": float(rho),
        "p_value": float(pval),
        "mean_baseline_iou": float(mean_baseline_iou),
        "mean_mc_iou": float(mean_mc_iou),
        "iou_delta": float(iou_delta),
        "iou_drop": float(iou_drop),
        "ece_10bins": float(ece)
    }

    with open("spearman_results.json", "w") as f:
        json.dump(results, f, indent=2)

else:
    print(f"Not enough valid samples: {len(uncertainty_scores)}")
