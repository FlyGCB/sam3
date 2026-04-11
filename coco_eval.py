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

COCO_IMAGES_DIR = '/home/jovyan/DuanXu_1010728/SAM3/sam3/assets/uncertainImages'
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

random.seed(42)
eval_imgs = random.sample(valid_imgs, min(100, len(valid_imgs)))

for idx, img_info in enumerate(eval_imgs):
    img_path = os.path.join(COCO_IMAGES_DIR, img_info['file_name'])
    if not os.path.exists(img_path):
        continue
    image = Image.open(img_path).convert('RGB')
    H, W = img_info['height'], img_info['width']
    anns = id2anns[img_info['id']]
    if len(anns) == 0:
        continue
    cat_ids = list(set(a['category_id'] for a in anns))
    text_prompt = ' and '.join(set(cat_names.get(c, 'object') for c in cat_ids[:3]))
    gt_masks = [ann_to_binary_mask(a, H, W) for a in anns]
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD, device="cuda")
    model.eval()
    enable_mc_dropout(model)
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
        print(f"  [{idx+1}] Error: {e}")
        continue
    if len(logits_list) == 0:
        continue
    logits_stack = np.array([np.max(l, axis=0) for l in logits_list])
    probs = 1 / (1 + np.exp(-logits_stack))
    mean_mask = np.mean(probs, axis=0)
    std_mask  = np.std(probs, axis=0)
    pred_binary = (mean_mask > 0.5).astype(bool)
    if pred_binary.sum() == 0:
        continue
    uncertainty = float(np.mean(std_mask[pred_binary]))
    iou = compute_best_iou(pred_binary, gt_masks)
    iou_error = 1.0 - iou
    uncertainty_scores.append(uncertainty)
    iou_errors.append(iou_error)
    print(f"  [{idx+1}/100] {img_info['file_name']} | prompt={text_prompt} | uncertainty={uncertainty:.4f} | IoU={iou:.4f}")

if len(uncertainty_scores) >= 5:
    rho, pval = stats.spearmanr(uncertainty_scores, iou_errors)
    print(f"\n{'='*50}")
    print(f"Spearman rho = {rho:.4f}  (p-value = {pval:.4f})")
    print(f"N = {len(uncertainty_scores)} images")
    print(f"Proposal target: rho > 0.3 -> {'PASSED' if rho > 0.3 else 'FAILED'}")
    results = {'spearman_rho': float(rho), 'p_value': float(pval), 'n_images': len(uncertainty_scores), 'uncertainty_scores': uncertainty_scores, 'iou_errors': iou_errors}
    with open('spearman_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to spearman_results.json")
else:
    print(f"Not enough valid samples: {len(uncertainty_scores)}")
