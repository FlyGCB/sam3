import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def create_stitched_comparison(image_paths, output_dir):
    """Create 2x4 stitched grid including entropy map"""
    for i, image_path in enumerate(image_paths, 1):
        base_name = f"result_{i}"
        
        # Load original image
        image_name = "groceries.jpg" if i == 1 else "truck.jpg"
        actual_original_path = os.path.join(os.path.dirname(output_dir), "assets", "uncertainImages", image_name)
        actual_original = Image.open(actual_original_path)

        # Load all processed images
        original_segmentation_path = os.path.join(output_dir, f"{base_name}_original_segmentation.png")
        uncertainty_map_path = os.path.join(output_dir, f"{base_name}_uncertainty_map.png")
        entropy_map_path = os.path.join(output_dir, f"{base_name}_entropy_map.png")
        mc_segmentation_path = os.path.join(output_dir, f"{base_name}_mc_segmentation.png")

        original_segmentation = Image.open(original_segmentation_path)
        uncertainty_map = Image.open(uncertainty_map_path)
        entropy_map = Image.open(entropy_map_path)
        mc_segmentation = Image.open(mc_segmentation_path)

        # Create 2x4 figure
        fig, axes = plt.subplots(2, 4, figsize=(28, 14))

        # Row 1: Original
        axes[0, 0].imshow(actual_original)
        axes[0, 0].set_title('Original Image', fontsize=13, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(original_segmentation)
        axes[0, 1].set_title('Original SAM3\n(Segmentation)', fontsize=13, fontweight='bold')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(uncertainty_map)
        axes[0, 2].set_title('Uncertainty Map\n(Variance)', fontsize=13, fontweight='bold')
        axes[0, 2].axis('off')

        axes[0, 3].imshow(entropy_map)
        axes[0, 3].set_title('Entropy Map\n(MC-Dropout)', fontsize=13, fontweight='bold')
        axes[0, 3].axis('off')

        # Row 2: MC-Dropout
        axes[1, 0].imshow(actual_original)
        axes[1, 0].set_title('Original Image', fontsize=13, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(mc_segmentation)
        axes[1, 1].set_title('MC-Dropout\n(Segmentation)', fontsize=13, fontweight='bold')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(uncertainty_map)
        axes[1, 2].set_title('Uncertainty Map\n(Variance)', fontsize=13, fontweight='bold')
        axes[1, 2].axis('off')

        axes[1, 3].imshow(entropy_map)
        axes[1, 3].set_title('Entropy Map\n(MC-Dropout)', fontsize=13, fontweight='bold')
        axes[1, 3].axis('off')

        fig.suptitle(f'SAM3 Analysis Grid - {image_name}', fontsize=18, fontweight='bold')
        
        stitched_path = os.path.join(output_dir, f"stitched_comparison_{i}_{image_name.replace('.jpg', '')}.png")
        plt.tight_layout()
        plt.savefig(stitched_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Created stitched comparison: {stitched_path}")

if __name__ == "__main__":
    output_dir = "inference_results_uncertainty"
    image_paths = ["groceries.jpg", "truck.jpg"]