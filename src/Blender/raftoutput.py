import os
import glob
import cv2
import numpy as np
import raft_segment
from seg import find_largest_gap_from_flow


def create_2x2_grid(orig_img_path, gt_mask_path, flow_img_path, pred_mask_path, iou_value):
    """
    Create a 2x2 grid visualization with:
    - Top-left: Original image
    - Top-right: Ground truth mask
    - Bottom-left: Flow visualization
    - Bottom-right: Predicted mask
    
    Returns the grid image with IoU text overlay.
    """
    # Load all images
    orig_img = cv2.imread(orig_img_path)
    gt_mask = cv2.imread(gt_mask_path)
    flow_img = cv2.imread(flow_img_path)
    pred_mask = cv2.imread(pred_mask_path)
    
    # Convert grayscale masks to BGR for consistent display
    if len(gt_mask.shape) == 2:
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    if len(pred_mask.shape) == 2:
        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
    
    # Get target size (use original image size as reference)
    target_h, target_w = orig_img.shape[:2]
    
    # Resize all images to same size
    gt_mask_resized = cv2.resize(gt_mask, (target_w, target_h))
    flow_img_resized = cv2.resize(flow_img, (target_w, target_h))
    pred_mask_resized = cv2.resize(pred_mask, (target_w, target_h))
    
    # Add labels to each image - scaled for 1920x1080 images
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5  # Larger font for 1080p images
    font_thickness = 5  # Thicker text
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    padding = 20  # Padding around text
    
    def add_label(img, text, position='top'):
        img_copy = img.copy()
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        
        if position == 'top':
            text_x = padding
            text_y = text_size[1] + padding * 2
        else:
            text_x = padding
            text_y = img.shape[0] - padding
        
        # Add background rectangle for better readability
        cv2.rectangle(img_copy, 
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     bg_color, -1)
        cv2.putText(img_copy, text, (text_x, text_y), 
                   font, font_scale, text_color, font_thickness)
        return img_copy
    
    # Add labels
    orig_labeled = add_label(orig_img, "Original Image")
    gt_labeled = add_label(gt_mask_resized, "Ground Truth")
    flow_labeled = add_label(flow_img_resized, "Optical Flow")
    pred_labeled = add_label(pred_mask_resized, "Predicted Mask")
    
    # Create top and bottom rows
    top_row = np.hstack([orig_labeled, gt_labeled])
    bottom_row = np.hstack([flow_labeled, pred_labeled])
    
    # Stack rows to create 2x2 grid
    grid = np.vstack([top_row, bottom_row])
    
    # Add IoU text at the bottom center - larger for visibility
    if iou_value is not None:
        iou_font_scale = 3.0  # Much larger font for IoU
        iou_font_thickness = 6
        iou_text = f"IoU: {iou_value:.4f} ({iou_value*100:.2f}%)"
        text_size = cv2.getTextSize(iou_text, font, iou_font_scale, iou_font_thickness)[0]
        text_x = (grid.shape[1] - text_size[0]) // 2
        text_y = grid.shape[0] - 50  # More space from bottom
        
        # Add background with more padding
        bg_padding = 30
        cv2.rectangle(grid,
                     (text_x - bg_padding, text_y - text_size[1] - bg_padding),
                     (text_x + text_size[0] + bg_padding, text_y + bg_padding),
                     (0, 0, 0), -1)
        cv2.putText(grid, iou_text, (text_x, text_y),
                   font, iou_font_scale, (0, 255, 0), iou_font_thickness)
    
    return grid


def calculate_iou(pred_mask, gt_mask, use_center_only=True, center_width_percent=0.7):
    """
    Calculate Intersection over Union (IoU) between predicted and ground truth masks.
    
    Parameters:
    - pred_mask: Binary mask (0 or 255)
    - gt_mask: Binary mask (0 or 255)
    - use_center_only: If True, calculate IoU only on center region (default: True)
    - center_width_percent: Percentage of width to use from center (default: 0.6 = 60%)
    
    Returns:
    - iou: Float value between 0 and 1
    """
    # Convert to binary (0 and 1)
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    
    # Crop to center region if requested
    if use_center_only:
        height, width = pred_binary.shape
        
        # Calculate crop boundaries for center region
        center_width = int(width * center_width_percent)
        left_edge = (width - center_width) // 2
        right_edge = left_edge + center_width
        
        # Crop both masks to center region
        pred_binary = pred_binary[:, left_edge:right_edge]
        gt_binary = gt_binary[:, left_edge:right_edge]
        
        print(f"    [IoU calculated on center {center_width_percent*100:.0f}% width: columns {left_edge}-{right_edge} of {width}]")
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou


def process_folder_with_raft_and_segmentation():
    """
    Process all images in a folder with RAFT optical flow and segmentation.
    """
    # ============================================================
    # HARDCODED PATHS
    # ============================================================
    input_folder = "Data/Outputs4_brick/Overlays"           # Input images
    flow_folder = "Data/Outputs4_brick/Flow"                # Flow visualizations
    mask_folder = "Data/Outputs4_brick/Masks"               # Segmentation masks
    raft_model_path = "./RAFT/models/raft-things.pth"
    gt_mask_path = "Data/Outputs4_brick/GTMasks/Image0001.png"  # Ground truth mask
    # ============================================================
    
    # Create output directories
    os.makedirs(flow_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs("./output", exist_ok=True)  # RAFT default output
    
    # ============================================================
    # Load Ground Truth Mask (once, used for all comparisons)
    # ============================================================
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.resize(gt_mask, (1280, 720))  # Resize GT mask to 720p for consistency
    if gt_mask is None:
        print(f"ERROR: Could not load ground truth mask from: {gt_mask_path}")
        print("Please check the path and try again.")
        return
    else:
        print(f"✓ Loaded ground truth mask: {gt_mask_path}")
        print(f"  GT mask shape: {gt_mask.shape}")
    
    # Get all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    
    # Sort images by filename
    image_paths.sort()
    
    if len(image_paths) < 2:
        print(f"Error: Found only {len(image_paths)} image(s). Need at least 2.")
        return
    
    print(f"\n{'='*60}")
    print(f"Found {len(image_paths)} images in {input_folder}")
    print(f"Will generate {len(image_paths)-1} flow + mask outputs")
    print(f"{'='*60}\n")
    
    # Store IoU results for saving to file
    iou_results = []
    
    # Process consecutive pairs
    for i in range(len(image_paths) - 1):
        img_prev = image_paths[i]
        img_curr = image_paths[i + 1]
        
        print(f"\nProcessing pair {i+1}/{len(image_paths)-1}:")
        print(f"  {os.path.basename(img_prev)} → {os.path.basename(img_curr)}")
        
        # ============================================================
        # STEP 1: Run RAFT
        # ============================================================
        raft_segment.run_raft_on_pair(
            img_prev,
            img_curr,
            raft_model_path,
            i+1
        )
        
        # RAFT saves to ./output/{i+1:02d}.png
        flow_default = f"./output/{i+1:02d}.png"
        
        if not os.path.exists(flow_default):
            print(f"  ⚠ Warning: RAFT output not found: {flow_default}")
            iou_results.append((f"mask_{i+1:02d}.png", None, "RAFT output not found"))
            continue
        
        # Copy flow visualization to output folder
        flow_output = os.path.join(flow_folder, f"flow_{i+1:02d}.png")
        import shutil
        shutil.copy(flow_default, flow_output)
        print(f"  ✓ Saved flow: {flow_output}")
        
        # ============================================================
        # STEP 2: Load flow image
        # ============================================================
        flow_img = cv2.imread(flow_default, cv2.IMREAD_COLOR)
        flow_img = cv2.resize(flow_img, (1280, 720))  # Resize flow to 720p for consistency
        
        if flow_img is None:
            print(f"  ⚠ Warning: Could not load flow image")
            iou_results.append((f"mask_{i+1:02d}.png", None, "Could not load flow"))
            continue
        
        # ============================================================
        # STEP 3: Run segmentation
        # ============================================================
        mask_main, current_center = find_largest_gap_from_flow(flow_img)
        
        # ============================================================
        # STEP 4: Save mask
        # ============================================================
        if mask_main is not None:
            mask_path = os.path.join(mask_folder, f"mask_{i+1:02d}.png")
            cv2.imwrite(mask_path, mask_main)
            print(f"  ✓ Saved mask: {mask_path}")
            
            if current_center is not None:
                print(f"  ✓ Gap center: ({current_center[0]:.1f}, {current_center[1]:.1f})")
            
            # ============================================================
            # STEP 5: Calculate IoU with Ground Truth
            # ============================================================
            # Resize predicted mask to match GT size if needed
            if mask_main.shape != gt_mask.shape:
                mask_main_resized = cv2.resize(mask_main, (gt_mask.shape[1], gt_mask.shape[0]))
            else:
                mask_main_resized = mask_main
            
            # IMPORTANT: Invert predicted mask to match GT convention
            # Your mask: black (0) = gap, white (255) = background
            # GT mask: white (255) = gap, black (0) = background
            mask_main_aligned = mask_main_resized
            
            # Calculate IoU
            iou = calculate_iou(mask_main_aligned, gt_mask)
            print(f"  ✓ IoU: {iou:.4f} ({iou*100:.2f}%)")
            
            # Store result
            iou_results.append((f"mask_{i+1:02d}.png", iou, "Success"))
            
            # ============================================================
            # STEP 6: Create 2x2 Grid Visualization
            # ============================================================
            try:
                grid_img = create_2x2_grid(
                    orig_img_path=img_curr,  # Use the current/second image
                    gt_mask_path=gt_mask_path,
                    flow_img_path=flow_output,
                    pred_mask_path=mask_path,
                    iou_value=iou
                )
                
                # Save grid visualization
                grid_path = os.path.join(mask_folder, f"grid_{i+1:02d}.png")
                cv2.imwrite(grid_path, grid_img)
                print(f"  ✓ Saved 2x2 grid: {grid_path}")
                
            except Exception as e:
                print(f"  ⚠ Warning: Could not create grid visualization: {e}")
            
        else:
            print(f"  ⚠ Warning: No mask detected")
            iou_results.append((f"mask_{i+1:02d}.png", None, "No mask detected"))
    
    # ============================================================
    # STEP 7: Save IoU results to text file
    # ============================================================
    iou_file_path = os.path.join(mask_folder, "iou_results.txt")
    with open(iou_file_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("IoU Results - Comparison with Ground Truth\n")
        f.write(f"Ground Truth: {gt_mask_path}\n")
        f.write("="*60 + "\n\n")
        
        valid_ious = []
        for mask_name, iou, status in iou_results:
            if iou is not None:
                f.write(f"{mask_name}: IoU = {iou:.4f} ({iou*100:.2f}%) - {status}\n")
                valid_ious.append(iou)
            else:
                f.write(f"{mask_name}: IoU = N/A - {status}\n")
        
        f.write("\n" + "="*60 + "\n")
        if valid_ious:
            avg_iou = np.mean(valid_ious)
            min_iou = np.min(valid_ious)
            max_iou = np.max(valid_ious)
            f.write(f"Average IoU: {avg_iou:.4f} ({avg_iou*100:.2f}%)\n")
            f.write(f"Min IoU:     {min_iou:.4f} ({min_iou*100:.2f}%)\n")
            f.write(f"Max IoU:     {max_iou:.4f} ({max_iou*100:.2f}%)\n")
            f.write(f"Total masks evaluated: {len(valid_ious)}\n")
        else:
            f.write("No valid IoU values computed.\n")
        f.write("="*60 + "\n")
    
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Flow outputs: {flow_folder}")
    print(f"  Mask outputs: {mask_folder}")
    print(f"  Grid visualizations: {mask_folder}/grid_*.png")
    print(f"  IoU results saved to: {iou_file_path}")
    print(f"{'='*60}\n")
    
    # Print summary
    if iou_results:
        print("IoU Summary:")
        for mask_name, iou, status in iou_results:
            if iou is not None:
                print(f"  {mask_name}: {iou:.4f} ({iou*100:.2f}%)")
            else:
                print(f"  {mask_name}: N/A ({status})")


if __name__ == "__main__":
    process_folder_with_raft_and_segmentation()