import cv2
import numpy as np
import os
from pathlib import Path

def create_triple_side_by_side_video(
    images_folder="./Images",
    flow_folder="./Flow",
    output_folder="./output",
    output_path="./output_video.mp4",
    fps=10,
    slow_motion_factor=8,
    resize_height=None,
    images_pattern="step_",
    flow_pattern="mask_",
    output_pattern=""
):
    """
    Create a 3-window side-by-side video from three folders of images with slow motion.
    
    Parameters:
    - images_folder: Path to folder with step_{:02d}.png images
    - flow_folder: Path to folder with mask_{:02d}.png images
    - output_folder: Path to folder with output images
    - output_path: Output video file path
    - fps: Frames per second for output video
    - slow_motion_factor: How many times to repeat each frame (1=normal, 2=2x slower, 3=3x slower, etc.)
    - resize_height: If specified, resize all images to this height (maintains aspect ratio)
    - images_pattern: Prefix pattern for images folder files
    - flow_pattern: Prefix pattern for flow folder files
    - output_pattern: Prefix pattern for output folder files
    """
    
    # Get list of images from all three folders
    images_files = sorted([f for f in os.listdir(images_folder) 
                          if f.startswith(images_pattern) and f.endswith('.png')])
    flow_files = sorted([f for f in os.listdir(flow_folder) 
                        if f.startswith(flow_pattern) and f.endswith('.png')])
    output_files = sorted([f for f in os.listdir(output_folder) 
                          if f.endswith('.png') and (not output_pattern or f.startswith(output_pattern))])
    
    print(f"Found {len(images_files)} images in {images_folder}")
    print(f"Found {len(flow_files)} masks in {flow_folder}")
    print(f"Found {len(output_files)} outputs in {output_folder}")
    print(f"\nFiles found:")
    print(f"  Images: {images_files[:5]}{'...' if len(images_files) > 5 else ''}")
    print(f"  Flow: {flow_files[:5]}{'...' if len(flow_files) > 5 else ''}")
    print(f"  Output: {output_files[:5]}{'...' if len(output_files) > 5 else ''}")
    
    if len(images_files) == 0 or len(flow_files) == 0 or len(output_files) == 0:
        print("Error: No images found in one or more folders!")
        return
    
    # Use the MAXIMUM number of frames to ensure we don't skip any
    num_frames = max(len(images_files), len(flow_files), len(output_files))
    total_output_frames = num_frames * slow_motion_factor
    print(f"\nCreating video with {num_frames} frames (using max across all folders)")
    print(f"Slow motion factor: {slow_motion_factor}x")
    print(f"Total output frames: {total_output_frames}")
    
    # Pad the shorter lists with None to handle missing frames
    while len(images_files) < num_frames:
        images_files.append(None)
    while len(flow_files) < num_frames:
        flow_files.append(None)
    while len(output_files) < num_frames:
        output_files.append(None)
    
    # Read first valid set to get dimensions
    first_img = None
    first_mask = None
    first_output = None
    
    for i in range(num_frames):
        if images_files[i] and first_img is None:
            first_img = cv2.imread(os.path.join(images_folder, images_files[i]))
        if flow_files[i] and first_mask is None:
            first_mask = cv2.imread(os.path.join(flow_folder, flow_files[i]))
        if output_files[i] and first_output is None:
            first_output = cv2.imread(os.path.join(output_folder, output_files[i]))
        if first_img is not None and first_mask is not None and first_output is not None:
            break
    
    if first_img is None or first_mask is None or first_output is None:
        print("Error: Could not read any valid frames from one or more folders!")
        return
    
    # Resize if requested
    if resize_height is not None:
        aspect_ratio_img = first_img.shape[1] / first_img.shape[0]
        aspect_ratio_mask = first_mask.shape[1] / first_mask.shape[0]
        aspect_ratio_output = first_output.shape[1] / first_output.shape[0]
        
        new_width_img = int(resize_height * aspect_ratio_img)
        new_width_mask = int(resize_height * aspect_ratio_mask)
        new_width_output = int(resize_height * aspect_ratio_output)
        
        first_img = cv2.resize(first_img, (new_width_img, resize_height))
        first_mask = cv2.resize(first_mask, (new_width_mask, resize_height))
        first_output = cv2.resize(first_output, (new_width_output, resize_height))
    
    # Make sure all images have same height
    h1, w1 = first_img.shape[:2]
    h2, w2 = first_mask.shape[:2]
    h3, w3 = first_output.shape[:2]
    
    # Find target height (use minimum or common height)
    target_height = min(h1, h2, h3)
    
    # Resize all to target height while maintaining aspect ratio
    aspect_ratio_img = w1 / h1
    aspect_ratio_mask = w2 / h2
    aspect_ratio_output = w3 / h3
    
    new_width_img = int(target_height * aspect_ratio_img)
    new_width_mask = int(target_height * aspect_ratio_mask)
    new_width_output = int(target_height * aspect_ratio_output)
    
    first_img = cv2.resize(first_img, (new_width_img, target_height))
    first_mask = cv2.resize(first_mask, (new_width_mask, target_height))
    first_output = cv2.resize(first_output, (new_width_output, target_height))
    
    h1, w1 = first_img.shape[:2]
    h2, w2 = first_mask.shape[:2]
    h3, w3 = first_output.shape[:2]
    
    # Create side-by-side frame dimensions (3 windows)
    frame_height = target_height
    frame_width = w1 + w2 + w3
    
    print(f"\nOutput video dimensions: {frame_width}x{frame_height}")
    print(f"Left image size: {w1}x{h1}")
    print(f"Middle image size: {w2}x{h2}")
    print(f"Right image size: {w3}x{h3}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("Error: Could not open video writer!")
        return
    
    # Create black placeholder frames
    black_img = np.zeros((h1, w1, 3), dtype=np.uint8)
    black_mask = np.zeros((h2, w2, 3), dtype=np.uint8)
    black_output = np.zeros((h3, w3, 3), dtype=np.uint8)
    
    # Keep track of last valid frames for fallback
    last_valid_img = first_img.copy()
    last_valid_mask = first_mask.copy()
    last_valid_output = first_output.copy()
    
    # Process each frame - NO SKIPPING
    frames_written = 0
    print("\nProcessing frames...")
    
    for i in range(num_frames):
        # Try to read images, use black or last valid frame if missing
        img = None
        mask = None
        output_img = None
        
        # Read original image
        if images_files[i] is not None:
            img_path = os.path.join(images_folder, images_files[i])
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (w1, h1))
                last_valid_img = img.copy()
            else:
                print(f"  Warning: Could not read {img_path}, using last valid frame")
                img = last_valid_img.copy()
        else:
            print(f"  Warning: No image file for frame {i}, using last valid frame")
            img = last_valid_img.copy()
        
        # Read flow mask
        if flow_files[i] is not None:
            mask_path = os.path.join(flow_folder, flow_files[i])
            mask = cv2.imread(mask_path)
            if mask is not None:
                mask = cv2.resize(mask, (w2, h2))
                last_valid_mask = mask.copy()
            else:
                print(f"  Warning: Could not read {mask_path}, using last valid frame")
                mask = last_valid_mask.copy()
        else:
            print(f"  Warning: No mask file for frame {i}, using last valid frame")
            mask = last_valid_mask.copy()
        
        # Read output image
        if output_files[i] is not None:
            output_img_path = os.path.join(output_folder, output_files[i])
            output_img = cv2.imread(output_img_path)
            if output_img is not None:
                output_img = cv2.resize(output_img, (w3, h3))
                last_valid_output = output_img.copy()
            else:
                print(f"  Warning: Could not read {output_img_path}, using last valid frame")
                output_img = last_valid_output.copy()
        else:
            print(f"  Warning: No output file for frame {i}, using last valid frame")
            output_img = last_valid_output.copy()
        
        # Create side-by-side frame (3 windows) - ALWAYS create, never skip
        triple_view = np.hstack([img, mask, output_img])
        
        # Add frame number text (top-left)
        cv2.putText(triple_view, f"Frame {i:02d}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add slow motion indicator
        cv2.putText(triple_view, f"{slow_motion_factor}x Slow Motion", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        # Add labels at the bottom of each window
        cv2.putText(triple_view, "Original", (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(triple_view, "Flow Mask", (w1 + 10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(triple_view, "Output", (w1 + w2 + 10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw vertical separators
        cv2.line(triple_view, (w1, 0), (w1, frame_height), (255, 255, 255), 2)
        cv2.line(triple_view, (w1 + w2, 0), (w1 + w2, frame_height), (255, 255, 255), 2)
        
        # Write frame multiple times for slow motion effect
        for repeat in range(slow_motion_factor):
            out.write(triple_view)
            frames_written += 1
        
        if (i + 1) % 5 == 0 or i == num_frames - 1:
            print(f"  Processed {i + 1}/{num_frames} unique frames ({frames_written} total frames written)")
    
    # Release video writer
    out.release()
    
    actual_duration = frames_written / fps
    print(f"\n✓ Video saved to: {output_path}")
    print(f"  Unique frames: {num_frames}")
    print(f"  Total frames written: {frames_written}")
    print(f"  Slow motion factor: {slow_motion_factor}x")
    print(f"  FPS: {fps}")
    print(f"  Duration: {actual_duration:.2f} seconds")
    print(f"  (Would be {num_frames/fps:.2f}s at normal speed)")
    print(f"\n✓ NO FRAMES WERE SKIPPED!")


if __name__ == "__main__":
    # Slow motion - 3x slower (default)
    create_triple_side_by_side_video()