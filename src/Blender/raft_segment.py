import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
# Absolute path to the folder that contains this file (Code/)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to RAFT/core (where raft.py and utils/ live)
RAFT_CORE_DIR = os.path.join(THIS_DIR, "RAFT", "core")

# Add RAFT/core to Python path
sys.path.append(RAFT_CORE_DIR)
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'


def load_image(imfile, max_size=None):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    
    # Check if grayscale and convert to RGB
    if len(img.shape) == 2:
        # Grayscale image (H, W) -> convert to RGB (H, W, 3)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        # RGBA image -> convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Add downsampling
    if max_size is not None:
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
    
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


# ---------- VISUALIZATION WITH BBOX ----------

def viz(img_tensor, flow_tensor, index, output_dir="output"):
    """
    img_tensor: (1,3,H,W) torch
    flow_tensor: (1,2,H,W) torch
    """
    os.makedirs(output_dir, exist_ok=True)

    # to numpy
    flo = flow_tensor[0].permute(1, 2, 0).cpu().numpy()

    # flow visualization (RGB uint8)
    flow_rgb = flow_viz.flow_to_image(flo)

    # flow_bgr = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR)

    out_path = os.path.join(output_dir, f"{index:02d}.png")
    cv2.imwrite(out_path, flow_rgb)
    print("Saved:", out_path)



def load_latest_raft_output(output_dir="./output"):
    pngs = sorted(glob.glob(os.path.join(output_dir, "*.png")))
    if not pngs:
        print(f"[load_latest_raft_output] No PNGs found in {output_dir}")
        return None, None
    latest = pngs[-1]
    img = cv2.imread(latest)
    if img is None:
        print(f"[load_latest_raft_output] Failed to read {latest}")
        return None, None
    print(f"[load_latest_raft_output] Using RAFT output: {latest}")
    return img, latest

def run_raft_on_pair(img1_path, img2_path, raft_model_path,ind):
    """
    Run RAFT ONLY on two specific images you provide.

    img1_path → older frame
    img2_path → newer frame
    """

    print(f"\nRunning RAFT on:\n  {img1_path}\n  {img2_path}")

    # Create temporary folder
    os.makedirs("./raft_tmp", exist_ok=True)
    os.system("rm -f ./raft_tmp/*")   # clean before reuse

    # Copy images to ordered names required by RAFT
    os.system(f"cp {img1_path} ./raft_tmp/0.png")
    os.system(f"cp {img2_path} ./raft_tmp/1.png")

    # Build RAFT arguments
    args = argparse.Namespace(
        model=raft_model_path,
        path="./raft_tmp/",     # only these 2 images
        small=False,
        mixed_precision=False,
        alternate_corr=False
    )

    # RAFT demo call
    demo(args,ind)

   

# ---------- RAFT DEMO LOOP ----------

def demo(args,ind):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        images = sorted(images)

        for i, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            torch.cuda.empty_cache()

            image1 = load_image(imfile1, max_size= None)
            image2 = load_image(imfile2, max_size= None)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up, index=ind)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true',
                        help='use efficient correlation implementation')
    args = parser.parse_args()

    demo(args)
