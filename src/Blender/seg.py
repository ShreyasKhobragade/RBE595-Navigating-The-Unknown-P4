import cv2
import numpy as np

def find_largest_gap_from_flow(
    flow_bgr,
    blur_ksize=5,
    canny_low=10,
    canny_high=40,
    min_area_ratio=0.001,
    max_area_ratio=0.9,
):
    """
    Given a RAFT / flow visualization image (BGR),
    returns (mask_main, center):
      - mask_main: uint8, same HxW as input, BLACK (0) for gap, WHITE (255) for background
                   largest closed blob region, or None if none found
      - center:    (cx, cy) float coordinates of blob centroid (image coords),
                   or None if no valid blob found
    """
    # 1) grayscale + blur + Canny
    gray = cv2.cvtColor(flow_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(gray_blur, canny_low, canny_high)
    cv2.imwrite("debug_edges.png", edges)
    
    h, w = edges.shape[:2]
    img_area = h * w
    
    # 2) Binarize & thicken/close edges
    edge_bin = (edges > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edge_bin, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 3) Find closed contours
    contours, hierarchy = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"Found {len(contours)} contours")
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * img_area:
            continue
        if area > max_area_ratio * img_area:
            continue
        valid_contours.append(cnt)
    print(f"Found {len(valid_contours)} valid contours")
    
    if not valid_contours:
        # No usable blob
        return None, None
    
    # 4) Largest contour
    largest = max(valid_contours, key=cv2.contourArea)
    
    # 5) Create mask for largest blob (initially white on black)
    mask_main = np.zeros_like(edges, dtype=np.uint8)
    cv2.drawContours(mask_main, [largest], -1, 255, thickness=-1)
    
    # 6) INVERT: Make gap BLACK (0) and background WHITE (255)
    mask_main = cv2.bitwise_not(mask_main)
    
    # 7) Compute centroid
    M = cv2.moments(largest)
    if M["m00"] == 0:
        # degenerate contour, treat as no blob
        return None, None
    
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    center = (cx, cy)
    
    return mask_main, center