#!/usr/bin/env python3
"""
task_2_code.py

Detect missing objects between BEFORE (X.jpg) and AFTER (X~2.jpg) images.
Saves annotated after image as X~3.jpg and mask as X~3_mask.jpg

Usage:
  python task_2_code.py --input "./input-images" --output "./task_2_output"
  python task_2_code.py --input "./input-images" --output "./task_2_output" --min-area 0.000075 --close-factor 0.008

"""

import os, sys, argparse, json
import cv2
import numpy as np
from pathlib import Path

def find_pairs(input_folder):
    # BEFORE: X.jpg ; AFTER: X~2.jpg -> output X~3.jpg
    files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg','.jpeg','.png')))
    before = {}
    after = {}
    for f in files:
        if f.endswith('~2.jpg') or f.endswith('~2.JPG') or f.endswith('~2.jpeg') or f.endswith('~2.PNG'):
            key = f.rsplit('~2',1)[0]
            after[key] = f
        else:
            key = f
            before[key] = f
    pairs = []
    for key, bf in before.items():
        # match when after variant exists for same base name without extension
        base = os.path.splitext(key)[0]
        # build expected after names: base~2.jpg (could also keep original extension)
        expected_after = base + "~2.jpg"
        expected_after_caps = base + "~2.JPG"
        if expected_after in files:
            pairs.append((bf, expected_after))
        elif expected_after_caps in files:
            pairs.append((bf, expected_after_caps))
        else:
            # also try same filename but with ~2 inserted before extension in other case
            found = None
            for cand in after.values():
                if os.path.splitext(cand)[0].startswith(base):
                    found = cand
                    break
            if found:
                pairs.append((bf, found))
    return pairs

def auto_detect_missing(before_img, after_img, min_area_frac=0.0001, close_factor=0.006,
                        merge_kernel_factor=0.01, use_otsu=True, debug=False):
    """
    Returns mask (uint8 0/255) and list of bounding boxes (x,y,w,h).
    min_area_frac = component area / (W*H) threshold
    close_factor = morphological closing kernel size factor of max(W,H)
    merge_kernel_factor = dilation kernel to merge islands (factor of max(W,H))
    """

    h, w = after_img.shape[:2]
    area = float(w * h)

    # 1) convert to grayscale and normalize
    b_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
    a_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)

    # 2) strong blur to suppress texture-noise (tuneable)
    blur_size = max(3, int(round(max(w,h) * 0.002)))  # small blur but scales with size
    if blur_size % 2 == 0: blur_size += 1
    b_blur = cv2.GaussianBlur(b_gray, (blur_size, blur_size), 0)
    a_blur = cv2.GaussianBlur(a_gray, (blur_size, blur_size), 0)

    # 3) absolute difference and enhance edges
    diff = cv2.absdiff(b_blur, a_blur)

    # 4) optional contrast stretching
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # 5) threshold: either Otsu or a fixed scaled threshold
    if use_otsu:
        _, th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # adaptive fixed thresh from mean
        th_val = max(10, int(np.mean(diff) * 1.0))
        _, th = cv2.threshold(diff, th_val, 255, cv2.THRESH_BINARY)

    # 6) morphological operations to remove noise and merge holes
    maxdim = max(w,h)
    # closing to fill holes inside missing objects (helps produce one blob)
    close_k = max(3, int(round(maxdim * close_factor)))
    if close_k % 2 == 0: close_k += 1
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    th_closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close)

    # small opening to remove very tiny speckles
    open_k = max(3, int(round(maxdim * 0.001)))
    if open_k % 2 == 0: open_k += 1
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    th_open = cv2.morphologyEx(th_closed, cv2.MORPH_OPEN, kernel_open)

    # dilation to merge nearby fragments (then we will erode back)
    merge_k = max(3, int(round(maxdim * merge_kernel_factor)))
    kernel_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_k, merge_k))
    th_merge = cv2.dilate(th_open, kernel_merge, iterations=1)
    # optional slight erosion to reduce oversize
    th_merge = cv2.erode(th_merge, kernel_open, iterations=1)

    # 7) connected components and filter by area
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(th_merge, connectivity=8)
    mask = np.zeros_like(th_merge)
    boxes = []
    min_area_px = max(1, int(min_area_frac * area))
    for i in range(1, nb_components):
        x, y, w_c, h_c, comp_area = stats[i]
        if comp_area >= min_area_px:
            # write to mask
            mask[labels == i] = 255
            boxes.append((x, y, w_c, h_c, comp_area))

    # 8) final cleanup: fill small holes inside kept components
    final_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, final_kernel)

    # 9) remove tiny islands created by dilation/erosion (final pass)
    nb2, labels2, stats2, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pure_boxes = []
    for i in range(1, nb2):
        x,y,w_c,h_c,comp_area = stats2[i]
        if comp_area >= min_area_px:
            pure_boxes.append((x,y,w_c,h_c,comp_area))

    # produce simplified mask and boxes
    mask_final = np.zeros_like(mask)
    for (x,y,w_c,h_c,comp_area) in pure_boxes:
        sub = (labels2 == list(range(nb2))[0])  # dummy to avoid unreferenced
        mask_final[labels2 == labels2[y + h_c//2, x + w_c//2]] = 255  # ensure region copied

    # safer approach: re-extract contours from mask and compute bounding boxes
    contours, _ = cv2.findContours((mask > 0).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_boxes = []
    final_mask = np.zeros_like(mask)
    for cnt in contours:
        area_cnt = cv2.contourArea(cnt)
        if area_cnt < min_area_px: 
            continue
        x,y,w_c,h_c = cv2.boundingRect(cnt)
        # convex hull to reduce many inner holes
        hull = cv2.convexHull(cnt)
        cv2.drawContours(final_mask, [hull], -1, 255, -1)
        final_boxes.append((x,y,w_c,h_c, area_cnt))

    if debug:
        return final_mask, final_boxes, {
            'diff_mean': float(np.mean(diff)),
            'nb_components': int(nb_components),
            'nb_final': len(final_boxes),
            'min_area_px': int(min_area_px),
            'close_k': int(close_k),
            'merge_k': int(merge_k)
        }

    return final_mask, final_boxes

def process_all(input_folder, output_folder, min_area_frac, close_factor, merge_kernel_factor, use_otsu, debug):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(input_folder)
    print(f"Found {len(pairs)} pairs to process.")
    summary = []
    for before_fn, after_fn in pairs:
        base_name = os.path.splitext(before_fn)[0]
        print("Processing:", base_name, "->", base_name + "~3.jpg")
        before_img = cv2.imread(os.path.join(input_folder, before_fn))
        after_img = cv2.imread(os.path.join(input_folder, after_fn))
        if before_img is None or after_img is None:
            print("  ERROR reading pair; skipping")
            continue
        mask, boxes = auto_detect_missing(before_img, after_img,
                                          min_area_frac=min_area_frac,
                                          close_factor=close_factor,
                                          merge_kernel_factor=merge_kernel_factor,
                                          use_otsu=use_otsu,
                                          debug=False)
        # annotate on a copy of the after image (do not alter original colors)
        annotated = after_img.copy()
        for (x,y,w_c,h_c,area) in boxes:
            cv2.rectangle(annotated, (x,y), (x+w_c, y+h_c), (0,0,255), 2)  # red box
        # Save outputs:
        out_img_name = os.path.join(output_folder, base_name + "~3.jpg")
        out_mask_name = os.path.join(output_folder, base_name + "~3_mask.jpg")
        cv2.imwrite(out_img_name, annotated)
        cv2.imwrite(out_mask_name, mask)
        print(f"  result: ok regions: {len(boxes)}")
        summary.append({'base': base_name, 'regions': len(boxes)})
    # Save a summary json
    summary_path = os.path.join(output_folder, "task2_detections.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Wrote {len(summary)} outputs to {output_folder}")
    print("Summary saved to:", summary_path)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Input folder containing BEFORE (X.jpg) and AFTER (X~2.jpg)')
    p.add_argument('--output', required=True, help='Output folder to write X~3.jpg and X~3_mask.jpg')
    p.add_argument('--min-area', type=float, default=0.0001,
                   help='Minimum region area as fraction of image (e.g. 0.0001). Lower -> detect smaller items.')
    p.add_argument('--close-factor', type=float, default=0.006,
                   help='Morphological closing kernel factor of max(width,height). Increase to merge fragmented regions.')
    p.add_argument('--merge-kernel-factor', type=float, default=0.01,
                   help='Dilation kernel factor to merge near fragments. Increase to merge more.')
    p.add_argument('--use-otsu', action='store_true', help='Use Otsu threshold instead of fixed scaled threshold')
    p.add_argument('--debug', action='store_true', help='Print debug info for each image')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_all(args.input, args.output,
                min_area_frac=args.min_area,
                close_factor=args.close_factor,
                merge_kernel_factor=args.merge_kernel_factor,
                use_otsu=args.use_otsu,
                debug=args.debug)