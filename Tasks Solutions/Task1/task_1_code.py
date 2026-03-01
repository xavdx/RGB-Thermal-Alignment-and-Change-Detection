#!/usr/bin/env python3
"""
task_1_code.py

Robust RGB-Thermal alignment pipeline for Task 1.

Usage examples:
  python task_1_code.py --input "./input-images" --output "./task_1_output"
  python task_1_code.py --output "./task_1_output" --list-flagged
  python task_1_code.py --output "./task_1_output" --cleanup
"""

import os
import re
import json
import cv2
import argparse
import numpy as np
from pathlib import Path

#Defaults
DEFAULT_INPUT="./"               #safe placeholder (can be overriden)
DEFAULT_OUTPUT="./task_1_output"
MIN_INLIERS_FOR_H=12
MAX_CONDITION_NUMBER=1e8
MIN_NONBLACK_RATIO=0.02

def extract_timestamp_and_seq(filename):
    m=re.search(r'DJI_(\d+)_(\d{4})_[TZ]\.JPG$', filename, re.IGNORECASE)
    return (m.group(1), m.group(2)) if m else (None, None)

def cond_number(H):
    try:
        s =np.linalg.svd(H)[1]
        return float(s[0]/s[-1])
    except Exception:
        return float('inf')

def nonblack_ratio(img):
    if img is None:
        return 0.0
    if len(img.shape)==3:
        nonblack=np.count_nonzero(np.any(img != 0, axis=2))
    else:
        nonblack=np.count_nonzero(img!=0)
    return nonblack/float(img.shape[0] * img.shape[1])

def find_pairs(input_folder):
    files=sorted([f for f in os.listdir(input_folder) if f.upper().endswith('.JPG')])
    thermal_files=[]
    rgb_files=[]
    for f in files:
        t,s=extract_timestamp_and_seq(f)
        if t and s:
            if f.upper().endswith('_T.JPG'):
                thermal_files.append({'file': f, 'ts': t, 'seq': s})
            elif f.upper().endswith('_Z.JPG'):
                rgb_files.append({'file': f, 'ts': t, 'seq': s})
    rgb_map={}
    for r in rgb_files:
        rgb_map.setdefault(r['seq'], []).append(r)
    pairs=[]
    for t in thermal_files:
        seq=t['seq']
        cands=rgb_map.get(seq, [])
        if not cands:
            continue
        best=min(cands, key=lambda r: abs(int(r['ts']) - int(t['ts'])))
        pairs.append((t['file'], best['file']))
    return pairs

def try_auto_align(th,rgb,scale_limit=8.0):
    h_rgb,w_rgb=rgb.shape[:2]
    h_th,w_th=th.shape[:2]
    scale=max(w_rgb/w_th,h_rgb/h_th)
    scale=min(scale,scale_limit)
    th_big=cv2.resize(th,(int(w_th * scale), int(h_th * scale)), interpolation=cv2.INTER_LINEAR)

    gray_t=cv2.cvtColor(th_big, cv2.COLOR_BGR2GRAY)
    gray_z=cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_t_e=clahe.apply(gray_t)
    gray_z_e=clahe.apply(gray_z)
    #Try SIFT if available
    if hasattr(cv2,'SIFT_create'):
        sift=cv2.SIFT_create(nfeatures=4000)
        kp1,des1=sift.detectAndCompute(gray_t_e, None)
        kp2,des2=sift.detectAndCompute(gray_z_e, None)
        if des1 is not None and des2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
            FLANN_INDEX_KDTREE=1
            index_params=dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params=dict(checks=200)
            flann=cv2.FlannBasedMatcher(index_params, search_params)
            matches=flann.knnMatch(des1, des2, k=2)
            good=[]
            for m_n in matches:
                if len(m_n)==2:
                    m,n=m_n
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
            n_good=len(good)
            if n_good>=10:
                src_pts=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst_pts=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                H,mask=cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, maxIters=5000)
                inliers=int(np.sum(mask)) if mask is not None else 0
                cnd=cond_number(H) if H is not None else float('inf')
                if H is not None and inliers >= MIN_INLIERS_FOR_H and cnd < MAX_CONDITION_NUMBER:
                    S_pre=np.array([[scale,0,0],[0,scale,0],[0,0,1]], dtype=np.float64)
                    H_adj=H @ S_pre
                    warped=cv2.warpPerspective(th, H_adj, (w_rgb, h_rgb),
                                                 flags=cv2.INTER_LINEAR,
                                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
                    nb_ratio = nonblack_ratio(warped)
                    return (True, warped, {'matches': n_good, 'inliers': inliers, 'cond': cnd, 'nb_ratio': nb_ratio}) \
                           if nb_ratio >= MIN_NONBLACK_RATIO else (False, None, {'matches': n_good, 'inliers': inliers, 'cond': cnd, 'nb_ratio': nb_ratio})
                else:
                    return False, None, {'matches': n_good, 'inliers': inliers, 'cond': cnd}
            else:
                return False, None, {'matches': n_good}
    #fallback
    return False, None, {}

def process_all(input_folder,output_folder):
    Path(output_folder).mkdir(parents=True,exist_ok=True)
    pairs=find_pairs(input_folder)
    print(f"Found {len(pairs)} matched pairs.")
    diagnostics=[]
    for tfile,zfile in pairs:
        base=zfile[:-6]
        print(f"\nProcessing: {base}")
        tpath=os.path.join(input_folder, tfile)
        zpath=os.path.join(input_folder, zfile)
        rgb=cv2.imread(zpath)
        th=cv2.imread(tpath, cv2.IMREAD_COLOR)
        if rgb is None or th is None:
            print("ERROR reading files; skipping")
            diagnostics.append({'base': base,'error': 'read_failed'})
            continue
        ok,warped,info=try_auto_align(th,rgb)
        if ok:
            print("Auto-align accepted:",info)
            cv2.imwrite(os.path.join(output_folder, f"{base}_Z.JPG"), rgb)
            cv2.imwrite(os.path.join(output_folder, f"{base}_AT.JPG"), warped)
            entry={'base': base, 'diag': {'reason': 'ok_auto', **info}}
            diagnostics.append(entry)
        else:
            print("Auto alignment rejected or not available; using safe scaled fallback.")
            th_scaled=cv2.resize(th, (int(rgb.shape[1]), int(rgb.shape[0])), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(output_folder, f"{base}_Z.JPG"), rgb)
            cv2.imwrite(os.path.join(output_folder, f"{base}_AT.JPG"), th_scaled)
            entry={'base': base, 'diag': {'reason': 'fallback_scaled', **info}}
            diagnostics.append(entry)
    diag_path=os.path.join(output_folder, "diagnostics.json")
    with open(diag_path, "w") as wf:
        json.dump(diagnostics, wf, indent=2)
    print("\nProcessing complete. Wrote diagnostics.json and saved outputs to:", output_folder)

def cleanup_outputs(output_folder):
    files=[f for f in os.listdir(output_folder) if f.upper().endswith('.JPG')]
    groups={}
    for f in files:
        base=None
        name=f
        for suf in ["_Z.JPG", "_AT.JPG", "_overlay.JPG", "_matches_debug.JPG", "_debug.JPG"]:
            if name.endswith(suf):
                base=name[:-len(suf)]
                break
        if base is None:
            base=Path(name).stem
        groups.setdefault(base,[]).append(f)
    for base, flist in groups.items():
        z_candidates=[f for f in flist if f.endswith("_Z.JPG")]
        a_candidates=[f for f in flist if f.endswith("_AT.JPG")]
        def keep_newest(cands):
            if not cands:
                return None
            fullpaths=[os.path.join(output_folder, f) for f in cands]
            newest=max(fullpaths, key=os.path.getmtime)
            for p in fullpaths:
                if p!=newest:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            return os.path.basename(newest)
        keep_newest(z_candidates)
        keep_newest(a_candidates)
    print("Cleanup complete.")

def list_flagged(output_folder):
    diag_path=os.path.join(output_folder, "diagnostics.json")
    if not os.path.exists(diag_path):
        print("diagnostics.json not found:", diag_path)
        return
    with open(diag_path, "r") as f:
        diagnostics=json.load(f)
    flagged=[]
    for d in diagnostics:
        base=d.get("base")
        reason=d.get("diag", {}).get("reason", "")
        inliers=d.get("diag", {}).get("inliers", 0) or 0
        if reason not in ("ok_auto",) or inliers < MIN_INLIERS_FOR_H:
            flagged.append({"base": base, "reason": reason, "inliers": inliers, "debug": d.get("diag", {})})
    print("Flagged pairs to manually correct (base, reason, inliers):")
    for f in flagged:
        print(f" - {f['base']}  reason={f['reason']}  inliers={f['inliers']}  debug={f['debug']}")
    print("\nTotal flagged:", len(flagged))
    with open(os.path.join(output_folder, "flagged_pairs.txt"), "w") as wf:
        for f in flagged:
            wf.write(f"{f['base']}\t{f['reason']}\t{f['inliers']}\n")
    print("Wrote flagged_pairs.txt to output folder.")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--input', default=DEFAULT_INPUT, help='Input images folder (contains _T and _Z files)')
    p.add_argument('--output', default=DEFAULT_OUTPUT, help='Output folder to save _Z and _AT files')
    p.add_argument('--cleanup', action='store_true', help='Cleanup duplicate outputs (keep newest)')
    p.add_argument('--list-flagged', action='store_true', help='List flagged pairs from diagnostics.json')
    return p.parse_args()

def main():
    args=parse_args()
    if not os.path.exists(args.input):
        print(f"Input folder not found: {args.input}")
        print("If you are submitting without input-images (per instructions), this is expected.")
        print("The grader can run your script with --input pointing to their dataset.")
    process_all(args.input, args.output)
    if args.cleanup:
        cleanup_outputs(args.output)
    if args.list_flagged:
        list_flagged(args.output)
if __name__=="__main__":
    main()