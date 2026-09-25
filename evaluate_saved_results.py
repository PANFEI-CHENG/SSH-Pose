#!/usr/bin/env python3
"""Evaluate saved NOCS result PKLs, including a GT-R/t CATRE size metric."""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "utils"))

from CATRE_evaluation_utils import (
    compute_3d_iou_new,
    compute_3d_matches as compute_catre_3d_matches,
)
from evaluation_utils import (
    compute_ap_from_matches_scores,
    compute_independent_mAP,
)


SYNSET_NAMES = [
    "BG", "bottle", "bowl", "camera", "can", "laptop", "mug"]
DEFAULT_RESULTS_DIR = os.path.join(
    BASE_DIR, "log", "REAL275", "results")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved pose PKLs with the original NOCS protocol and "
            "CATRE corrected IoU using GT rotation/translation"))
    parser.add_argument(
        "results_dir", nargs="?", default=DEFAULT_RESULTS_DIR,
        help="Directory containing results_*.pkl files")
    parser.add_argument(
        "--report", default=None,
        help=(
            "Combined text report path; default: "
            "<results_dir>/evaluation_report_gt_rt_catre.txt"))
    parser.add_argument(
        "--assignment_iou", type=float, default=0.5,
        help=(
            "Minimum class-aware 2D box IoU for assigning a prediction to "
            "the GT whose R/t will be injected (default: 0.5)"))
    parser.add_argument(
        "--save_json", action="store_true",
        help="Also save aggregate original/CATRE arrays as JSON")
    return parser.parse_args()


def load_results(results_dir):
    result_paths = sorted(glob.glob(os.path.join(
        os.path.abspath(results_dir), "results_*.pkl")))
    if not result_paths:
        raise FileNotFoundError(
            "No results_*.pkl files found in " + os.path.abspath(results_dir))

    required = {
        "gt_class_ids", "gt_bboxes", "gt_RTs", "gt_scales",
        "pred_class_ids", "pred_bboxes", "pred_scores", "pred_RTs",
        "pred_scales",
    }
    results = []
    for result_path in result_paths:
        with open(result_path, "rb") as input_file:
            result = pickle.load(input_file)
        if not isinstance(result, dict):
            raise TypeError(
                "Expected a result dictionary in " + result_path)
        missing = sorted(required.difference(result))
        if missing:
            raise KeyError(
                "{} is missing fields: {}".format(
                    result_path, ", ".join(missing)))
        result = dict(result)
        result["pkl_path"] = result_path
        if "gt_handle_visibility" not in result:
            result["gt_handle_visibility"] = np.ones(
                len(result["gt_class_ids"]), dtype=np.float32)
        results.append(result)
    return results, result_paths


def bbox_iou_matrix(boxes_a, boxes_b):
    """Pairwise IoU for NOCS boxes in [y1, x1, y2, x2] order."""
    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    top = np.maximum(a[..., 0], b[..., 0])
    left = np.maximum(a[..., 1], b[..., 1])
    bottom = np.minimum(a[..., 2], b[..., 2])
    right = np.minimum(a[..., 3], b[..., 3])
    intersection = np.maximum(bottom - top, 0) * np.maximum(right - left, 0)
    area_a = np.maximum(a[..., 2] - a[..., 0], 0) * np.maximum(
        a[..., 3] - a[..., 1], 0)
    area_b = np.maximum(b[..., 2] - b[..., 0], 0) * np.maximum(
        b[..., 3] - b[..., 1], 0)
    return intersection / np.maximum(area_a + area_b - intersection, 1e-12)


def class_aware_2d_assignment(result, minimum_iou):
    """One-to-one detection association used only to select the GT R/t."""
    pred_class_ids = np.asarray(result["pred_class_ids"], dtype=np.int32)
    gt_class_ids = np.asarray(result["gt_class_ids"], dtype=np.int32)
    pred_bboxes = np.asarray(result["pred_bboxes"])
    gt_bboxes = np.asarray(result["gt_bboxes"])
    assignment = {}
    assignment_ious = {}
    for class_id in range(1, len(SYNSET_NAMES)):
        pred_indices = np.flatnonzero(pred_class_ids == class_id)
        gt_indices = np.flatnonzero(gt_class_ids == class_id)
        if len(pred_indices) == 0 or len(gt_indices) == 0:
            continue
        ious = bbox_iou_matrix(
            pred_bboxes[pred_indices], gt_bboxes[gt_indices])
        pred_rows, gt_columns = linear_sum_assignment(-ious)
        for pred_row, gt_column in zip(pred_rows, gt_columns):
            iou = float(ious[pred_row, gt_column])
            if iou >= minimum_iou:
                pred_index = int(pred_indices[pred_row])
                assignment[pred_index] = int(gt_indices[gt_column])
                assignment_ious[pred_index] = iou
    return assignment, assignment_ious


def gt_pose_with_predicted_scale(pred_rt, gt_rt):
    """Keep predicted isotropic scale while replacing rotation and t by GT."""
    pred_rt = np.asarray(pred_rt, dtype=np.float64)
    gt_rt = np.asarray(gt_rt, dtype=np.float64)
    pred_scale = float(np.cbrt(np.linalg.det(pred_rt[:3, :3])))
    gt_scale = float(np.cbrt(np.linalg.det(gt_rt[:3, :3])))
    if (not np.isfinite(pred_scale) or pred_scale <= 0
            or not np.isfinite(gt_scale) or gt_scale <= 0):
        return None
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = gt_rt[:3, :3] / gt_scale * pred_scale
    result[:3, 3] = gt_rt[:3, 3]
    return result


def compute_gt_rt_catre_iou(results, thresholds, assignment_iou):
    """CATRE-corrected IoU AP with fixed 2D association and GT R/t."""
    threshold_count = len(thresholds)
    class_pred_matches = [
        np.zeros((threshold_count, 0), dtype=np.float64)
        for _ in SYNSET_NAMES]
    class_pred_scores = [
        np.zeros((threshold_count, 0), dtype=np.float64)
        for _ in SYNSET_NAMES]
    class_gt_matches = [
        np.zeros((threshold_count, 0), dtype=np.float64)
        for _ in SYNSET_NAMES]
    assigned_total = 0
    prediction_total = 0
    gt_total = 0

    for result in results:
        gt_class_ids = np.asarray(result["gt_class_ids"], dtype=np.int32)
        pred_class_ids = np.asarray(result["pred_class_ids"], dtype=np.int32)
        gt_rts = np.asarray(result["gt_RTs"])
        pred_rts = np.asarray(result["pred_RTs"])
        gt_scales = np.asarray(result["gt_scales"])
        pred_scales = np.asarray(result["pred_scales"])
        pred_scores = np.asarray(result["pred_scores"], dtype=np.float64)
        handle_visibility = np.asarray(result["gt_handle_visibility"])
        assignment, _ = class_aware_2d_assignment(result, assignment_iou)
        assigned_total += len(assignment)
        prediction_total += len(pred_class_ids)
        gt_total += len(gt_class_ids)

        for class_id in range(1, len(SYNSET_NAMES)):
            pred_indices = np.flatnonzero(pred_class_ids == class_id)
            gt_indices = np.flatnonzero(gt_class_ids == class_id)
            local_gt_index = {
                int(global_index): local_index
                for local_index, global_index in enumerate(gt_indices)}
            pred_matches = np.full(
                (threshold_count, len(pred_indices)), -1.0,
                dtype=np.float64)
            gt_matches = np.full(
                (threshold_count, len(gt_indices)), -1.0,
                dtype=np.float64)

            for local_pred_index, pred_index in enumerate(pred_indices):
                pred_index = int(pred_index)
                if pred_index not in assignment:
                    continue
                gt_index = assignment[pred_index]
                if gt_index not in local_gt_index:
                    raise AssertionError("2D assignment changed object class")
                injected_rt = gt_pose_with_predicted_scale(
                    pred_rts[pred_index], gt_rts[gt_index])
                if injected_rt is None:
                    continue
                visibility = (handle_visibility[gt_index]
                              if class_id == 6 else 1)
                overlap = compute_3d_iou_new(
                    injected_rt, gt_rts[gt_index],
                    pred_scales[pred_index], gt_scales[gt_index],
                    visibility, SYNSET_NAMES[class_id],
                    SYNSET_NAMES[class_id])
                local_gt = local_gt_index[gt_index]
                for threshold_index, threshold in enumerate(thresholds):
                    if overlap > threshold:
                        pred_matches[
                            threshold_index, local_pred_index] = local_gt
                        gt_matches[threshold_index, local_gt] = local_pred_index

            scores = pred_scores[pred_indices]
            class_pred_matches[class_id] = np.concatenate(
                (class_pred_matches[class_id], pred_matches), axis=1)
            class_pred_scores[class_id] = np.concatenate(
                (class_pred_scores[class_id],
                 np.tile(scores, (threshold_count, 1))), axis=1)
            class_gt_matches[class_id] = np.concatenate(
                (class_gt_matches[class_id], gt_matches), axis=1)

    aps = np.zeros((len(SYNSET_NAMES) + 1, threshold_count), dtype=np.float64)
    for class_id in range(1, len(SYNSET_NAMES)):
        for threshold_index in range(threshold_count):
            aps[class_id, threshold_index] = compute_ap_from_matches_scores(
                class_pred_matches[class_id][threshold_index],
                class_pred_scores[class_id][threshold_index],
                class_gt_matches[class_id][threshold_index])
    aps[-1] = np.mean(aps[1:-1], axis=0)
    counts = {
        "images": len(results),
        "predictions": prediction_total,
        "ground_truth": gt_total,
        "assigned_pairs": assigned_total,
    }
    return aps, counts


def compute_standard_catre_iou(results, thresholds):
    """Official CATRE corrected-axis IoU AP with predicted R/t/size."""
    threshold_count = len(thresholds)
    class_pred_matches = [
        np.zeros((threshold_count, 0), dtype=np.float64)
        for _ in SYNSET_NAMES]
    class_pred_scores = [
        np.zeros((threshold_count, 0), dtype=np.float64)
        for _ in SYNSET_NAMES]
    class_gt_matches = [
        np.zeros((threshold_count, 0), dtype=np.float64)
        for _ in SYNSET_NAMES]

    for result in results:
        gt_class_ids = np.asarray(result["gt_class_ids"], dtype=np.int32)
        pred_class_ids = np.asarray(result["pred_class_ids"], dtype=np.int32)
        gt_rts = np.asarray(result["gt_RTs"])
        pred_rts = np.asarray(result["pred_RTs"])
        gt_scales = np.asarray(result["gt_scales"])
        pred_scales = np.asarray(result["pred_scales"])
        pred_scores = np.asarray(result["pred_scores"], dtype=np.float64)
        pred_bboxes = np.asarray(result["pred_bboxes"])
        handle_visibility = np.asarray(result["gt_handle_visibility"])

        for class_id in range(1, len(SYNSET_NAMES)):
            gt_selected = gt_class_ids == class_id
            pred_selected = pred_class_ids == class_id
            class_gt_ids = gt_class_ids[gt_selected]
            class_pred_ids = pred_class_ids[pred_selected]
            class_scores = pred_scores[pred_selected]
            class_visibility = (handle_visibility[gt_selected]
                                if class_id == 6
                                else np.ones(len(class_gt_ids)))
            gt_matches, pred_matches, _, sorted_indices = (
                compute_catre_3d_matches(
                    class_gt_ids,
                    gt_rts[gt_selected],
                    gt_scales[gt_selected],
                    class_visibility,
                    SYNSET_NAMES,
                    pred_bboxes[pred_selected],
                    class_pred_ids,
                    class_scores,
                    pred_rts[pred_selected],
                    pred_scales[pred_selected],
                    thresholds))
            if len(sorted_indices):
                class_scores = class_scores[sorted_indices]
            class_pred_matches[class_id] = np.concatenate(
                (class_pred_matches[class_id], pred_matches), axis=1)
            class_pred_scores[class_id] = np.concatenate(
                (class_pred_scores[class_id],
                 np.tile(class_scores, (threshold_count, 1))), axis=1)
            class_gt_matches[class_id] = np.concatenate(
                (class_gt_matches[class_id], gt_matches), axis=1)

    aps = np.zeros((len(SYNSET_NAMES) + 1, threshold_count), dtype=np.float64)
    for class_id in range(1, len(SYNSET_NAMES)):
        for threshold_index in range(threshold_count):
            aps[class_id, threshold_index] = compute_ap_from_matches_scores(
                class_pred_matches[class_id][threshold_index],
                class_pred_scores[class_id][threshold_index],
                class_gt_matches[class_id][threshold_index])
    aps[-1] = np.mean(aps[1:-1], axis=0)
    return aps


def format_catre_report(aps, thresholds, assignment_iou, counts):
    lines = [
        "=" * 80,
        "CATRE IoU mAP evaluation report with GT R/t (AP, %)",
        "Setting: GT rotation + GT translation + predicted size; "
        "class-aware 2D assignment IoU >= {:.2f}".format(assignment_iou),
        "=" * 80,
    ]
    rows = [(index, SYNSET_NAMES[index])
            for index in range(1, len(SYNSET_NAMES))]
    rows.append((len(SYNSET_NAMES), "mean"))
    headers = ["IoU@{:.0f}".format(value * 100) for value in thresholds]
    category_width = 10
    value_width = 10
    header = "{:<{}} | {}".format(
        "category", category_width,
        " | ".join("{:>{}}".format(item, value_width)
                   for item in headers))
    lines.extend([header, "-" * len(header)])
    for row_index, name in rows:
        lines.append("{:<{}} | {}".format(
            name, category_width,
            " | ".join("{:>{}.1f}".format(
                aps[row_index, threshold_index] * 100.0, value_width)
                for threshold_index in range(len(thresholds)))))
    lines.extend([
        "-" * len(header),
        "images={images}, predictions={predictions}, GT={ground_truth}, "
        "assigned prediction-GT pairs={assigned_pairs}".format(**counts),
    ])
    return "\n".join(lines)


def format_standard_catre_report(aps, thresholds):
    lines = [
        "=" * 80,
        "CATRE IoU mAP evaluation report (AP, %)",
        "=" * 80,
    ]
    rows = [(index, SYNSET_NAMES[index])
            for index in range(1, len(SYNSET_NAMES))]
    rows.append((len(SYNSET_NAMES), "mean"))
    headers = ["IoU@{:.0f}".format(value * 100) for value in thresholds]
    category_width = 10
    value_width = 10
    header = "{:<{}} | {}".format(
        "category", category_width,
        " | ".join("{:>{}}".format(item, value_width)
                   for item in headers))
    lines.extend([header, "-" * len(header)])
    for row_index, name in rows:
        lines.append("{:<{}} | {}".format(
            name, category_width,
            " | ".join("{:>{}.1f}".format(
                aps[row_index, threshold_index] * 100.0, value_width)
                for threshold_index in range(len(thresholds)))))
    return "\n".join(lines)


def save_catre_csv(path, aps, thresholds):
    rows = []
    for class_id, name in [
            *[(index, SYNSET_NAMES[index])
              for index in range(1, len(SYNSET_NAMES))],
            (len(SYNSET_NAMES), "mean")]:
        row = {"category": name}
        for threshold_index, threshold in enumerate(thresholds):
            row["iou_{}_ap_percent".format(
                int(round(threshold * 100)))] = (
                    aps[class_id, threshold_index] * 100.0)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def run(args):
    results_dir = os.path.abspath(args.results_dir)
    if not 0.0 <= args.assignment_iou <= 1.0:
        raise ValueError("--assignment_iou must be in [0, 1]")
    report_path = (os.path.abspath(args.report) if args.report else
                   os.path.join(
                       results_dir, "evaluation_report_gt_rt_catre.txt"))
    results, result_paths = load_results(results_dir)
    print("Loaded {} result files from {}".format(
        len(result_paths), results_dir))

    # Start a fresh combined report, then append the original report through
    # the repository's unchanged scoring implementation.
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as output_file:
        output_file.write(
            "Saved-results evaluation\nresults_dir: {}\nimages: {}\n\n".format(
                results_dir, len(results)))
    original_iou_aps, original_pose_aps = compute_independent_mAP(
        results, SYNSET_NAMES,
        degree_thresholds=[5, 10],
        shift_thresholds=[2, 5, 10],
        iou_3d_thresholds=[0.10, 0.25, 0.50, 0.75],
        iou_pose_thres=0.10,
        use_matches_for_pose=True,
        report_path=report_path,
        save_raw_details=False,
        save_overlap_excel=False)

    thresholds = [0.10, 0.25, 0.50, 0.75]
    standard_catre_aps = compute_standard_catre_iou(results, thresholds)
    standard_catre_report = format_standard_catre_report(
        standard_catre_aps, thresholds)
    print(standard_catre_report)
    with open(report_path, "a", encoding="utf-8") as output_file:
        if output_file.tell() > 0:
            output_file.write("\n")
        output_file.write(standard_catre_report.rstrip() + "\n")

    catre_aps, counts = compute_gt_rt_catre_iou(
        results, thresholds, args.assignment_iou)
    catre_report = format_catre_report(
        catre_aps, thresholds, args.assignment_iou, counts)
    print(catre_report)
    with open(report_path, "a", encoding="utf-8") as output_file:
        if output_file.tell() > 0:
            output_file.write("\n")
        output_file.write(catre_report.rstrip() + "\n")

    standard_csv_path = os.path.join(results_dir, "catre_iou.csv")
    gt_rt_csv_path = os.path.join(results_dir, "catre_iou_gt_rt.csv")
    save_catre_csv(standard_csv_path, standard_catre_aps, thresholds)
    save_catre_csv(gt_rt_csv_path, catre_aps, thresholds)
    if args.save_json:
        json_path = os.path.join(
            results_dir, "evaluation_aggregate_gt_rt_catre.json")
        with open(json_path, "w", encoding="utf-8") as output_file:
            json.dump({
                "results_dir": results_dir,
                "assignment_iou": args.assignment_iou,
                "thresholds": thresholds,
                "counts": counts,
                "original_iou_aps": original_iou_aps.tolist(),
                "original_pose_aps": original_pose_aps.tolist(),
                "catre_iou_aps": standard_catre_aps.tolist(),
                "catre_gt_rt_iou_aps": catre_aps.tolist(),
            }, output_file, indent=2, ensure_ascii=False)
        print("Saved aggregate JSON:", json_path)
    print("Saved combined report:", report_path)
    print("Saved CATRE aggregate CSV:", standard_csv_path)
    print("Saved GT-R/t CATRE aggregate CSV:", gt_rt_csv_path)


if __name__ == "__main__":
    run(parse_args())
