import os
import pickle as pkl

import numpy as np
import torch
from mesh_to_sdf import scale_to_unit_sphere, mesh_to_sdf
from mesh_to_sdf.surface_point_cloud import create_from_scans


def F1_metric(decoder, mesh, eps=1e-3, cache_path=None):
    cache_path = os.path.join("cache", cache_path)

    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pkl.load(f)
            points = data["points"]
            gt_sdf_sign = data["gt_sdf_sign"]
    else:
        mesh = scale_to_unit_sphere(mesh)
        surface_pc = create_from_scans(mesh, scan_count=100, scan_resolution=400)
        points = surface_pc.points + np.random.normal(scale=eps, size=(surface_pc.points.shape[0], 3))
        gt_sdf_sign = np.sign(mesh_to_sdf(mesh, points))

        with open(cache_path, "wb") as f:
            pkl.dump({
                "points": points,
                "gt_sdf_sign": gt_sdf_sign
            }, f)

    with torch.no_grad():

        points = torch.from_numpy(points.astype(np.float32)).cuda()
        pred_sdf = np.concatenate([decoder(batch).cpu().numpy() for batch in torch.chunk(points, 1024 * 16)])
        pred_sdf_sign = np.sign(np.squeeze(pred_sdf))

    TP = np.sum(np.logical_and(gt_sdf_sign > 0, pred_sdf_sign > 0))
    FP = np.sum(np.logical_and(gt_sdf_sign < 0, pred_sdf_sign > 0))
    FN = np.sum(np.logical_and(gt_sdf_sign > 0, pred_sdf_sign < 0))

    F1 = TP / (TP + (FP + FN) / 2.0)
    return F1
