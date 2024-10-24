import argparse
import sys
from pathlib import Path

import cv2
import g2o
import numpy as np
import open3d as o3d
import torch
import yaml

sys.path.append(str(Path(__file__).absolute().parent.parent))
from feature_matchers.LightGlue.lightglue_handler import LightGlueDataHandler
from feature_matchers.LightGlue.lightglue_matcher import \
    LightGlueFeatureMatcher

torch.set_grad_enabled(False)


def read_camera_matrix(path_to_matrix):
    skip_lines = 0
    with open(path_to_matrix) as infile:
        for i in range(skip_lines):
            _ = infile.readline()
        data = yaml.safe_load(infile)

    intrinsic = data["camera_matrix"]["data"]
    K = np.asarray(
        [[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[4], intrinsic[5]], [0, 0, 1]]
    )
    D = np.asarray(data["distortion_coefficients"]["data"])
    return K, D


def estimate_pose(kpn_ref, kpn_cur, K, D):
    ransac_method = cv2.RANSAC
    kpn_cur = kpn_cur.numpy().astype(np.float32)
    kpn_ref = kpn_ref.numpy().astype(np.float32)
    E, mask_match = cv2.findEssentialMat(kpn_cur, kpn_ref, K, method=ransac_method)
    _, E, R, t, mask = cv2.recoverPose(
        points1=kpn_cur,
        points2=kpn_ref,
        cameraMatrix1=K,
        distCoeffs1=D,
        cameraMatrix2=K,
        distCoeffs2=D,
        E=E,
        mask=mask_match,
    )
    return R, t.T


def viz(R, t):
    mesh_frame_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t
    mesh_frame.transform(transformation_matrix)
    o3d.visualization.draw_geometries([mesh_frame_1, mesh_frame])


def optimize(R, t):
    opt = g2o.SparseOptimizer()
    block_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
    opt.set_algorithm(solver)

    v1 = g2o.VertexSE3Expmap()
    v1.set_estimate(g2o.SE3Quat(np.eye(3), np.array([0, 0, 1])))
    v1.set_id(0)
    v1.set_fixed(True)
    opt.add_vertex(v1)

    v2 = g2o.VertexSE3Expmap()
    v2.set_id(1)
    v2.set_estimate(g2o.SE3Quat(R, t.flatten()))
    opt.add_vertex(v2)

    edge = g2o.EdgeSE3()
    edge.set_id(0)
    edge.set_vertex(0, v1)
    edge.set_vertex(1, v2)
    opt.add_edge(edge)
    opt.initialize_optimization()
    opt.optimize(10)
    opt_v1 = opt.vertex(0).estimate()
    opt_v2 = opt.vertex(1).estimate()
    return opt_v1, opt_v2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_dir", type=str, help="Path to the directory that contains the images"
    )
    parser.add_argument(
        "--input_pairs", type=str, help="Path to the list of image pairs"
    )
    parser.add_argument(
        "-m",
        "--matrix_dir",
        type=str,
        help="Path to the directory that contains intrinsics and distortion",
    )

    args = parser.parse_args()
    with open(args.input_pairs, "r") as f:
        pairs = [l.split() for l in f.readlines()]
    K, D = read_camera_matrix(args.matrix_dir)
    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        data_handler = LightGlueDataHandler(args.input_dir, args.output_dir)
        matcher = LightGlueFeatureMatcher()
        img0, inp0 = data_handler.read_image(name0)
        img1, inp1 = data_handler.read_image(name1)
        features0 = matcher.extract_features(inp0)
        features1 = matcher.extract_features(inp1)
        mkpts0, mkpts1 = matcher.match_features(features0, features1)
        R, t = estimate_pose(mkpts0, mkpts1, K, D)
        v1, v2 = optimize(R, t)
        print(R, t)
        print(v2.rotation().matrix(), v2.translation())
        viz(R, t)
