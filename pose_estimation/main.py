import argparse
import sys
from pathlib import Path

import cv2
import g2o
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import yaml

sys.path.append(str(Path(__file__).absolute().parent.parent))
from feature_matchers.LightGlue.lightglue_handler import LightGlueDataHandler
from feature_matchers.LightGlue.lightglue_matcher import \
    LightGlueFeatureMatcher
from feature_matchers.SuperGlue.superglue_handler import SuperGlueDataHandler
from feature_matchers.SuperGlue.superglue_matcher import \
    SuperGlueFeatureMatcher

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
    Kinv = np.array(
        [
            [1 / intrinsic[0], 0, -intrinsic[2] / intrinsic[0]],
            [0, 1 / intrinsic[4], -intrinsic[5] / intrinsic[4]],
            [0, 0, 1],
        ]
    )
    D = np.asarray(data["distortion_coefficients"]["data"])
    return K, Kinv, D


# def pixel_to_cam_norm_plane(pts, K):
#     pts = np.array(pts)
#
#     x_norm = (pts[:, 0] - K[0, 2]) / K[0, 0]
#     y_norm = (pts[:, 1] - K[1, 2]) / K[1, 1]
#
#     pts_norm = np.column_stack((x_norm, y_norm))
#
#     return pts_norm
#
# def add_ones(x):
#     if len(x.shape) == 1:
#         return np.array([x[0], x[1], 1])
#     else:
#         return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
#
# def unproject_points(uvs, Kinv):
#     return np.dot(Kinv, add_ones(uvs).T).T[:, 0:2]
#
# def undistort_points(uvs, D, K):
#     uvs_contiguous = np.ascontiguousarray(uvs[:, :2]).reshape((uvs.shape[0], 1, 2))
#     uvs_undistorted = cv2.undistortPoints(uvs_contiguous, K, D, None, K)
#     return uvs_undistorted.ravel().reshape(uvs_undistorted.shape[0], 2)


def estimate_pose(points1, points2, K, D):
    inliers_index = []
    ransac_method = cv2.RANSAC
    E, mask_match = cv2.findEssentialMat(points1, points2, K, method=ransac_method)
    for i in range(mask_match.shape[0]):
        if mask_match[i][0] == 1:
            inliers_index.append(i)
    _, E, R, t, mask = cv2.recoverPose(
        points1=points1,
        points2=points2,
        cameraMatrix1=K,
        distCoeffs1=D,
        cameraMatrix2=K,
        distCoeffs2=D,
        E=E,
        mask=mask_match,
    )
    return R, t.T, inliers_index


def visualize(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="b", s=100)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("3D график точек")

    plt.show()


def do_triangulation(
    pts_on_np1, pts_on_np2, R_cam1_to_cam2, t_cam1_to_cam2, inliers, K
):
    inlier_pts_on_np1 = [pts_on_np1[idx] for idx in inliers]
    inlier_pts_on_np2 = [pts_on_np2[idx] for idx in inliers]

    inlier_pts_on_np1 = np.array(inlier_pts_on_np1).T
    inlier_pts_on_np2 = np.array(inlier_pts_on_np2).T

    T_cam1_to_world = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    T_cam2_to_world = np.hstack((R_cam1_to_cam2, t_cam1_to_cam2.reshape(-1, 1)))

    pts4d_in_world = cv2.triangulatePoints(
        K @ T_cam1_to_world, K @ T_cam2_to_world, inlier_pts_on_np1, inlier_pts_on_np2
    )

    pts3d_in_world = []
    for i in range(pts4d_in_world.shape[1]):
        x = pts4d_in_world[:, i]
        x /= x[3]
        pt3d_in_world = x[:3]
        pts3d_in_world.append(pt3d_in_world)
    return np.array(pts3d_in_world)


def vis(R, t, points):
    mesh_frame_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    colors = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    mesh_frame_1.vertex_colors = o3d.utility.Vector3dVector(
        np.repeat(colors, 2, axis=0)
    )
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t
    mesh_frame.transform(transformation_matrix)
    points = np.array(points, dtype=np.float64)
    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(points)
    pcd_1.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([mesh_frame_1, mesh_frame, pcd_1])


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
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the directory in which the visualization images are written",
    )

    parser.add_argument(
        "-w",
        type=str,
        help="Path to the SuperGlue weights",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs="+",
        default=[640, 480],
        help="Resize the input image before running SuperGlue",
    )

    args = parser.parse_args()
    with open(args.input_pairs, "r") as f:
        pairs = [l.split() for l in f.readlines()]
    K, Kinv, D = read_camera_matrix(args.matrix_dir)
    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        data_handler = SuperGlueDataHandler(
            args.input_dir, args.output_dir, args.resize
        )
        matcher = SuperGlueFeatureMatcher(args.w)
        img0, inp0, K1, D1 = data_handler.read_image(name0, K, D)
        img1, inp1, K1, D1 = data_handler.read_image(name1, K, D)
        features0 = matcher.extract_features(inp0)
        features1 = matcher.extract_features(inp1)
        mkpts0, mkpts1 = matcher.match_features(features0, features1)
        R, t, inliers_index = estimate_pose(mkpts0, mkpts1, K1, D1)
        pts3 = do_triangulation(mkpts0, mkpts1, R, t, inliers_index, K1)
        vis(R, t, pts3)
        visualize(pts3)
