import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import rosbag
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

POINT_CLOUD_TOPIC = "/os1_node/points_raw"


def convert_to_pcd(msg: PointCloud2):
    points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    points_np = np.array(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    return pcd


def extract_point_cloud(bag_file, output_path, topic_name):
    bag = rosbag.Bag(bag_file)
    point_cloud_folder = Path(output_path) / f"{topic_name.replace('/', '_')}"
    point_cloud_folder.mkdir(exist_ok=True)
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        output_pcd_file = point_cloud_folder / f"{msg.header.stamp}.pcd"
        pcd = convert_to_pcd(msg)
        o3d.io.write_point_cloud(str(output_pcd_file), pcd)
    bag.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-file",
        type=Path,
        help="Path to the input ROSBag file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Path to the output directory",
    )
    args = parser.parse_args()

    bag_path = args.input_file
    output_path = args.output_dir
    output_path.mkdir(exist_ok=True)
    print(f"Extracting point cloud from {bag_path}")
    extract_point_cloud(bag_path, output_path, POINT_CLOUD_TOPIC)
