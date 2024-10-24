import argparse
from pathlib import Path

import cv2
import numpy as np
from cv_bridge import CvBridge
from rosbags.highlevel import AnyReader

VISUAL_CAM_TOPIC = "/cam_blackfly/cam_blackfly"
THERMAL_CAM_TOPIC = "/tau_nodelet/thermal_image"


def equalize_histogram(img, cummulative_pixels):
    intensity_min = 0
    intensity_max = np.iinfo(np.uint16).max - 1
    hist_size = intensity_max - intensity_min
    range_values = np.array([intensity_min, intensity_max], dtype=np.float32)

    histogram = cv2.calcHist([img], [0], None, [hist_size], range_values)

    bin_index = 1
    sum_pixels = 0
    while sum_pixels < cummulative_pixels and bin_index < hist_size:
        sum_pixels += histogram[bin_index][0]
        bin_index += 1
    min_val = bin_index - 1

    bin_index = hist_size - 1
    sum_pixels = 0
    while sum_pixels < cummulative_pixels and bin_index > 0:
        sum_pixels += histogram[bin_index][0]
        bin_index -= 1
    max_val = bin_index + 1

    alpha = 255.0 / (max_val - min_val)
    beta = -min_val * alpha

    out_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return out_img


def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=200.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def extract_images(reader, topics, output_path, histogram, cummulative_pixels):
    bridge = CvBridge()
    for i, topic in enumerate(topics):
        camera_folder = output_path / f"cam{topic.replace('/', '_')}"
        camera_folder.mkdir(exist_ok=True)

        connections = [x for x in reader.connections if x.topic == topic]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            time = str(msg.header.stamp.sec) + str(msg.header.stamp.nanosec).rjust(
                9, "0"
            )
            path_to_image = camera_folder / f"{time}.png"
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
            if topic == THERMAL_CAM_TOPIC:
                if histogram == "default":
                    cv_image = equalize_histogram(cv_image, cummulative_pixels)
                else:
                    cv_image = clahe(cv_image)
            cv2.imwrite(str(path_to_image), cv_image)


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
    parser.add_argument(
        "-hist",
        "--histogram",
        choices={"default", "clahe"},
        default="default",
        type=str,
        help="Normalization method",
    )
    parser.add_argument(
        "--cummulative_pixels",
        type=int,
        default=10000,
        help="Minimum count of pixels",
    )

    args = parser.parse_args()
    bag_path = args.input_file
    output_path = args.output_dir
    output_path.mkdir(exist_ok=True)
    with AnyReader([bag_path]) as reader:
        print(f"Extracting images from {bag_path}")
        extract_images(
            reader,
            [VISUAL_CAM_TOPIC, THERMAL_CAM_TOPIC],
            output_path,
            args.histogram,
            args.cummulative_pixels,
        )
