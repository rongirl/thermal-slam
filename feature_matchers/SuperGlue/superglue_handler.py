import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
from feature_matchers.common.data_handler import DataHandler


class SuperGlueDataHandler(DataHandler):
    def __init__(self, input_path: Path, output_path: Path, resize: List[int]):
        super().__init__(input_path, output_path)
        Path(output_path).mkdir(exist_ok=True)
        self.resize = resize

    def _resize_image(self, image):
        return cv2.resize(
            image, (self.resize[0], self.resize[1]), interpolation=cv2.INTER_AREA
        )

    def undistort_image(self, image, K, D):
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newcameramtx, (w, h), cv2.CV_16SC2)
        undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        D = np.zeros((4, 1))
        return undistorted_img, newcameramtx, D

    def read_image(self, filename, K, D):
        path_to_image = Path(self.input_path) / filename
        image = cv2.imread(str(path_to_image), 0)
        if image is None:
            return None, None
        image = self._resize_image(image)
        image, K, D = self.undistort_image(image, K, D)
        img_tensor = torch.from_numpy(image / 255.0).float()[None, None]
        return image, img_tensor, K, D

    def save_macthes(self, matches):
        margin = 10
        H0, W0 = matches["image0"].shape
        H1, W1 = matches["image1"].shape
        H, W = max(H0, H1), W0 + W1 + margin

        out = 255 * np.ones((H, W), np.uint8)
        out[:H0, :W0] = matches["image0"]
        out[:H1, W0 + margin :] = matches["image1"]
        out = np.stack([out] * 3, -1)
        mkpts0, mkpts1 = np.round(matches["mkpts0"]).astype(int), np.round(
            matches["mkpts1"]
        ).astype(int)
        for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
            # cv2.line(
            #     out,
            #     (int(x0), int(y0)),
            #     (int(x1 + margin + W0), int(y1)),
            #     color=(0, 0, 255),
            #     thickness=1,
            #     lineType=cv2.LINE_AA,
            # )
            cv2.circle(
                out, (int(x0), int(y0)), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA
            )
            cv2.circle(
                out,
                (int(x1 + margin + W0), int(y1)),
                2,
                (0, 0, 255),
                -1,
                lineType=cv2.LINE_AA,
            )
        cv2.imwrite(str(Path(self.output_path) / matches["path"]), out)
