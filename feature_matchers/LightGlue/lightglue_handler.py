import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
from feature_matchers.common.data_handler import DataHandler


class LightGlueDataHandler(DataHandler):
    def __init__(
        self,
        input_path: Path,
        output_path: Path,
    ):
        super().__init__(input_path, output_path)
        Path(output_path).mkdir(exist_ok=True)

    def read_image(self, filename):
        path_to_image = Path(self.input_path) / filename
        image = cv2.imread(str(path_to_image), cv2.IMREAD_COLOR)
        image = image[..., ::-1]
        if image.ndim == 3:
            image = image.transpose((2, 0, 1))
        elif image.ndim == 2:
            image = image[None]
        img_tensor = torch.tensor(image / 255.0, dtype=torch.float)
        return image, img_tensor

    def save_macthes(self, matches):
        margin = 10
        matches["image0"] = np.transpose(matches["image0"], (1, 2, 0))
        matches["image1"] = np.transpose(matches["image1"], (1, 2, 0))
        H0, W0, _ = matches["image0"].shape
        H1, W1, _ = matches["image1"].shape
        H, W = max(H0, H1), W0 + W1 + margin

        out = 255 * np.ones((H, W, 3), dtype=np.uint8)
        out[:H0, :W0, :] = matches["image0"]
        out[:H1, W0 + margin :, :] = matches["image1"]
        mkpts0, mkpts1 = (
            torch.round(matches["mkpts0"]).type(torch.int32),
            torch.round(matches["mkpts1"]).type(torch.int32),
        )
        for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
            cv2.line(
                out,
                (int(x0), int(y0)),
                (int(x1 + margin + W0), int(y1)),
                color=(0, 255, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                out, (int(x0), int(y0)), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA
            )
            cv2.circle(
                out,
                (int(x1 + margin + W0), int(y1)),
                2,
                (0, 255, 0),
                -1,
                lineType=cv2.LINE_AA,
            )
        cv2.imwrite(str(Path(self.output_path) / matches["path"]), out)
