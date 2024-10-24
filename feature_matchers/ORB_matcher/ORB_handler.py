from pathlib import Path

import cv2

from feature_matchers.common.data_handler import DataHandler


class ORBDataHandler(DataHandler):
    def __init__(
        self,
        input_path: Path,
        output_path: Path,
    ):
        super().__init__(input_path, output_path)
        Path(output_path).mkdir(exist_ok=True)

    def read_image(self, filename):
        path_to_image = Path(self.input_path) / filename
        image = cv2.imread(str(path_to_image), cv2.IMREAD_GRAYSCALE)
        return image

    def save_macthes(self, matches):
        out_image = cv2.drawMatches(
            matches["image0"],
            matches["kpts0"],
            matches["image1"],
            matches["kpts1"],
            matches["matches"],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(str(Path(self.output_path) / matches["path"]), out_image)
