import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
from feature_detectors.SuperPoint.superpoint import SuperPoint

from feature_matchers.common.feature_matcher import FeatureMatcher
from feature_matchers.SuperGlue.models.superglue import SuperGlue


class SuperGlueFeatureMatcher(FeatureMatcher):
    def __init__(
        self,
        weights_path: Path,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weights_path = weights_path
        self.superpoint = SuperPoint().eval().to(self.device)
        self.superglue = SuperGlue(weights_path).eval().to(self.device)

    def extract_features(self, image):
        features = self.superpoint({"image": image})
        shape = image.shape[2:]
        features["shape"] = shape
        return features

    def match_features(self, features_0, features_1):
        prediction = {}
        keys = ["shape0", "shape1"]
        prediction = {
            **prediction,
            **{k + "0": (v if k in keys else v) for k, v in features_0.items()},
        }
        prediction = {
            **prediction,
            **{k + "1": (v if k in keys else v) for k, v in features_1.items()},
        }
        for k in prediction:
            if isinstance(prediction[k], (list, tuple)) and not k in keys:
                prediction[k] = torch.stack(prediction[k])
        prediction = {**prediction, **self.superglue(prediction)}
        prediction = {
            k: (v[0].cpu().numpy() if not k in keys else v)
            for k, v in prediction.items()
        }
        kpts0, kpts1 = prediction["keypoints0"], prediction["keypoints1"]
        matches = prediction["matches0"]
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        return mkpts0, mkpts1
