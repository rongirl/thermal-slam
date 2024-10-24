import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
from feature_detectors.SuperPoint.superpoint import SuperPoint

from feature_matchers.common.feature_matcher import FeatureMatcher
from feature_matchers.LightGlue.models.lightglue import LightGlue


class LightGlueFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.superpoint = SuperPoint().eval().to(self.device)
        self.lightglue = LightGlue().eval().to(self.device)

    def extract_features(self, image):
        if image.dim() == 3:
            image = image[None]
        shape = image.shape[-2:][::-1]
        features = self.superpoint({"image": image})
        features["image_size"] = torch.tensor(shape)[None].to(image).float()
        features["descriptors"] = features["descriptors"].transpose(-1, -2).contiguous()
        return features

    def match_features(self, features_0, features_1):
        matches = self.lightglue({"image0": features_0, "image1": features_1})
        kpts0, kpts1, matches = (
            features_0["keypoints"][0],
            features_1["keypoints"][0],
            matches["matches"][0],
        )
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        return m_kpts0, m_kpts1
