import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).parent.parent.parent))
from feature_detectors.ORB.ORB import ORB

from feature_matchers.common.feature_matcher import FeatureMatcher


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.orb = ORB()

    def extract_features(self, image):
        features = self.orb.detect(image)
        return features

    def match_features(self, features_0, features_1):
        matcher = cv2.BFMatcher()
        matches = matcher.match(features_0["descriptors"], features_1["descriptors"])
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
