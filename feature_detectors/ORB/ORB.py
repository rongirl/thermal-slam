import cv2

from feature_detectors.feature_detector import FeatureDetector


class ORB(FeatureDetector):
    def detect(self, image):
        orb = cv2.ORB_create()
        features = {}
        features["keypoints"], features["descriptors"] = orb.detectAndCompute(
            image, None
        )
        return features
