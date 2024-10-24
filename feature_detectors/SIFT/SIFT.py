import cv2

from feature_detectors.feature_detector import FeatureDetector


class SIFT(FeatureDetector):
    def detect(self, image):
        sift = cv2.SIFT_create()
        features = {}
        features["keypoints"], features["descriptors"] = sift.detectAndCompute(
            image, None
        )
        return features
