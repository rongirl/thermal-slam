from abc import ABC, abstractmethod


class FeatureMatcher(ABC):

    @abstractmethod
    def extract_features(self, image):
        pass

    @abstractmethod
    def match_features(self, features_0, features_1):
        pass
