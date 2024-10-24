from abc import ABC, abstractmethod


class FeatureDetector(ABC):

    @abstractmethod
    def detect(self, image):
        pass
