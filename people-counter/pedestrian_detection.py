"""
 Pedestrian detection wrapper class (OpenVino model wrapper)

 * @author Wajih Ouertani
 * @email wajih.ouertani@gmail.com
 """

import logging
from typing import List

import numpy as np

from openvino_utils.single_image_openvino_model import SingleImageOpenVinoModel
from openvino_utils.utils import RatioDetection, RatioPoint

LOGGER = logging.getLogger()


class PedestrianDetection(SingleImageOpenVinoModel):

    """Pedestrian detection Model class"""

    model_name: str = "pedestrian-detection-adas-0002"
    model_directory: str = "models/intel/pedestrian-detection-adas-0002/FP32"

    def infer(
        self,
        image: np.ndarray,
    ) -> List[RatioDetection]:
        """Extract sematic segmentation"""

        # check if the image is matching the expected dimension otherwise pre-process it
        to_infer = image.copy()
        if to_infer.shape != self.expected_input_shape:
            to_infer = self.preprocess_input(to_infer)
        # run inference
        results = self.predict(to_infer)["detection_out"][0][0]
        return results

    def detect(
        self,
        image: np.ndarray,
        detections=None,
        min_confidence: float = 0.95,
    ) -> List[RatioDetection]:
        """Extract/get detection
        :param image: model ready preprocessed image
        :param output_dimension: output dimension for each of the detected face
        :param detections: detection if set to None an inference is run to generate the detections
        :param min_confidence: minimum confidence to filter detections

        :return: pedestrian RationDetection list
        """
        # Check if the model is loaded correctly!
        if detections is None:
            # check if the image is matching the expected dimension otherwise pre-process it
            to_infer = image.copy()
            if to_infer.shape != self.expected_input_shape:
                to_infer = self.preprocess_input(to_infer)
            # run inference
            detections = self.predict(to_infer)["detection_out"][0][0]

        # filter the detection to keep only high confidence detection (bounding boxes)
        filtered = detections[detections[:, 2] >= min_confidence]

        LOGGER.debug(
            "detections (at confidence > %.2f): %d", min_confidence, filtered.shape[0]
        )

        pedestrians: List[RatioDetection] = []

        for _, _, confidence, start_x, start_y, end_x, end_y in filtered:
            pedestrians.append(
                RatioDetection(
                    top_left=RatioPoint(start_x, start_y),
                    bottom_right=RatioPoint(end_x, end_y),
                    confidence=confidence,
                )
            )

        return sorted(
            pedestrians, key=lambda ratio_detection: ratio_detection.size, reverse=True
        )
