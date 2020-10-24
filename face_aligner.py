import PIL.Image
import os

from .ffhq_dataset.face_alignment import image_align
from .ffhq_dataset.landmarks_detector import LandmarksDetector


class FaceAligner:
    def __init__(self,
                 output_size=256,
                 x_scale=1,
                 y_scale=1,
                 em_scale=0.1,
                 use_alpha=False):
        shape_predictor_dir = os.path.dirname(os.path.abspath(__file__))
        self.landmarks_detector = LandmarksDetector(
            os.path.join(shape_predictor_dir, 'shape_predictor_68_face_landmarks.dat'))
        self.output_size = output_size
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.em_scale = em_scale
        self.alpha = use_alpha

    def align(self, image: PIL.Image) -> (PIL.Image, str):
        face_landmarks_list = self.landmarks_detector.get_landmarks(image)

        if len(face_landmarks_list) == 0:
            return None, 'No face detected'

        if len(face_landmarks_list) > 1:
            return None, 'More than one face found'

        face_landmarks = face_landmarks_list[0]
        aligned_image = image_align(image,
                                    face_landmarks,
                                    output_size=self.output_size,
                                    x_scale=self.x_scale,
                                    y_scale=self.y_scale,
                                    em_scale=self.em_scale,
                                    alpha=self.alpha)

        return aligned_image, None
