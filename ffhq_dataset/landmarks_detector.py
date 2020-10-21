import dlib
import numpy as np
import PIL
import traceback

class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, pil_image: PIL.Image):
        # img = dlib.load_rgb_image(image)
        img = np.array(pil_image)
        dets = self.detector(img, 1)
        face_landmarks_list = []
        
        for detection in dets:
            try:
                face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
                face_landmarks_list.append(face_landmarks)
                # yield face_landmarks
            except:
                traceback.print_exc()
        
        return face_landmarks_list
