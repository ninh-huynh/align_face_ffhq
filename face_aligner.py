import PIL
import bz2
from keras.utils import get_file
from .ffhq_dataset.face_alignment import image_align
from .ffhq_dataset.landmarks_detector import LandmarksDetector

def unpack_bz2(src_path):
  data = bz2.BZ2File(src_path).read()
  dst_path = src_path[:-4]
  with open(dst_path, 'wb') as fp:
    fp.write(data)
  return dst_path  

class FaceAligner:
  def __init__(self, 
               output_size = 256,
               x_scale=1,
               y_scale=1,
               em_scale=0.1,
               use_alpha=False):
    LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))
    self.landmarks_detector = LandmarksDetector(landmarks_model_path)
    self.output_size = output_size
    self.x_scale = x_scale
    self.y_scale = y_scale
    self.em_scale = em_scale
    self.alpha = use_alpha

  def align(self, image: PIL.Image) -> PIL.Image:
    face_landmarks_list = self.landmarks_detector.get_landmarks(image)
    
    if len(face_landmarks_list) == 0 or len(face_landmarks_list) > 1:
      return null

    face_landmarks = face_landmarks_list[0]
    return image_align(image,
                face_landmarks, 
                output_size=self.output_size, 
                x_scale=self.x_scale, 
                y_scale=self.y_scale, 
                em_scale=self.em_scale, 
                alpha=self.alpha)

