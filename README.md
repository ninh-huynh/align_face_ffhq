# align_face_ffhq
Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step


```python
import PIL.Image
from align_face.ffhq.face_aligner import FaceAligner

face_aligner = FaceAligner()

src_image = PIL.Image.open('sample_image.jpg')
cropped_image, error_message = face_aligner.align(src_image)

```