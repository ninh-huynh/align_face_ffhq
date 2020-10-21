# align_face_ffhq
Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step

```
usage: align_images.py [-h] [--output_size OUTPUT_SIZE] [--x_scale X_SCALE]
                       [--y_scale Y_SCALE] [--em_scale EM_SCALE]
                       raw_dir aligned_dir

Align faces from input images

positional arguments:
  raw_dir               Directory with raw images for face alignment
  aligned_dir           Directory for storing aligned images

optional arguments:
  -h, --help            show this help message and exit
  --output_size OUTPUT_SIZE
                        The dimension of images for input to the model
                        (default: 1024)
  --x_scale X_SCALE     Scaling factor for x dimension (default: 1)
  --y_scale Y_SCALE     Scaling factor for y dimension (default: 1)
  --em_scale EM_SCALE   Scaling factor for eye-mouth distance (default: 0.1)
```
