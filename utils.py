from skimage import img_as_float
from skimage.io import imread

def load_image(filepath, as_gray=True):
    return img_as_float(imread(filepath, as_gray=as_gray))
