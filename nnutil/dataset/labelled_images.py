import os
from .labelled import labelled
from .image_files import image_files

def labelled_images(path, shape, labels=None):
    path = os.path.abspath(path)

    if labels is None:
        labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    dataset = labelled({
        lb: image_files(os.path.join(path, lb),
                        glob='*.bmp',
                        shape=shape) for lb in labels
    }, label_key='label')

    return dataset
