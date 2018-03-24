import os
from .labelled import labelled
from .image_files import image_files

def labelled_images(path, shape, labels):
    path = os.path.abspath(path)

    dataset = labelled({
        lb: image_files(os.path.join(path, lb),
                        glob='*.bmp',
                        shape=shape) for lb in labels
    }, label_key='label')

    return dataset
