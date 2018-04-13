import os

import tensorflow as tf
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorflow.python.data.util import nest

class PlotPage:
    def __init__(self, feature, nbatch):
        self._feature = feature

        image = self._feature['image']
        self._nbatch = nbatch

    def draw_bbox(self, ax, bbox):
        ax.add_patch(mpl.patches.Rectangle((bbox[1], bbox[0]), bbox[3], bbox[2], fill=False, edgecolor='red'))

    def get_path(self, n):
        path = self._feature.get('path', None)
        if path is not None:
            return path[n].decode()

    def draw_image(self, ax, image):
        shape = image.shape

        if len(shape) != 3:
            raise Exception("Input images must have rank 3: {}".format(len(shape)))

        if shape[-1] == 1:
            ax.imshow(image[...,0], cmap='gray')
        elif shape[-1] == 3:
            ax.imshow(image)
        else:
            raise Exception("Cannot draw image of shape: {}".format(str(shape)))

    def draw(self, ax_array):
        n = 0

        label = self._feature.get('label', None)
        image = self._feature.get('image', None)
        plate = self._feature.get('plate', None)

        for row_i in ax_array:
            for ax_ij in row_i:
                ax_ij.clear()
                ax_ij.set_aspect('equal', 'box')
                ax_ij.axis('off')

                if n < self._nbatch:
                    if label is not None:
                        ax_ij.set_title('{0}'.format(label[n]))

                    if image is not None:
                        image_slice = image[n,...]
                        self.draw_image(ax_ij, image_slice)

                    if plate is not None:
                        plate_slice = plate[n,...]
                        self.draw_bbox(ax_ij, plate_slice)

                n = n+1


class PlotWindow:
    def __init__(self, sess, dataset):
        self._sess = sess

        # Dataset
        self._dataset = dataset

        it = self._dataset.make_one_shot_iterator()
        feature = it.get_next()

        self._feature = feature

        self._nbatch = 1
        if 'label' in self._feature:
            shape = self._sess.run([tf.shape(self._feature['label'])])
            self._nbatch = shape[0]

        # Prepare figure
        gridh, gridw = self.grid_shape(self._nbatch, 16/9)
        fig, ax_array = plt.subplots(gridh, gridw, squeeze=True)

        self._figure = fig
        self._ax_array = ax_array

        self.draw()

        self._figure.canvas.mpl_connect('key_press_event', self.onkeypress)
        self._figure.canvas.mpl_connect('button_press_event', self.onclick)

        plt.show(self._figure)

    def grid_shape(self, nbatch, aspect):
        x = np.sqrt(nbatch / aspect)
        gridh = int(np.floor(x))
        gridw = int(np.ceil(aspect * x))

        if gridh * gridw <= nbatch:
            return gridh, gridw
        else:
            return gridh+1, gridw

    def get_page(self):
        flat = nest.flatten(self._feature)
        flat_val = self._sess.run(flat)
        feature = nest.pack_sequence_as(self._feature, flat_val)
        return PlotPage(feature, self._nbatch)

    def draw(self):
        self._page = self.get_page()
        self._page.draw(self._ax_array)
        self._figure.canvas.draw()

    def onkeypress(self, event):
        if event.key == 'pagedown':
            print("pagedown")
            self.draw()

    def onclick(self, event):
        n = 0
        for row_i in self._ax_array:
            for ax_ij in row_i:
                if ax_ij.in_axes(event):
                    if (n < self._nbatch):
                        path = self._page.get_path(n)
                        print("path: %s" % path)
                        os.startfile(path)
                n = n+1


def plot_sample(dataset):
    with tf.Session() as sess:
        win = PlotWindow(sess, dataset)


def print_sample(dataset):
    with tf.Session() as sess:
        it = dataset.make_one_shot_iterator()
        feature = it.get_next()
        while True:
            flat = nest.flatten(feature)
            flat_val = sess.run(flat)
            np_feature = nest.pack_sequence_as(feature, flat_val)
            print(np_feature)
