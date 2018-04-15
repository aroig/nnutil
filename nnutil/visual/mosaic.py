import os
import platform
import subprocess

import tensorflow as tf
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorflow.python.data.util import nest


class MosaicItem:
    def __init__(self, ax, image_fn=None, path_fn=None, label_fn=None, box_fn=None):
        self._feature = None
        self._ax = ax

        if image_fn is None:
            self._image_fn = lambda x: x['image']
        elif type(image_fn) == str:
            self._image_fn = lambda x: x[image_fn]
        else:
            self._image_fn = image_fn

        if path_fn is None:
            self._path_fn = lambda x: x['path'] if 'path' in x else None
        elif type(path_fn) == str:
            self._path_fn = lambda x: x[path_fn] if path_fn in x else None
        else:
            self._path_fn = path_fn

        if label_fn is None:
            self._label_fn = lambda x: x['label'] if 'label' in x else None
        elif type(label_fn) == str:
            self._label_fn = lambda x: x[label_fn] if label_fn in x else None
        else:
            self._label_fn = label_fn

        if box_fn is None:
            self._box_fn = lambda x: x['box'] if 'box' in x else None
        elif type(box_fn) == str:
            self._box_fn = lambda x: x[box_fn] if box_fn in x else None
        else:
            self._box_fn = box_fn

    def _draw_bbox(self, bbox, shape):
        # BBOX: ymin, xmin, height, width
        self._ax.add_patch(mpl.patches.Rectangle(
            (bbox[1] * shape[1], bbox[0] * shape[0]),
            bbox[3] * shape[1], bbox[2] * shape[0],
            fill=False,
            edgecolor='red'))

    def _draw_image(self, image):
        shape = image.shape

        if len(shape) != 3:
            raise Exception("Input images must have rank 3: {}".format(len(shape)))

        if shape[-1] == 1:
            self._ax.imshow(image[...,0], cmap='gray')

        elif shape[-1] == 3:
            self._ax.imshow(image)

        else:
            raise Exception("Cannot draw image of shape: {}".format(str(shape)))

    @property
    def image(self):
        return self._image_fn(self._feature)

    @property
    def path(self):
        val = self._path_fn(self._feature)
        return np.asscalar(val).decode() if val is not None else None

    @property
    def label(self):
        val = self._label_fn(self._feature)
        if val is None:
            return None

        val = np.asscalar(val)
        if type(val) == bytes:
            return val.decode()
        elif type(val) == int:
            return val
        else:
            return None

    @property
    def box(self):
        return self._box_fn(self._feature)

    @property
    def feature(self):
        return self._feature

    def draw(self, selected=False):
        self._ax.clear()

        self._ax.set_yticklabels([])
        self._ax.set_xticklabels([])
        for spine in self._ax.spines.values():
            spine.set_linewidth(2)

        self._ax.set_aspect('equal', 'box')

        for spine in self._ax.spines.values():
            if selected:
                spine.set_color('red')
                spine.set_visible(True)
            else:
                spine.set_visible(False)

        if self._feature is not None:
            if self.label is not None:
                self._ax.set_title('{0}'.format(self.label))

            self._draw_image(self.image)

            shape = self.image.shape
            if self.box is not None:
                self._draw_bbox(self.box, shape)

    def in_axes(self, event):
        return self._ax.in_axes(event)

    def set_feature(self, feature):
        self._feature = feature


class MosaicWindow:
    def __init__(self, sess, dataset, image_fn=None, label_fn=None, path_fn=None):
        self._sess = sess
        self._dataset = dataset
        self._selected = None

        it = self._dataset.make_one_shot_iterator()
        self._feature = it.get_next()

        self._image_fn = image_fn
        self._label_fn = label_fn
        self._path_fn = path_fn

        self._items = None
        self.next_page()

        self._figure.canvas.mpl_connect('key_press_event', self.onkeypress)
        self._figure.canvas.mpl_connect('button_press_event', self.onclick)

        plt.show(self._figure)

    @property
    def selected_item(self):
        if self._selected is not None:
            return self._items[self._selected]
        else:
            return None

    def grid_shape(self, nbatch, aspect):
        x = np.sqrt(nbatch / aspect)
        gridh = int(np.floor(x))
        gridw = int(np.ceil(aspect * x))

        if gridh * gridw <= nbatch:
            return gridh, gridw
        else:
            return gridh+1, gridw

    def init_items(self, nbatch):
        self._nbatch = nbatch

        # Prepare figure
        gridh, gridw = self.grid_shape(self._nbatch, 16/9)
        fig, ax_array = plt.subplots(gridh, gridw, squeeze=True)

        self._figure = fig
        self._ax_array = ax_array

        self._items = []
        for row in self._ax_array:
            for ax in row:
                self._items.append(MosaicItem(
                    ax,
                    image_fn=self._image_fn,
                    label_fn=self._label_fn,
                    path_fn=self._path_fn))

    def draw(self):
        for i, item in enumerate(self._items):
            item.draw(selected=(self._selected == i))
        self._figure.canvas.draw()

    def next_page(self):
        flat = nest.flatten(self._feature)
        flat_val = self._sess.run(flat)

        nbatch = flat_val[0].shape[0]
        for x in flat_val:
            if x.shape[0] != nbatch:
                raise Exception("Uneven batch dimension in feature")

        if self._items is None:
            self.init_items(nbatch)

        self._selected = None
        for i, item in enumerate(self._items):
            flat_i = tuple([x[i, ...] for x in flat_val])
            feature = nest.pack_sequence_as(self._feature, flat_i)
            item.set_feature(feature)

        self.draw()

    def onkeypress(self, event):
        if event.key == 'pagedown':
            print("pagedown")
            self.next_page()
        elif event.key == 'v':
            item = self.selected_item
            if item is not None:
                path = item.path
                print("path: %s" % path)
                if platform.system() == 'Windows':
                    os.startfile(path)
                else:
                    subprocess.run(['xdg-open', path])

    def onclick(self, event):
        for i, item in enumerate(self._items):
            if item.in_axes(event):
                self._selected = i
                self.draw()
                print("selection: {}".format(i))
                return
        self._selected = None
