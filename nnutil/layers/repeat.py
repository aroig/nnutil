import copy

import tensorflow as tf
from .segment import Segment

class Repeat(Segment):
    def __init__(self, layer, num):
        layers = [copy.copy(layer) for i in range(0, num)]
        super(Repeat, self).__init__(layers=layers)