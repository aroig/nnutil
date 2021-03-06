#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# nnutil - Neural network utilities for tensorflow
# Copyright (c) 2018, Abdó Roig-Maranges <abdo.roig@gmail.com>
#
# This file is part of 'nnutil'.
#
# This file may be modified and distributed under the terms of the 3-clause BSD
# license. See the LICENSE file for details.


__version__      = '0.1'
__description__  = 'Neural network utilities for tensorflow'

from . import dataset

try:
    from . import visual
except:
    pass

from . import train
from . import model
from . import util
from . import layers
from . import summary
from . import math
from . import image

from tensorflow.python.framework.tensor_spec import TensorSpec
