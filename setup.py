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


from setuptools import setup, find_packages
from nnutil import __version__

setup(
    name             = 'nnutil',
    version          = __version__,
    license          = 'BSD',
    description      = 'Neural network utilities for tensorflow',
    author           = 'Abdó Roig-Maranges',
    author_email     = 'abdo.roig@gmail.com',
    packages         = find_packages(),
    install_requires = [
        'Click',
    ],
    entry_points     = '''
    [console_scripts]
    nnutil=nnutil.cli:main
    ''',
)
