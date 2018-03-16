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


import unittest


class TestCase1(unittest.TestCase):

    def setUp(self):
        pass

    def test_match(self):
        self.assertEqual(1,1)


if __name__ == '__main__':
    unittest.main()
