import unittest
import os

import numpy as np
import tensorflow as tf
import nlnnutil as nl

class Dataset_TextFiles(unittest.TestCase):
    def test_dataset_text_files(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../Tests/text")
        ds = nl.dataset.text_files(directory=path, glob='*.txt')

        with tf.Session() as sess:
            it = ds.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature['content']])

        self.assertEqual("AAA", data[0].decode().strip())


if __name__ == '__main__':
    unittest.main()
