import time

import tensorflow as tf
import numpy as np

from tensorflow.python.data.util import nest
from tensorflow.python.ops import gen_dataset_ops

def benchmark_dataset(dataset):
    with tf.Session() as sess:
        # feature = tf.contrib.data.get_single_element(dataset)

        it = dataset.make_initializable_iterator()
        sess.run(it.initializer)

        feature = it.get_next()

        # feature = gen_dataset_ops.dataset_to_single_element(
        #     dataset._as_variant_tensor(),
        #     output_types=nest.flatten(dataset.output_types),
        #     output_shapes=nest.flatten(dataset.output_shapes))

        count = 0
        last = time.time()
        while True:
            flat = nest.flatten(feature)
            flat_val = sess.run(flat)
            count += 1
            now = time.time()
            if now - last > 1.0 and count > 10:
                print("batches / s: {}".format(float(count) / (now - last)))
                last = now
                count = 0
