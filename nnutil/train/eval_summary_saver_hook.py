
import tensorflow as tf
from tensorflow.python.training.summary_io import SummaryWriterCache


class EvalSummarySaverHook(tf.train.SessionRunHook):
    def __init__(self, output_dir, summary_op):
        self._output_dir = output_dir
        self._summary_op = summary_op
        self._summary_writer = None

    def begin(self):
        self._summary_writer = SummaryWriterCache.get(self._output_dir)

    def end(self, session):
        summary, global_step = session.run([self._summary_op, tf.train.get_global_step()])

        if self._summary_writer:
            self._summary_writer.add_summary(summary, global_step)
            self._summary_writer.flush()
