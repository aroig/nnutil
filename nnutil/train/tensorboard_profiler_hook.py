import os

import tensorflow as tf

from tensorflow.python.client import timeline

class TensorboardProfilerHook(tf.train.ProfilerHook):
    """Extends the profiler hook to also write summaries for tensorboard"""

    def __init__(self, save_secs, output_dir):
        self._output_dir = output_dir
        self._summary_writer = tf.summary.FileWriter(output_dir)
        super().__init__(save_secs=save_secs,
                         show_dataflow=True,
                         show_memory=True,
                         output_dir=output_dir)

    def after_run(self, run_context, run_values):
        if self._request_summary:
            global_step = run_context.session.run(self._global_step_tensor)
            self._summary_writer.add_run_metadata(run_values.run_metadata,
                                                  's{}'.format(global_step))

            # Generate chrome trace
            tl = timeline.Timeline(run_values.run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            if os.path.exists(self._output_dir):
                path = os.path.join(self._output_dir, 'timeline-{}.json')
                with open(path, 'w') as f:
                    f.write(ctf)

        super().after_run(run_context, run_values)
