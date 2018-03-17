import numpy as np
import math

_vbars = " ▁▂▃▄▅▆▇█"

def bar_graph(data):
    if len(data) > 64:
        data = np.interp(np.linspace(0, len(data), 64),
                         np.arange(0, len(data)),
                         np.array(data))

    M = max(data)
    def _bar(alpha):
        if math.isnan(alpha):
            return 'N'
        else:
            n = int((len(_vbars) - 1) * max(0.0, min(1.0, alpha)))
            return _vbars[n]

    if M > 0:
        return ''.join([_bar(x/M) for x in data])
    else:
        return len(data) * ' '
