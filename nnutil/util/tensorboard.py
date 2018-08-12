import os
import subprocess

class Tensorboard:
    def __init__(self, path):
        self._path = os.path.abspath(path)
        self._port = 6006
        self._debugger = False

    def __enter__(self):

        args = ["tensorboard", "--port={}".format(self._port)]

        if self._debugger:
            args.append("--deugger_port={}".format(self._port + 1))

        args.append("--logdir={0}".format(self._path))

        self._tboard_proc = subprocess.Popen(args)
        # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return self

    def __exit__(self, type, value, traceback):
        if self._tboard_proc is not None:
            self._tboard_proc.terminate()
