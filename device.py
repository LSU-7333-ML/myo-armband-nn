import threading
import time

import collections
import numpy as np

import myo


class MyListener(myo.DeviceListener):

    def __init__(self, queue_size=8):
        self.lock = threading.Lock()
        self.emg_data_queue = collections.deque(maxlen=queue_size)

    def on_connect(self, device, timestamp, firmware_version):
        device.set_stream_emg(myo.StreamEmg.enabled)

    def on_emg_data(self, device, timestamp, emg_data):
        with self.lock:
            self.emg_data_queue.append((timestamp, emg_data))

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)


def on_emg_data(fun, args=None, timeout=-1):
    """
    Deal with the emg data
    :param fun: function to handle the data
    :param args: function arguments
    :param timeout: if -1 never timeout.
    :return: None
    """
    # open device
    # myo.init(r"C:\Users\ShengfengLi\myo-sdk-win-0.9.0\bin")
    myo.init()
    hub = myo.Hub()
    start = time.time()
    whole_data = []
    try:
        listener = MyListener()
        hub.run(2000, listener)
        while True:
            data = listener.get_emg_data()
            # Collect data for 60 seconds
            if timeout >= 0 and time.time() - start >= float(timeout):
                # Call non real time function
                if args is None:
                    fun(whole_data)
                else:
                    fun(*args, whole_data)
                break
            if len(data) > 0:
                tmp = []
                for v in data:
                    tmp.append(v[1])
                    if len(tmp) >= 2:                        
                        tmp = list(np.stack(tmp).flatten())
                        # push 64-value array with data from each sensor
                        if len(tmp) >= 16:
                            if timeout < 0:
                                # Call real time function
                                if args is None:
                                    fun(tmp)
                                else:
                                    fun(*args, tmp)
                            else:
                                whole_data.append(tmp)
                        tmp = []
            time.sleep(0.01)
    finally:
        hub.shutdown()
