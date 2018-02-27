import collections
import myo
import threading
import time
import numpy as np
#import tensorflow as tf
#from include.model import model

import pandas as pd

#sess = tf.Session()
"""
x, y, output, global_step, y_pred_cls = model()

saver = tf.train.Saver()
_SAVE_PATH = "./data/tensorflow_sessions/myo_armband/"



try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    print(last_chk_path)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())
"""

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
        

myo.init()
hub = myo.Hub()
start = time.time()
temp = []
try:
    listener = MyListener()
    hub.run(2000, listener)
    while True:
        data = listener.get_emg_data()
        #data collection more than 200s
        if time.time() - start >= 200:
            temp = []
            start = time.time()
        if len(data) > 0:
            tmp = []
            for v in listener.get_emg_data():
                tmp.append(v[1])
            tmp = list(np.stack(tmp).flatten())
            #push 64-value array with data from each sensor
            if len(tmp) >= 64:
                temp.append(tmp)
                #temp_arr as x column: raw data
                temp_arr =np.asarray(temp, dtype=np.int64)
                # GestureType as y column: label
                GestureType = np.zeros((temp_arr.shape[0], 6), dtype=np.int64)
                GestureType_1 = np.ones(temp_arr.shape[0], dtype=np.int64)
                GestureType[:, 5] = GestureType_1 # change type here
                # 0 - relax, 1 - fist, 2 - roll, 3 - hold, 4 - drag, 5 - rock
                np.savez('wei_5.npz', x = temp_arr, y = GestureType) # and change type here
        time.sleep(0.01)

finally:
    hub.shutdown()
    #sess.close()
"""
myo.init()
hub = myo.Hub()
start = time.time()
temp = []
try:
    listener = MyListener()
    hub.run(2000, listener)
    while True:
        data = listener.get_emg_data()
        if time.time() - start >= 1 and len(tmp) >= 64:
          
            temp.append(tmp)
            tmp = []                        
            start = time.time()
        if len(data) > 0:
            tmp = []
            for v in listener.get_emg_data():
                tmp.append(v[1])
            tmp = list(np.stack(tmp).flatten())              
        
        #temp_array =np.asanyarray(temp)
        print(temp)
        #with open('champ_rest_1.csv', 'w', newline='') as myfile:
            #wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            #wr.writerows(zip(temp))       
        time.sleep(0.01)
"""
