import os

import numpy as np

import include.common as ic
import device

collect_dir = './collect/'


def collect(file_name, i_gesture):
    if not isinstance(i_gesture, int):
        raise TypeError('i_gesture should be index')
    c_len = len(ic.class_names)
    if i_gesture >= c_len:
        raise ValueError('i_gesture should less than ' + str(c_len))
    if not os.path.exists(collect_dir):
        os.makedirs(collect_dir)
    device.on_emg_data(save_to_file, args=(file_name, i_gesture), timeout=60)


def save_to_file(file_name, i_gesture, data):
    # temp_arr as x column: raw data
    x_arr = np.asarray(data, dtype=np.int64)
    # GestureType as y column: label
    gesture_type = np.zeros((x_arr.shape[0], len(ic.class_names)), dtype=np.int64)
    gesture_type[:, i_gesture] = 1  # change type here
    np.savez(collect_dir+file_name, x=x_arr, y=gesture_type)  # and change type here


collect('test.npz', 3)
