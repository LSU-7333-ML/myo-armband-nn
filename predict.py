import numpy as np
import time

import include.model as im
import device

import include.common as ic

model = im.get_model()

start = time.time()


def print_predict_label(data):
    # predict every 3 second
    start = time.time()
    if not isinstance(data, (np.ndarray, np.generic)):
        data = np.asarray(data)
    class_pred = model.predict(data.reshape(1, data.shape[0]))
    # rounded = [round(x[0]) for x in class_pred]
    # print(rounded)
    # Predict class for test set gesture
    labels_pred = np.argmax(class_pred, axis=1)
    print(ic.class_names[labels_pred[0]])


device.on_emg_data(print_predict_label)
