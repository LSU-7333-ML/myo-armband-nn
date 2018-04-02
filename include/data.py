import numpy as np
from os import listdir, path


def get_data_set(f_path):
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    if path.isfile(f_path):
        npzfile = np.load(f_path)
        array_x = npzfile['x']
        y = npzfile['y']
        array_y = convert_y(y)
        array_raw = np.append(array_x, array_y, axis=1)
        np.random.shuffle(array_raw)
        size = array_raw.shape[0]
        train_data = array_raw[:int(size * 0.8)]
        test_data = array_raw[int(size * 0.8):]
        x_train = train_data[:, :-1]
        y_train = train_data[:, -1:]
        x_test = test_data[:, :-1]
        y_test = test_data[:, -1:]
    elif path.isdir(f_path):
        array_raw = []
        onlyfiles = [f for f in listdir(f_path) if
                     path.isfile(path.join(f_path, f)) and f.endswith(".npz")]
        for file in onlyfiles:
            npzfile = np.load(path.join(f_path, file))
            array_x = npzfile['x']
            y = npzfile['y']
            array_y = convert_y(y)
            if len(array_raw) == 0:
                array_raw = np.append(array_x, array_y, axis=1)
            else:
                array_raw = np.concatenate(
                        (array_raw, np.append(array_x, array_y, axis=1)))
        np.random.shuffle(array_raw)
        size = array_raw.shape[0]
        train_data = array_raw[:int(size * 0.8)]
        test_data = array_raw[int(size * 0.8):]
        x_train = train_data[:, :-1]
        y_train = train_data[:, -1:]
        x_test = test_data[:, :-1]
        y_test = test_data[:, -1:]

    return (x_train, y_train), (x_test, y_test)


def convert_y(data):
    result = []
    for i in range(data.shape[0]):
        result.append(np.where(data[i] == 1)[0][0])
    return np.array(result).reshape([len(result), 1])
