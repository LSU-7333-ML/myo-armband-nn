import numpy as np
from os import listdir, path


def get_data_set(separate_test=False):
    """
    Get data to be trained
    :param separate_test: whether test data is completely out from train data
    :return: data
    """
    train_dir = "./data/train"
    if not separate_test:
        array_raw = []
        only_files = [f for f in listdir(train_dir) if
                      path.isfile(path.join(train_dir, f)) and f.endswith(
                          ".npz")]
        for file in only_files:
            npzfile = np.load(path.join(train_dir, file))
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
    else:
        train_data = []
        only_files = [f for f in listdir(train_dir) if
                      path.isfile(path.join(train_dir, f)) and f.endswith(
                          ".npz")]
        for file in only_files:
            npzfile = np.load(path.join(train_dir, file))
            array_x = npzfile['x']
            y = npzfile['y']
            array_y = convert_y(y)
            if len(train_data) == 0:
                train_data = np.append(array_x, array_y, axis=1)
            else:
                train_data = np.concatenate(
                        (train_data, np.append(array_x, array_y, axis=1)))
        np.random.shuffle(train_data)
        x_train = train_data[:, :-1]
        y_train = train_data[:, -1:]

        test_dir = "./data/test"
        test_data = []
        only_files = [f for f in listdir(test_dir) if
                      path.isfile(path.join(test_dir, f)) and f.endswith(
                          ".npz")]
        for file in only_files:
            npzfile = np.load(path.join(test_dir, file))
            array_x = npzfile['x']
            y = npzfile['y']
            array_y = convert_y(y)
            if len(test_data) == 0:
                test_data = np.append(array_x, array_y, axis=1)
            else:
                test_data = np.concatenate(
                        (test_data, np.append(array_x, array_y, axis=1)))
        np.random.shuffle(test_data)
        x_test = test_data[:, :-1]
        y_test = test_data[:, -1:]

    return (x_train, y_train), (x_test, y_test)


def convert_y(data):
    result = []
    for i in range(data.shape[0]):
        result.append(np.where(data[i] == 1)[0][0])
    return np.array(result).reshape([len(result), 1])
