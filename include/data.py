import numpy as np


# def get_data_set(path, num_class=6):
#     npzfile = np.load(path)
#     array_x = npzfile['x']
#     array_y = npzfile['y']
#     # np.random.shuffle(npzfile)
#     array_raw = np.append(array_x, array_y, axis=1)
#     np.random.shuffle(array_raw)
#     size = array_raw.shape[0]
#     train_data = array_raw[:int(size*0.8)]
#     test_data = array_raw[int(size*0.8):]
#     x_train = train_data[:, :-num_class]
#     y_train = train_data[:, -num_class:]
#     x_test = test_data[:, :-num_class]
#     y_test = test_data[:, -num_class:]
#
#     return (x_train, y_train), (x_test, y_test)


def get_data_set(path):
    npzfile = np.load(path)
    array_x = npzfile['x']
    y = npzfile['y']
    array_y = convert_y(y)
    # np.random.shuffle(npzfile)
    array_raw = np.append(array_x, array_y, axis=1)
    np.random.shuffle(array_raw)
    size = array_raw.shape[0]
    train_data = array_raw[:int(size*0.8)]
    test_data = array_raw[int(size*0.8):]
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
