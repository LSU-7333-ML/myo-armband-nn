import os

from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model

save_dir = './saved'
h5_path = save_dir + '/my_model.h5'
his_path = save_dir + '/model_history'


def get_model(num_class, input_shape=(64,)):
    # Load saved model if exists
    if os.path.exists(h5_path):
        model = load_model(h5_path)
        print("-------------------Load model---------------------------")
    else:
        model = Sequential()
        model.add(Dense(128, input_shape=input_shape, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(num_class, activation="softmax"))
    model.summary()
    return model
