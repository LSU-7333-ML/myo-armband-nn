import matplotlib.pyplot as plt
import numpy as np
import os
import include.model as im
import pickle
from keras.callbacks import History


# Model accuracy and loss plots
def plot_model(model_details):
    # Create sub-plots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Summarize history for accuracy
    axs[0].plot(range(1, len(model_details.history['acc']) + 1),
                model_details.history['acc'])
    axs[0].plot(range(1, len(model_details.history['val_acc']) + 1),
                model_details.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_details.history['acc']) + 1),
                      len(model_details.history['acc']) / 10)
    axs[0].legend(['train', 'val'], loc='best')

    # Summarize history for loss
    axs[1].plot(range(1, len(model_details.history['loss']) + 1),
                model_details.history['loss'])
    axs[1].plot(range(1, len(model_details.history['val_loss']) + 1),
                model_details.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_details.history['loss']) + 1),
                      len(model_details.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


# Plot saved model history
if os.path.exists(im.his_path):
    # Save model history
    with open(im.his_path, 'rb') as file_pi:
        model_history = History()
        model_history.history = pickle.load(file_pi)
        plot_model(model_history)
else:
    raise ValueError('No model history found')
