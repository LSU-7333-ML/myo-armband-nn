import pickle

import os
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix

import include.model as im
from include.common import class_names
from include.data import get_data_set

# Obtain Data from path. Separate it by train 80% and test 20%
(x_train, y_train), (x_test, y_test) = get_data_set()
num_class = len(np.unique(y_train))

_BATCH_SIZE = 300

model = im.get_model(num_class)

# Compile model
model.compile(optimizer=RMSprop(lr=1.0e-4),
              # RMSprop optimizer with 1.0e-4 learning rate
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # Metrics to be evaluated by the model

checkpoint = ModelCheckpoint(im.h5_path,  # model filename
                             monitor='val_loss',  # quantity to monitor
                             verbose=0,  # verbosity - 0 or 1
                             save_best_only=True,
                             # The latest best model will not be overwritten
                             mode='auto')  # The decision to overwrite model

earlyStopping = EarlyStopping(monitor="val_loss", patience=100, verbose=0,
                              mode='auto')

model_details = model.fit(x_train, y_train,
                          batch_size=_BATCH_SIZE,
                          # number of samples per gradient update
                          epochs=10000,  # number of iterations
                          validation_data=(x_test, y_test),
                          callbacks=[checkpoint, earlyStopping],
                          verbose=1)

# if not os.path.exists(im.save_dir):
#    os.makedirs(im.save_dir)

# Save model
model.save(im.h5_path)

# Save model history
with open(im.his_path, 'w+b') as file_pi:
    pickle.dump(model_details.history, file_pi)

# Evaluate the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Evaluate on test, accuracy: %.2f%%" % (scores[1] * 100) + '\n')

# Predictions
class_pred = model.predict(x_test, batch_size=32)
# print(class_pred[0])

# Predict class for test set gesture
labels_pred = np.argmax(class_pred, axis=1)
# print(labels_pred)

# Check which labels have been predicted correctly
correct = (labels_pred == y_test[:, 0]).astype('int32')
# print(correct)
print("Number of correct predictions: %d" % sum(correct) + "\n")

# Print metrics
cm = confusion_matrix(y_true=y_test[:, 0], y_pred=labels_pred)
# for i in range(6):
#     class_name = "({}) {}".format(i, class_names[i])
#     print(cm[i, :], class_name)
# class_numbers = [" ({0})".format(i) for i in range(6)]
# print("  ".join(class_numbers))
row_format = '{:>5}' * (len(class_names) + 1)
print(row_format.format(*class_names, " "))
for label, row, i in zip(class_names, cm, range(len(class_names))):
    print(row_format.format(*row, " ({}){}".format(i, label)))

# Calculate accuracy using manual calculation
num_gesture = len(correct)
print("\nPredict Accuracy: %.2f%%" % ((sum(correct) * 100) / num_gesture))
