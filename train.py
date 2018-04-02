import pickle

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop

import include.model as im
from include.data import get_data_set

# Obtain Data from path. Separate it by train 80% and test 20%
(x_train, y_train), (x_test, y_test) = get_data_set("./data/")
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

earlyStopping = EarlyStopping(monitor="val_loss", patience=5, verbose=0,
                              mode='auto')

model_details = model.fit(x_train, y_train,
                          batch_size=_BATCH_SIZE,
                          # number of samples per gradient update
                          epochs=10000,  # number of iterations
                          validation_data=(x_test, y_test),
                          callbacks=[checkpoint, earlyStopping],
                          verbose=1)

# Save model
model.save(im.h5_path)

# Save model history
with open(im.his_path, 'w+b') as file_pi:
    pickle.dump(model_details.history, file_pi)

# Evaluate the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

# Predictions
class_pred = model.predict(x_test, batch_size=32)
print(class_pred[0])

# Predict class for test set gesture
labels_pred = np.argmax(class_pred, axis=1)
print(labels_pred)

# Check which labels have been predicted correctly
correct = (labels_pred == y_test[:, 0]).astype('int32')
print(correct)
print("Number of correct predictions: %d" % sum(correct))

# Calculate accuracy using manual calculation
num_gesture = len(correct)
print("Accuracy: %.2f%%" % ((sum(correct) * 100) / num_gesture))

# x, y, output, global_step, y_pred_cls = model(_CLASS_SIZE)
#
#
# loss = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
# tf.summary.scalar("Loss", loss)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(loss,
#
# global_step=global_step)
#
#
# correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, dimension=1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# tf.summary.scalar("Accuracy/train", accuracy)
#
#
# init = tf.global_variables_initializer()
# merged = tf.summary.merge_all()
# saver = tf.train.Saver()
# sess = tf.Session()
# train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)
#
#
# try:
#     print("Trying to restore last checkpoint ...")
#     last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
#     saver.restore(sess, save_path=last_chk_path)
#     print("Restored checkpoint from:", last_chk_path)
# except:
#     print("Failed to restore checkpoint. Initializing variables instead.")
#     sess.run(tf.global_variables_initializer())
#
#
# def train(num_iterations = 1000):
#     for i in range(num_iterations):
#         randidx = np.random.randint(len(train_x), size=_BATCH_SIZE)
#         batch_xs = train_x[randidx]
#         batch_ys = train_y[randidx]
#
#         start_time = time()
#         i_global, _ = sess.run([global_step, optimizer], feed_dict={x:
# batch_xs, y: batch_ys})
#         duration = time() - start_time
#
#         if (i_global % 10 == 0) or (i == num_iterations - 1):
#             _loss, batch_acc = sess.run([loss, accuracy], feed_dict={x:
# batch_xs, y: batch_ys})
#             msg = "Global Step: {0:>6}, accuracy: {1:>6.1%}, loss = {
# 2:.2f} ({3:.1f} examples/sec, {4:.2f} sec/batch)"
#             print(msg.format(i_global, batch_acc, _loss, _BATCH_SIZE /
# duration, duration))
#
#         if (i_global % 100 == 0) or (i == num_iterations - 1):
#             data_merged, global_1 = sess.run([merged, global_step],
# feed_dict={x: batch_xs, y: batch_ys})
#             train_writer.add_summary(data_merged, global_1)
#             saver.save(sess, save_path=_SAVE_PATH, global_step=global_step)
#             print("Saved checkpoint.")
#
#
# train(75000)
#
#
# sess.close()
