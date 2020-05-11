import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model
from keras.layers.core import Dense, Dropout, Activation

# loading the train data
X_train = np.load('train_image.npy')
y_train = np.load('train_labels.npy')
# loading the test data
X_test = np.load('test_image.npy')
y_test = np.load('test_labels.npy')

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# building a linear stack of layers with the sequential model
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model and saving metrics in history
history = model.fit(X_train, Y_train,batch_size=128, epochs=20,
          verbose=2,validation_data=(X_test, Y_test))

# saving the model
model_name = 'keras_mnist.h5'
model.save(model_name)
print('Saved trained model : %s ' % model_name)

# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

plt.show()

mnist_model = load_model('keras_mnist.h5')
loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose=2)

print()
print("Test Loss ->", loss_and_metrics[0])
print("Test Accuracy ->", loss_and_metrics[1])

