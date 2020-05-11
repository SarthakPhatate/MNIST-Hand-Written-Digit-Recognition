# imports for array-handling and plotting
import numpy as np
import matplotlib.pyplot as plt

# keras imports for the dataset
from keras.datasets import mnist

#load the dataset using this handy function which splits the MNIST data into train and test sets.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(X_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()

# let's print the shape before we reshape and normalize
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)

print('Saving the train and test data!')
# Saving Train data & label
np.save('train_image.npy', X_train)
np.save('train_labels.npy', y_train)
# Saving test data & label
np.save('test_image.npy', X_test)
np.save('test_labels.npy', y_test)