import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# loading the test data
X_test = np.load('test_image.npy')
Y_test = np.load('test_labels.npy')
# load the model and create predictions on the test set
model = load_model('keras_mnist.h5')
Y_predicted = model.predict_classes(X_test)

# see which we predicted correctly and which not
correct_indices = np.nonzero(Y_predicted == Y_test)[0]
incorrect_indices = np.nonzero(Y_predicted != Y_test)[0]
print()
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")


plt.figure(figsize=(7,10))

# plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted: {}, Truth: {}".format(Y_predicted[correct],Y_test[correct]))
    plt.xticks([])
    plt.yticks([])

# plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6,3,i+10)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Truth: {}".format(Y_predicted[incorrect],Y_test[incorrect]))
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()