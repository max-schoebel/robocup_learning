import os
from keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

def custom_xentropy(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

model = load_model('segnet_checkpoint69.hdf5', custom_objects={'custom_xentropy' : custom_xentropy})
PLOTS = 10

def test_on_athome():
    path = "/home/max/data/robocup/scaled_athome_images"
    for _ in range(PLOTS):
        folder = np.random.randint(1,32)
        imagenames = os.listdir(os.path.join(path, str(folder)))
        img_int = np.random.randint(len(imagenames))
        rand_img = imagenames[img_int]
        image = cv2.imread(os.path.join(path, str(folder), rand_img))
        image = cv2.resize(image, (256, 256))
        image_expanded = np.expand_dims(image, 0)
        test_prediction = model.predict(image_expanded)

        plt.subplot(121), plt.imshow(image)
        plt.subplot(122), plt.imshow(np.argmax(test_prediction[0], axis=2))
        plt.show()
        
def test_on_training_data():
    images = np.load("./pascal_images.npy")
    labels = np.load("./pascal_labels.npy")
    for _ in range(PLOTS):
        img_int = np.random.randint(len(images))
        image = images[img_int:img_int+1]
        test_prediction = model.predict(image)

        plt.subplot(131), plt.imshow(image[0])
        plt.subplot(132), plt.imshow(np.argmax(test_prediction[0], axis=2))
        plt.subplot(133), plt.imshow(labels[img_int])
        plt.show()
        

if __name__ == "__main__":
    test_on_training_data()
    # test_on_athome()
