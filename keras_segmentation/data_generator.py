# CODE taken from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, images, labels, batch_size, n_classes, image_dim, num_channels=3, shuffle=True):
        'Initialization'
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.image_dim = image_dim
        self.num_channels = num_channels
        self.num_examples = images.shape[0]
        self.on_epoch_end()
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_examples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_examples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y


    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.image_dim, self.num_channels), dtype=np.uint8)
        y = np.empty((self.batch_size, *self.image_dim, self.n_classes), dtype=np.bool)

        # Generate data
        for i, indx in enumerate(indexes):
            # Store sample
            X[i] = self.images[indx]

            # Store class
            for j, rowvec in enumerate(self.labels[indx]):
                y[i,j] = keras.utils.to_categorical(rowvec, num_classes=self.n_classes)

        return X, y

