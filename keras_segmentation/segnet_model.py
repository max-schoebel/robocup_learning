from keras.models import Sequential
# from keras.layers import Reshape
# from keras.layers import Merge
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.losses import mean_squared_error
# from keras.layers.convolutional import Convolution4D, MaxPooling3D, ZeroPadding3D
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
# from keras.layers.convolutional import Convolution1D, MaxPooling1D
# from keras.layers.recurrent import LSTM
# from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
# from keras.layers.embeddings import Embedding
# from keras.utils import np_utils
# from keras.regularizers import ActivityRegularizer
# from keras import backend as K
import tensorflow as tf

def segnet(nClasses, optimizer=None, input_height=360, input_width=480):
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    
    model = Sequential()
    model.add(Layer(input_shape=(input_height, input_width, 3)))
    
    # encoder
    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Conv2D(filter_size, (kernel, kernel), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    
    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Conv2D(128, (kernel, kernel), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    
    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Conv2D(256, (kernel, kernel), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    
    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Conv2D(512, (kernel, kernel), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # decoder
    # model.add(ZeroPadding2D(padding=(pad, pad)))
    # model.add(Conv2D(512, (kernel, kernel), padding='valid'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    
    model.add(UpSampling2D(size=(pool_size, pool_size)))
    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Conv2D(256, (kernel, kernel), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # model.add(UpSampling2D(size=(pool_size, pool_size)))
    # model.add(ZeroPadding2D(padding=(pad, pad)))
    # model.add(Conv2D(128, (kernel, kernel), padding='valid'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    
    # model.add(UpSampling2D(size=(pool_size, pool_size)))
    # model.add(ZeroPadding2D(padding=(pad, pad)))
    # model.add(Conv2D(filter_size, (kernel, kernel), padding='valid'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    
    model.add(Conv2D(nClasses, (1, 1), padding='valid'))
    
    # model.outputHeight = model.output_shape[-2]
    # model.outputWidth = model.output_shape[-1]
    
    # model.add(Reshape((model.output_shape[1] * model.output_shape[2], nClasses),
    #                   input_shape=(model.output_shape[1], model.output_shape[2], nClasses)))
    
    # model.add(Permute((2, 1)))
    # model.add(Activation('softmax'))
    
    if not optimizer is None:
        def custom_xentropy(y_true, y_pred):
            y_true = tf.argmax(y_true, axis=-1)
            return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        model.compile(loss=custom_xentropy, optimizer=optimizer, metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    import numpy as np
    from keras.losses import categorical_crossentropy
    # images = np.load("./pascal_images.npy")
    # labels = np.load("./pascal_labels.npy")
    
    model = segnet(459, Adam(), 256, 256)
    model.summary()
