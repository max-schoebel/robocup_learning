from keras.applications.xception import Xception
from keras.layers.convolutional import ZeroPadding2D, Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.models import Model, Sequential
from keras import losses
from keras.optimizers import Adam
import tensorflow as tf


def custom_xentropy(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

def xception_segmenter(input_shape, num_classes, optimizer):
    kernel = (3,3)
    pad = 1
    pool_size = 2
    
    base_model = Xception(include_top=False, input_shape=input_shape, classes=num_classes)
    base_output = base_model.output
    base_input = base_model.input
    
    for layer in  base_model.layers:
        layer.trainable = False
        
    # output of last layer here is (None, 8, 8, 2048) for input size (256,256)
    x = UpSampling2D(size=(pool_size, pool_size))(base_output)
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Conv2D(num_classes, kernel, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Conv2D(num_classes, kernel, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Conv2D(num_classes, kernel, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Conv2D(num_classes, kernel, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Conv2D(num_classes, kernel, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = Conv2D(num_classes, (1, 1), padding='valid')(x)
    # x = Activation('softmax')(x_preactivation)
    
    model = Model(inputs=base_input, outputs= x)# [x, x_preactivation])
    model.compile(optimizer=optimizer, loss=custom_xentropy, metrics=['accuracy'])
    return model
    
if __name__ == '__main__':
    import numpy as np
    from keras.losses import categorical_crossentropy
    images = np.load("./pascal_images.npy")
    labels = np.load("./pascal_labels.npy")
    model = xception_segmenter((256, 256, 3), 459, Adam())
    model.summary()
    
    out = model.predict(images[0:1])