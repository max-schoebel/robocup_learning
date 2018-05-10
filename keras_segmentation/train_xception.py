from xception_model import xception_segmenter
from data_generator import DataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
import numpy as np
from xception_model import custom_xentropy

IMG_SIZE = (256,256)
NUM_CHANNELS = 3
NET_INPUT = (IMG_SIZE[0], IMG_SIZE[1], NUM_CHANNELS)
NUM_CLASSES = 459

opt = Adam()
model = xception_segmenter(NET_INPUT, NUM_CLASSES, opt)
model.summary()

images = np.load("./pascal_images.npy")
labels = np.load("./pascal_labels.npy") -1

dg_params = {'images': images,
             'labels': labels,
             'batch_size': 2,
             'n_classes': NUM_CLASSES,
             'image_dim': IMG_SIZE,
             'num_channels': NUM_CHANNELS
            }

dg = DataGenerator(**dg_params)
check_cb = ModelCheckpoint('./checkpoint{epoch:02d}.hdf5')
tensor_cb = TensorBoard()
# model.fit(images, labels, epochs=10, batch_size=2)
model.fit_generator(generator=dg,
                    use_multiprocessing=True,
                    workers=6,
                    epochs=100,
                    callbacks=[check_cb, tensor_cb])
