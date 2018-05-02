from keras.applications.xception import Xception

model = Xception(input_shape=(256,256,3), include_top=False)
model.summary()
