import keras

MODEL_WEIGHT_FILE = 'weights-0871-0.0219-0.9895-0.0822-0.9398.hdf5'

model = keras.models.load_model(MODEL_WEIGHT_FILE)
model.summary()
