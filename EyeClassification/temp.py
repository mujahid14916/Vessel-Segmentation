import tensorflow as tf

model = tf.keras.models.load_model('new_model/weights-0274-0.0639-0.9749-0.2027-0.9703.hdf5')
model.summary()

