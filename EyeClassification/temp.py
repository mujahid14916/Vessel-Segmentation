import tensorflow as tf

model = tf.keras.models.load_model('weights-0495-best.hdf5')
model.summary()

