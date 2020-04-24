import tensorflow as tf

model = tf.keras.models.load_model('weights-0197-0.0045-0.9951-0.0076-1.0000.hdf5')
model.summary()
