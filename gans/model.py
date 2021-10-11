import tensorflow as tf
from tensorflow.keras import layers


print(f"Tensorflow Version: {tf.__version__}")



# GANs Model
"""
IMAGE DIM: 256*256*3: 196608

Assumption for Random Noise: Gaussian Distribution
INPUT DIMENSION: 64
"""

class Generator(tf.keras.Model):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dense1 = layers.Dense(256*256*3, name="gen_dense_layer")
        self.reshape = layers.Reshape(())
        

        

