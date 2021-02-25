import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def CausalConvBlock(x):
    """
    2D Causal convolution.
    Args:
        x: [B, C, F, T]

    Returns:
        [B, C, F, T]
    """
    x=layers.Conv2D(1, kernel_size=(3, 2), strides=(2, 1), padding=(0, 1), name='conv')(x)
    return x

def CRN(x):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    e_1=CausalConvBlock(x)
    return e_1

if __name__=='__main__':
    layer=CRN
    # Create a tensor of shape [2, 1, 161, 200] consisting of random normal values
    a=tf.random.normal([2, 1, 161, 200])
    print(layer(a).shape)