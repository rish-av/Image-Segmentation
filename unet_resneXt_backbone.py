from keras import layers
from keras import models
import tensorflow as tf

img_height = 224
img_width = 224
img_channels = 3

cardinality = 32


def UNext(x):
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        return y

    def grouped_convolution(y, nb_channels, _strides):
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        shortcut = y
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = layers.BatchNormalization()(y)
        if _project_shortcut or _strides != (1, 1):
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])
        y = layers.LeakyReLU()(y)

        return y
    def build_resnext_32d(x):
        # conv1
        stored_output = []
        x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
        x = add_common_layers(x)
        stored_output.append(x)
        # conv2
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        for i in range(3):
            project_shortcut = True if i == 0 else False
            x = residual_block(x, 128, 256, _project_shortcut=project_shortcut)
        stored_output.append(x)
        # conv3
        for i in range(4):
            strides = (2, 2) if i == 0 else (1, 1)
            x = residual_block(x, 256, 512, _strides=strides)
        stored_output.append(x)
        # conv4
        for i in range(6):
            strides = (2, 2) if i == 0 else (1, 1)
            x = residual_block(x, 512, 1024, _strides=strides)
        stored_output.append(x)
        # conv5
        for i in range(3):
            strides = (2, 2) if i == 0 else (1, 1)
            x = residual_block(x, 1024, 2048, _strides=strides)
        return x,stored_output
    
    def build_upSample(resnext_out,stored_middle_layers):
        deconv1 = layers.Conv2DTranspose(1024,kernel_size=(1,1),dilation_rate=(2,2),strides=(2,2))(resnext_out)
        concat1 = layers.Concatenate()([stored_middle_layers[3],deconv1])
        add_common_layers(concat1)

        deconv2 = layers.Conv2DTranspose(512,kernel_size=(1,1),dilation_rate=(2,2),strides=(2,2))(concat1)
        concat2 = layers.Concatenate()([stored_middle_layers[2],deconv2])
        add_common_layers(concat2)

        deconv3 = layers.Conv2DTranspose(256,kernel_size=(1,1),dilation_rate=(2,2),strides=(2,2))(concat2)
        concat3 = layers.Concatenate()([stored_middle_layers[1],deconv3])
        add_common_layers(concat3)

        deconv4 = layers.Conv2DTranspose(128,kernel_size=(1,1),dilation_rate=(2,2),strides=(2,2))(concat3)
        concat4 = layers.Concatenate()([stored_middle_layers[0],deconv4])
        add_common_layers(concat4)

        final_upsample = layers.Conv2DTranspose(1,kernel_size =(1,1),strides=(2,2),activation='sigmoid')(concat4)
        return final_upsample
    final_layer,stored = build_resnext_32d(x)
    final_model = build_upSample(final_layer,stored)
    return final_model

def focal_loss(gamma=2., alpha=4.):
    
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed

img = layers.Input(shape = (img_height,img_width,img_channels))
out = UNext(img)
model = models.Model(input=[img],output=[out])
print(model.summary()) 