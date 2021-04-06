from re import U
from tensorflow import keras as K
from tensorflow.keras import layers as L

def d_block(x, n_filters, upsample=True, upsample_type='bilinear', use_bias=True, act=L.ReLU):
    if upsample:
        if upsample_type == 'bilinear':
            x = L.UpSampling2D()(x)
        else:
            x = L.Conv2DTranspose(
                n_filters,
                kernel_size=1,
                strides=2
            )(x)

    skip = L.Conv2D(
        n_filters,
        kernel_size=1,
        padding='same'
    )(x)
    skip = L.BatchNormalization()(skip)

    x = L.Conv2D(
        n_filters,
        kernel_size=3,
        use_bias=use_bias,
        padding='same'
    )(x)
    x = L.BatchNormalization()(x)
    x = act()(x)

    x = L.Conv2D(
        n_filters,
        kernel_size=3,
        use_bias=use_bias,
        padding='same'
    )(x)
    x = L.BatchNormalization()(x)
    x = act()(x)

    x = L.Conv2D(
        n_filters,
        kernel_size=1,
        padding='same'
    )(x)

    x = L.Add()([x, skip])
    x = L.BatchNormalization()(x)
    x = act()(x)


    return x

# def make_decoder_ganomaly(latent_size, channels=3, upsample_first = True, upsample_type='bilinear', act=L.ReLU):
#     i = L.Input([latent_size]) # 100,
    
#     x = L.Dense(2 * 2 * 48, use_bias=False)(i)
#     x = L.BatchNormalization()(x)
#     x = L.Reshape([2, 2, 48])(x)

#     x = L.Conv2DTranspose(
#         48,
#         kernel_size=4,
#         padding='same',
#         use_bias=False
#     )(x)
#     x = L.BatchNormalization()(x)
#     x = act()(x)

#     x = L.Conv2DTranspose(
#         24,
#         kernel_size=4,
#         strides=2,
#         padding='same',
#         use_bias=False
#     )(x)
#     x = L.BatchNormalization()(x)
#     x = act()(x)

#     x = L.Conv2DTranspose(
#         12,
#         kernel_size=4,
#         strides=2,
#         padding='same',
#         use_bias=False
#     )(x)
#     x = L.BatchNormalization()(x)
#     x = act()(x)

#     x = L.Conv2DTranspose(
#         9,
#         kernel_size=4,
#         strides=2,
#         padding='same',
#         use_bias=False
#     )(x)
#     x = L.BatchNormalization()(x)
#     x = act()(x)

#     x = L.Conv2DTranspose(
#         6,
#         kernel_size=4,
#         strides=2,
#         padding='same',
#         use_bias=False
#     )(x)
#     x = L.BatchNormalization()(x)
#     x = act()(x)

#     x = L.Conv2DTranspose(
#         3,
#         kernel_size=4,
#         strides=2,
#         padding='same',
#         use_bias=False
#     )(x)
#     x = L.BatchNormalization()(x)
#     x = act()(x)

#     x = L.Conv2DTranspose(
#         3,
#         kernel_size=1,
#         strides=2,
#         padding='same'
#     )(x)
#     x = L.BatchNormalization()(x)
#     x = act()(x)

#     o = L.Conv2D(
#         3,
#         kernel_size=1,
#         padding='same',
#         activation='tanh'
#     )(x)
#     # o = L.Conv2DTranspose(
#     #     3,
#     #     kernel_size=4,
#     #     padding='same',
#     #     use_bias=False,
#     #     activation='tanh'
#     # )(x)

#     return K.Model(inputs=i, outputs=o)


# def make_decoder(*args, **kwargs):
#     return make_decoder_ganomaly(*args, **kwargs)
# def make_decoder(latent_size, channels=3, upsample_first = True, upsample_type='bilinear', act=L.ReLU):
#     i = L.Input([latent_size])
#     x = L.Dense(2 * 2 * 16 * channels, use_bias=False)(i)
#     x = L.BatchNormalization()(x)
#     x = L.Reshape([2, 2, 16 * channels])(x)

#     x = d_block(x, 16 * channels, upsample=upsample_first, upsample_type=upsample_type, act=act)
#     x = d_block(x, 8 * channels, upsample_type=upsample_type, act=act)
#     x = d_block(x, 4 * channels, upsample_type=upsample_type, act=act)
#     x = d_block(x, 3 * channels, upsample_type=upsample_type, act=act)
#     x = d_block(x, 2 * channels, upsample_type=upsample_type, act=act)
#     x = d_block(x, 1 * channels, upsample_type=upsample_type, act=act)

#     o = L.Conv2D(3, kernel_size=1, activation='tanh', padding='same')(x)

#     return K.Model(inputs=i, outputs=o, name='decoder')
def generator_block(x, filters, act):
    # x = L.Conv2DTranspose(
    #     filters,
    #     kernel_size=4,
    #     strides=2,
    #     padding='same',
    #     use_bias=False
    # )(x)
    x = L.UpSampling2D()(x)
    x = L.Conv2D(
        filters,
        kernel_size=3,
        padding='same',
        use_bias=False,
        kernel_initializer='random_normal',
    )(x)
    x = L.BatchNormalization()(x)
    x = act()(x)

    return x

def make_decoder(latent_size, channels=3, act=L.ReLU):
    filters = 6

    i = L.Input([latent_size])

    x = L.Dense(16*filters, use_bias=False, kernel_initializer='random_normal',)(i)
    x = L.Dense(4*4*16*filters, use_bias=False, kernel_initializer='random_normal',)(x)
    x = L.Reshape([4,4,16*filters])(x)

    x = generator_block(x, 16*filters, act)
    x = generator_block(x, 8*filters, act)
    x = generator_block(x, 4*filters, act)
    x = generator_block(x, 2*filters, act)
    x = generator_block(x, filters, act)

    x = L.Conv2D(
        channels,
        3,
        padding='same',
        use_bias=False,
        activation='tanh',
        kernel_initializer='random_normal',
    )(x)

    return K.Model(inputs=i, outputs=x)

if __name__ == '__main__':
    d = make_decoder(3)
    d.summary()

