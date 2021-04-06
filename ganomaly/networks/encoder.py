from tensorflow import keras as K
from tensorflow.keras import layers as L

def e_block(x, n_filters, pool=True, use_bias=True, act=L.ReLU):
    skip = L.Conv2D(
        n_filters,
        kernel_size=1,
        use_bias=use_bias,
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

    if pool:
        # x = L.Conv2D(
        #     n_filters,
        #     kernel_size=1,
        #     use_bias=use_bias,
        #     padding='same',
        #     strides=2
        # )(x)
        x = L.AveragePooling2D()(x)

    return x   

# def make_encoder_ganomaly(image_size, latent_size, channels=3, act=L.ReLU, as_discriminator=False):
#     i = L.Input([image_size, image_size, channels]) # 128x128x3

#     x = L.Conv2D(
#         3,
#         kernel_size=4,
#         strides=2,
#         padding='same',
#         use_bias=False
#     )(i) # 64x64x64

#     x = act()(x)
    
#     x = L.Conv2D(
#         6,
#         kernel_size=4,
#         strides=2,
#         padding='same',
#         use_bias=False
#     )(x)
#     x = L.BatchNormalization()(x)
#     x = act()(x) # 32x32x128

#     x = L.Conv2D(
#         9,
#         kernel_size=4,
#         strides=2,
#         padding='same',
#         use_bias=False
#     )(x)
#     x = L.BatchNormalization()(x)
#     x = act()(x) # 16x16x256

#     x = L.Conv2D(
#         12,
#         kernel_size=4,
#         strides=2,
#         padding='same',
#         use_bias=False
#     )(x)
#     x = L.BatchNormalization()(x)
#     x = act()(x) # 8x8x512

#     x = L.Conv2D(
#         24,
#         kernel_size=4,
#         strides=2,
#         padding='same',
#         use_bias=False
#     )(x)
#     x = L.BatchNormalization()(x)
#     x = act()(x) # 4x4x1024

#     x = L.Conv2D(
#         48,
#         kernel_size=4,
#         strides=2,
#         padding='same',
#         use_bias=False
#     )(x)
#     x = L.BatchNormalization()(x)
#     x = act()(x) # 4x4x1024

#     if as_discriminator:
#         features = x
#         o = L.Conv2D(
#             1,
#             kernel_size=2,
#             use_bias=False,
#             activation='sigmoid'
#         )(x) # 1x1x1

#         o = L.Flatten()(o) # 1,

#         return K.Model(inputs=i, outputs=[features, o])


#     # x = L.Conv2D(
#     #     latent_size,
#     #     kernel_size=4,
#     #     use_bias=False
#     # )(x) # 1x1x100

#     x = L.Flatten()(x) # 100,
#     x = L.Dense(48)(x)
#     x = L.BatchNormalization()(x)
#     x = act()(x)

#     o = L.Dense(latent_size)(x)
    

#     return K.Model(inputs=i, outputs=o)    

# def make_encoder_test(image_size, latent_size, channels=3, act=L.ReLU, as_discriminator=False):
#     i = L.Input([image_size, image_size, channels])




# def make_encoder(*args, **kwargs):
#     return make_encoder_ganomaly(*args, **kwargs)
# def make_encoder(image_size, latent_size, channels=3, act=L.ReLU, as_discriminator=False):
#     i = L.Input([image_size, image_size, 3])

#     x = e_block(i, 1 * channels, act=act)
#     x = e_block(x, 2 * channels, act=act)
#     x = e_block(x, 3 * channels, act=act)
#     x = e_block(x, 4 * channels, act=act)
#     x = e_block(x, 8 * channels, act=act)
#     x = e_block(x, 16 * channels, act=act, pool=False)
    
#     if as_discriminator:
#         # features = x
#         o = L.Flatten()(x)
#         o = L.Dense(16 * channels)(o)
#         o = L.BatchNormalization()(o)
#         o = act()(o)
#         o = L.Dense(1, activation='sigmoid')(o)
        
#         return K.Model(inputs=i, outputs=o, name='discriminator')

#     x = L.Flatten()(x)

#     x = L.Dense(16 * channels)(x)
#     x = L.BatchNormalization()(x)
#     x = act()(x)
#     o = L.Dense(latent_size)(x)

#     return K.Model(inputs=i, outputs=o, name='encoder')

def encoder_block(x, filters, act):
    x = L.Conv2D(
        filters,
        kernel_size=3,
        strides=2,
        padding='same',
        use_bias=False,
        kernel_initializer='random_normal',
    )(x)

    x = L.BatchNormalization()(x)
    x = act()(x)

    return x

def make_encoder(image_size, latent_size, channels=3, act=L.ReLU, as_discriminator=False):
    filters = 6

    i = L.Input([image_size, image_size, channels]) #128x128x3
    x = encoder_block(i, filters, act) #64x64x64
    x = encoder_block(x, filters*2, act) #32x32x128
    x = encoder_block(x, filters*4, act) #16x16x256
    x = encoder_block(x, filters*8, act) #8x8x512
    x = encoder_block(x, filters*16, act) #4x4x1024

    x = L.Flatten()(x)
    x = L.Dense(16*filters, use_bias=False, kernel_initializer='random_normal')(x)
    x = L.BatchNormalization()(x)
    x = act()(x)

    if as_discriminator:
        o = L.Dense(1, activation='sigmoid', kernel_initializer='random_normal')(x)

    o = L.Dense(latent_size, kernel_initializer='random_normal')(x)

    return K.Model(inputs=i, outputs=o)
    # x = L.Conv2D(
    #     filters,
    #     kernel_size=4,
    #     strides=2,
    #     padding='same',
    #     use_bias=False
    # )(i)
    # x = L.BatchNormalization()(x)
    # x = act()(x)



if __name__ == '__main__':
    # encoder = make_encoder_ganomaly(128, 3)
    # encoder.summary()

    # disc = make_encoder_ganomaly(128, 3, as_discriminator=True)
    # disc.summary()
    encoder = make_encoder(128, 3)
    encoder.summary()

    discriminator = make_encoder(128, 3, as_discriminator=True)
    discriminator.summary()

