import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # do not print tf INFO messages

from functools import partial
import datetime

import tensorflow as tf
from tensorflow import keras as K
import tensorflow.keras.layers as L
import tensorflow.keras.losses as Lo
import tensorflow.keras.metrics as M
from tensorflow.keras.layers.experimental import preprocessing as P
from tensorflow.keras import optimizers as O
import numpy as np
import random
from ganomaly.mvtec.mvtec_ad import textures, objects, get_train_data, get_test_data
from ganomaly.networks.decoder import make_decoder
from ganomaly.networks.encoder import make_encoder
from tqdm import tqdm

def l1(true, pred):
    x = K.backend.reshape(true, (len(true), -1))
    y = K.backend.reshape(pred, (len(pred), -1))

    return K.backend.sum(K.backend.abs(x - y), axis=1)

def l2(true, pred):
    x = K.backend.reshape(true, (len(true), -1))
    y = K.backend.reshape(pred, (len(pred), -1))

    return K.backend.sum(K.backend.square(x - y), axis=1)

def gradient_penalty(discriminator, batch_size, real_images, fake_images):
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)

        pred = discriminator(interpolated, training=True)
    
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = K.backend.sqrt(K.backend.sum(K.backend.square(grads), axis=[1,2,3]))
    gp = K.backend.mean((norm - 1.0) ** 2)

    return gp

@tf.function
def train_step(images, encoder, generator, discriminator, \
                zdiscriminator, g_optimizer, discriminator_optimizer, \
                ge_optimizer, zd_optimizer, g_train, d_train
):

    batch_size = len(images)
    latent_size = generator.input.shape[1]
    bce = Lo.BinaryCrossentropy()
    
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    # latent = K.random_normal([batch_size, latent_size])

    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        D_real_result = discriminator(images, training=d_train)
        D_real_loss = K.backend.mean(D_real_result)
        # D_real_loss = bce(valid, D_real_result)

        z = K.backend.random_normal([batch_size, latent_size]) # z with mean=0, std=1
        x_fake = generator(z, training=g_train)
        D_fake_result = discriminator(x_fake, training=d_train)
        D_fake_loss = K.backend.mean(D_fake_result)
        # D_fake_loss = bce(fake, D_fake_result)
        #wgan-gp
        gp = gradient_penalty(discriminator, batch_size, images, x_fake)

        D_train_loss = D_fake_loss - D_real_loss + gp * 10
        G_train_loss = -K.backend.mean(D_fake_result)
        # G_train_loss = bce(valid, D_fake_result)
        
    if d_train:
        d_gradients = d_tape.gradient(D_train_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

    if g_train:
        g_gradients = g_tape.gradient(G_train_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

    # with tf.GradientTape() as g_tape:
    #     z = K.backend.random_normal([batch_size, latent_size]) # z with mean=0, std=1
    #     x_fake = generator(z, training=g_train)

    #     D_fake_result = discriminator(x_fake, training=False)
    #     G_train_loss = bce(valid, D_fake_result)

    # if g_train:
    #     g_gradients = g_tape.gradient(G_train_loss, generator.trainable_variables)
    #     g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

    with tf.GradientTape() as zd_tape:
        z_real = K.backend.random_normal([batch_size, latent_size]) # z with mean=0, std=1

        zd_real_result = zdiscriminator(z_real, training=True)
        zd_real_loss = bce(valid, zd_real_result)

        z_fake = encoder(images, training=True)
        zd_fake_result = zdiscriminator(z_fake, training=True)
        zd_fake_loss = bce(fake, zd_fake_result)

        zd_train_loss = zd_real_loss + zd_fake_loss
    
    zd_gradients = zd_tape.gradient(zd_train_loss, zdiscriminator.trainable_variables)
    zd_optimizer.apply_gradients(zip(zd_gradients, zdiscriminator.trainable_variables))

    with tf.GradientTape() as ge_tape:
        z = encoder(images, training=True)
        x_reconstructed = generator(z, training=True)

        zd_result = zdiscriminator(z, training=True)

        e_train_loss = bce(valid, zd_result)

        recon_loss = K.backend.mean(l2(images, x_reconstructed)) * 2

        ge_loss = e_train_loss + recon_loss

    ge_gradients = ge_tape.gradient(ge_loss, generator.trainable_variables + encoder.trainable_variables)
    ge_optimizer.apply_gradients(zip(ge_gradients, generator.trainable_variables + encoder.trainable_variables))
    

    return {
        'd_loss': D_train_loss,
        'g_loss': G_train_loss,
        'zd_loss': zd_train_loss,
        'recon_loss': recon_loss,
        'e_loss': e_train_loss,
        'd_fake_loss': D_fake_loss,
        'd_real_loss': D_real_loss

    }, x_reconstructed

def get_zd(latent_size, act):
    i = L.Input([latent_size])

    x = L.Dense(64)(i)
    x = act()(x)
    x = L.Dense(32)(x)
    x = act()(x)
    # x = L.Dense(16)(x)
    # x = act()(x)
    o = L.Dense(1, activation='sigmoid')(x)

    return K.Model(inputs=i, outputs=o)

def train():
    train_dataset = get_train_data('hazelnut', n_batches=1_200_000)
    test_dataset, test_labels = get_test_data('hazelnut')

    lrelu = partial(L.LeakyReLU, 0.2)
    encoder = make_encoder(128, 128, 3, act=lrelu)
    generator = make_decoder(128, 3)
    # encoder2 = make_encoder(128, 128, 3, act=lrelu)
    discriminator = make_encoder(128, 128, 3, act=lrelu, as_discriminator=True, bn=L.LayerNormalization)
    zdiscriminator = get_zd(128, lrelu)
    
    g_optimizer = O.Adam()
    d_optimizer = O.Adam()
    ge_optimizer = O.Adam()
    zd_optimizer = O.Adam()

    progress = tqdm(train_dataset, desc='hazelnut', dynamic_ncols=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    # test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for step, image_batch in enumerate(progress, start=1):
        # losses, images = train_step(image_batch, encoder1, generator, encoder2, discriminator, gee_optimizer, d_optimizer)
        if step == 1:
            g_train = True
            d_train = True
        else:
            g_train = step % 5 == 0
            d_train = True
            # d_train = step % 5 == 0

        # tf.summary.trace_on(graph=True)
        losses, images = train_step(
            image_batch,
            encoder,
            generator,
            discriminator,
            zdiscriminator,
            g_optimizer,
            d_optimizer,
            ge_optimizer,
            zd_optimizer,
            g_train,
            d_train
        )
        # print(images.shape)

        with train_summary_writer.as_default():
            for name, val in losses.items():
                if val:
                    tf.summary.scalar(name, val, step=step)

            # tf.summary.scalar('gee_loss', losses['l_tot'], step=step)
            # tf.summary.scalar('d_loss', losses['d_loss'], step=step)
            # tf.summary.scalar('d_loss_fake', losses['l_d_fake'], step=step)
            # tf.summary.scalar('d_loss_real', losses['l_d_real'], step=step)
            # tf.summary.scalar('gee_loss_con', losses['l_con'], step=step)
            # tf.summary.scalar('gee_loss_enc', losses['l_enc'], step=step)
            # tf.summary.scalar('gee_loss_adv', losses['l_adv'], step=step)

            if step % 50 == 0:
                # rescale = P.Rescaling(scale=127.5, offset=127.5)
                image_batch = (image_batch.numpy() + 1) / 2
                images = (images.numpy() + 1) / 2
                tf.summary.image('generated', images, step=step)
                tf.summary.image('real', image_batch, step=step)
            
                
        # g_train_loss(losses['l_tot'])
        # d_train_loss(losses['d_loss'])

    