import tensorflow as tf

# Classic loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Weissenberg loss

def reconstruction_loss(real, fake_rec):
    """ Lrec = || G(z*) - real ||^2 """
    return tf.reduce_mean(tf.square(fake_rec - real))

def generator_wgan_loss(discriminator, fake):
    """ Ladv(G) = -E[D(fake)] """
    return -tf.reduce_mean(discriminator(fake))

def discriminator_wgan_loss(discriminator, real, fake, batch_size=1):
    """ Ladv(D) = E[D(fake)] - E[D(real)] + GradientPenalty"""
    dis_loss = tf.reduce_mean(discriminator(fake)) - tf.reduce_mean(discriminator(real))
    alpha = tf.random.uniform(shape=[real.get_shape().as_list()[0], 1, 1, 1], minval=0., maxval=1.)  # real.shape
    interpolates = alpha * real + ((1 - alpha) * fake)
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        dis_interpolates = discriminator(interpolates)
    gradients = tape.gradient(dis_interpolates, [interpolates])[0]

    slopes = tf.sqrt(
        tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))  # compute pixelwise gradient norm; per image use [1, 2, 3]
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

    dis_loss = dis_loss + 10 * gradient_penalty
    return dis_loss