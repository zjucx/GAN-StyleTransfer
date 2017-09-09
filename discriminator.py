import tensorflow as tf

class Discriminator:
  def __init__(self, name, ngf=64, norm='instance', image_size=128):
    self.name = name
    self.ngf = ngf
    self.norm = norm
    self.image_size = image_size

  def __call__(self, input):
    """
    Args:
      input: batch_size x width x height x 3
    Returns:
      output:
    """
    with tf.variable_scope(name):
        f = 4

        o_c1 = = self.conv2d(inputdisc, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
        o_c2 = = self.conv2d(o_c1, ndf*2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = = self.conv2d(o_c2, ndf*4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = = self.conv2d(o_c3, ndf*8, f, f, 1, 1, 0.02, "SAME", "c4",relufactor=0.2)
        o_c5 = = self.conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return o_c5

    def conv2d(input, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="conv2d", do_norm=True, do_relu=True):
        with tf.variable_scope(name):
            conv = tf.contrib.layers.conv2d(input, o_d, f_w, s_w, padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=tf.constant_initializer(0.0))
            if do_norm:
                conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")
            if do_relu:
                conv = tf.nn.relu(conv,"relu")
            return conv

  #def sample(self, input):
    #image = utils.batch_convert2int(self.__call__(input))
    #image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
    #return image
