import tensorflow as tf

batch_size = 1
img_layer = 3

class Generator:
    def __init__(self, name, ngf=64, norm='instance', image_size=128):
        self.name = name
        self.ngf = ngf
        self.norm = norm

    def __call__(self, input):
        """
        Args:
          input: batch_size x width x height x 3
        Returns:
          output: same size as input
        """
        with tf.variable_scope(self.name):
            f = 7
            ks = 3


            input = tf.pad(input,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
            o_c1 = self.conv2d(input, self.ngf, f, f, 1, 1, 0.02,name="c1")
            o_c2 = self.conv2d(o_c1, self.ngf*2, ks, ks, 2, 2, 0.02,"SAME","c2")
            o_c3 = self.conv2d(o_c2, self.ngf*4, ks, ks, 2, 2, 0.02,"SAME","c3")

            o_r1 = self.resnet(o_c3, self.ngf*4, "r1")
            o_r2 = self.resnet(o_r1, self.ngf*4, "r2")
            o_r3 = self.resnet(o_r2, self.ngf*4, "r3")
            o_r4 = self.resnet(o_r3, self.ngf*4, "r4")
            o_r5 = self.resnet(o_r4, self.ngf*4, "r5")
            o_r6 = self.resnet(o_r5, self.ngf*4, "r6")

            o_c4 = self.deconv2d(o_r6, [batch_size, 64, 64, self.ngf*2], self.ngf*2, ks, ks, 2, 2, 0.02,"SAME","c4")
            o_c5 = self.deconv2d(o_c4, [batch_size, 128, 128, self.ngf], self.ngf, ks, ks, 2, 2, 0.02,"SAME","c5")
            o_c5_pad = tf.pad(o_c5,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
            o_c6 = self.conv2d(o_c5_pad, img_layer, f, f, 1, 1, 0.02,"VALID","c6",do_relu=False)

            # Adding the tanh layer

            out_gen = tf.nn.tanh(o_c6,"t1")

            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return out_gen

    def conv2d(self, input, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="conv2d", do_norm=True, do_relu=True):
        with tf.variable_scope(name, reuse=False):
            conv = tf.contrib.layers.conv2d(input, o_d, f_w, s_w, padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=tf.constant_initializer(0.0))
            if do_norm:
                conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")
            if do_relu:
                conv = tf.nn.relu(conv,"relu")
            return conv

    def deconv2d(self, input, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="deconv2d", do_norm=True, do_relu=True):
        with tf.variable_scope(name, reuse=False):
            conv = tf.contrib.layers.conv2d_transpose(input, o_d, [f_h, f_w], [s_h, s_w], padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=tf.constant_initializer(0.0))
            if do_norm:
                conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")
            if do_relu:
                conv = tf.nn.relu(conv,"relu")
            return conv

    def resnet(self, input, dim, name="resnet"):
        with tf.variable_scope(name):
            out_res = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            out_res = self.conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID","c1")
            out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            out_res = self.conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID","c2",do_relu=False)
            return tf.nn.relu(out_res + input)

  #def sample(self, input):
    #image = utils.batch_convert2int(self.__call__(input))
    #image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
    #return image
