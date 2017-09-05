import tensorflow as tf

img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width

class CycleGAN():

    def init_model(self):
        self.input_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_A")
        self.input_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_B")

        self.pool_gen_A = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="pool_gen_A")
        self.pool_gen_B = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="pool_gen_B")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.num_gen_inputs = 0

        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")

        with tf.variable_scope("Model") as scope:
            self.gen_B = generator(self.input_A, name="g_A")
            self.gen_A = generator(self.input_B, name="g_B")
            self.disc_A = discriminator(self.input_A, "d_A")
            self.disc_B = discriminator(self.input_B, "d_B")

            scope.reuse_variables()

            self.disc_gen_A = discriminator(self.gen_A, "d_A")
            self.disc_gen_B = discriminator(self.gen_B, "d_B")
            self.cyc_A = generator(self.gen_B, "g_B")
            self.cyc_B = generator(self.gen_A, "g_A")

            scope.reuse_variables()

            self.disc_pool_gen_A = discriminator(self.pool_gen_A, "d_A")
            self.disc_pool_gen_B = discriminator(self.pool_gen_B, "d_B")

    def init_loss(self):

        ''' In this function we are defining the variables for loss calcultions and traning model
        d_loss_A/d_loss_B -> loss for discriminator A/B
        g_loss_A/g_loss_B -> loss for generator A/B
        *_trainer -> Variaous trainer for above loss functions
        *_summ -> Summary variables for above loss functions'''

        cyc_loss = tf.reduce_mean(tf.abs(self.input_A - self.cyc_A)) + tf.reduce_mean(tf.abs(self.input_B - self.cyc_B))

        disc_loss_A = tf.reduce_mean(tf.squared_difference(self.disc_gen_A, 1))
        disc_loss_B = tf.reduce_mean(tf.squared_difference(self.disc_gen_B, 1))

        g_loss_A = cyc_loss*10 + disc_loss_B
        g_loss_B = cyc_loss*10 + disc_loss_A

        d_loss_A = (tf.reduce_mean(tf.square(self.disc_pool_gen_A)) + tf.reduce_mean(tf.squared_difference(self.disc_A,1)))/2.0
        d_loss_B = (tf.reduce_mean(tf.square(self.disc_pool_gen_B)) + tf.reduce_mean(tf.squared_difference(self.disc_B,1)))/2.0


        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]

        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

        for var in self.model_vars: print(var.name)

        #Summary variables for tensorboard

        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)

    def generator(inputgen, name="generator"):
        with tf.variable_scope(name):
            f = 7
            ks = 3
            ngf = 64

            pad_input = tf.pad(inputgen,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
            o_c1 = general_conv2d(pad_input, ngf, f, f, 1, 1, 0.02,name="c1")
            o_c2 = general_conv2d(o_c1, ngf*2, ks, ks, 2, 2, 0.02,"SAME","c2")
            o_c3 = general_conv2d(o_c2, ngf*4, ks, ks, 2, 2, 0.02,"SAME","c3")

            o_r1 = build_resnet_block(o_c3, ngf*4, "r1")
            o_r2 = build_resnet_block(o_r1, ngf*4, "r2")
            o_r3 = build_resnet_block(o_r2, ngf*4, "r3")
            o_r4 = build_resnet_block(o_r3, ngf*4, "r4")
            o_r5 = build_resnet_block(o_r4, ngf*4, "r5")
            o_r6 = build_resnet_block(o_r5, ngf*4, "r6")

            o_c4 = general_deconv2d(o_r6, [batch_size,64,64,ngf*2], ngf*2, ks, ks, 2, 2, 0.02,"SAME","c4")
            o_c5 = general_deconv2d(o_c4, [batch_size,128,128,ngf], ngf, ks, ks, 2, 2, 0.02,"SAME","c5")
            o_c5_pad = tf.pad(o_c5,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
            o_c6 = general_conv2d(o_c5_pad, img_layer, f, f, 1, 1, 0.02,"VALID","c6",do_relu=False)

            # Adding the tanh layer

            out_gen = tf.nn.tanh(o_c6,"t1")


            return out_gen

    def discriminator(inputdisc, name="discriminator"):
        with tf.variable_scope(name):
            f = 4
            ndf = 64

            o_c1 = general_conv2d(inputdisc, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
            o_c2 = general_conv2d(o_c1, ndf*2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
            o_c3 = general_conv2d(o_c2, ndf*4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
            o_c4 = general_conv2d(o_c3, ndf*8, f, f, 1, 1, 0.02, "SAME", "c4",relufactor=0.2)
            o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)

            return o_c5

    def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="conv2d", do_norm=True, do_relu=True, relufactor=0):
        with tf.variable_scope(name):
            conv = tf.contrib.layers.conv2d(inputconv, o_d, f_w, s_w, padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=tf.constant_initializer(0.0))
            conv = tf.nn.relu(conv,"relu")
            return conv

    def general_deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="deconv2d", do_norm=True, do_relu=True, relufactor=0):
        with tf.variable_scope(name):
            conv = tf.contrib.layers.conv2d_transpose(inputconv, o_d, [f_h, f_w], [s_h, s_w], padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=tf.constant_initializer(0.0))
            conv = tf.nn.relu(conv,"relu")
            return conv
