import tensorflow as tf

img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width

class CycleGAN():

    def init_model(self):
        self.input_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_A")
        self.input_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_B")

        self.pool_fake_b = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="pool_fake_b")
        self.pool_fake_a = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="pool_fake_a")

        self.G_A = Generator("g_A")
        self.G_B = Generator("g_B")
        self.D_A = Discriminator("d_A")
        self.D_B = Discriminator("d_B")

        with tf.variable_scope("Model") as scope:
            self.fake_a = G_A(self.input_A)
            self.fake_b = G_B(self.input_B)
            self.dis_a = D_A(self.input_A)
            self.dis_b = D_B(self.input_B)

            scope.reuse_variables()

            self.dis_fake_b = D_A(self.fake_b)
            self.dis_fake_a = D_B(self.fake_a)
            self.cyc_a = G_B(self.fake_a)
            self.cyc_b = G_A(self.fake_b)

            scope.reuse_variables()

            self.disc_pool_fake_b = D_A(self.pool_fake_b)
            self.disc_pool_gen_B = D_B(self.pool_fake_a)

    def init_loss(self):

        cyc_loss = tf.reduce_mean(tf.abs(self.input_A - self.cyc_a)) + tf.reduce_mean(tf.abs(self.input_B - self.cyc_b))

        loss_g_a = cyc_loss*10 + tf.reduce_mean(tf.squared_difference(self.dis_fake_b, 1))
        loss_g_b = cyc_loss*10 + tf.reduce_mean(tf.squared_difference(self.dis_fake_a, 1))

        loss_d_a = (tf.reduce_mean(tf.square(self.disc_pool_fake_b)) + tf.reduce_mean(tf.squared_difference(self.dis_a,1)))/2.0
        loss_d_b = (tf.reduce_mean(tf.square(self.disc_pool_gen_B)) + tf.reduce_mean(tf.squared_difference(self.dis_b,1)))/2.0


        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

        self.d_A_trainer = optimizer.minimize(loss_d_a, var_list=self.D_A.variables)
        self.d_B_trainer = optimizer.minimize(loss_d_b, var_list=self.D_B.variables)
        self.g_A_trainer = optimizer.minimize(loss_g_a, var_list=self.G_A.variables)
        self.g_B_trainer = optimizer.minimize(loss_g_b, var_list=self.G_B.variables)


        #Summary variables for tensorboard

        tf.summary.histogram('D_Y/true', self.dis_b)
        tf.summary.histogram('D_Y/fake', self.dis_fake_a)
        tf.summary.histogram('D_X/true', self.dis_a)
        tf.summary.histogram('D_X/fake', self.dis_fake_b)

        tf.summary.scalar("g_A_loss", loss_g_a)
        tf.summary.scalar("g_B_loss", loss_g_b)
        tf.summary.scalar("d_A_loss", loss_d_a)
        tf.summary.scalar("d_B_loss", loss_d_b)

        tf.summary.image('input_a', ((self.input_A.reshape((256, 256, 3))+1)*127.5).astype(np.uint8))
        tf.summary.image('input_b', ((self.input_B.reshape((256, 256, 3))+1)*127.5).astype(np.uint8))
        tf.summary.image('fake_a', ((self.fake_a.reshape((256, 256, 3))+1)*127.5).astype(np.uint8))
        tf.summary.image('cyc_a', ((self.cyc_a.reshape((256, 256, 3))+1)*127.5).astype(np.uint8))
        tf.summary.image('fake_b', ((self.fake_b.reshape((256, 256, 3))+1)*127.5).astype(np.uint8))
        tf.summary.image('cyc_b', ((self.cyc_b.reshape((256, 256, 3))+1)*127.5).astype(np.uint8))
