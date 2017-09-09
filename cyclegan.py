import tensorflow as tf
from generator import *
from discriminator import *

img_height = 256
img_width = 256
img_layer = 3
batch_size = 1
img_size = img_height * img_width

class CycleGAN():

    def init_model(self):
        self.input_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_A")
        self.input_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_B")

        self.pool_fake_b = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="pool_fake_b")
        self.pool_fake_a = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="pool_fake_a")

        with tf.variable_scope("model") as scope:

            self.G_A = Generator("g_A")
            self.G_B = Generator("g_B")
            self.D_A = Discriminator("d_A")
            self.D_B = Discriminator("d_B")

            self.fake_a = self.G_A(self.input_A)
            self.fake_b = self.G_B(self.input_B)
            self.dis_a = self.D_A(self.input_A)
            self.dis_b = self.D_B(self.input_B)

            scope.reuse_variables()

            self.dis_fake_b = self.D_A(self.fake_b)
            self.dis_fake_a = self.D_B(self.fake_a)
            self.cyc_a = self.G_B(self.fake_a)
            self.cyc_b = self.G_A(self.fake_b)

            scope.reuse_variables()

            self.disc_pool_fake_b = self.D_A(self.pool_fake_b)
            self.disc_pool_gen_B = self.D_B(self.pool_fake_a)

    def init_loss(self):
        cyc_loss = tf.reduce_mean(tf.abs(self.input_A - self.cyc_a)) + tf.reduce_mean(tf.abs(self.input_B - self.cyc_b))

        self.loss_g_a = cyc_loss*10 + tf.reduce_mean(tf.squared_difference(self.dis_fake_b, 1))
        self.loss_g_b = cyc_loss*10 + tf.reduce_mean(tf.squared_difference(self.dis_fake_a, 1))

        self.loss_d_a = (tf.reduce_mean(tf.square(self.disc_pool_fake_b)) + tf.reduce_mean(tf.squared_difference(self.dis_a,1)))/2.0
        self.loss_d_b = (tf.reduce_mean(tf.square(self.disc_pool_gen_B)) + tf.reduce_mean(tf.squared_difference(self.dis_b,1)))/2.0

        optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5)

        model_vars = tf.trainable_variables()
        d_A_vars = [var for var in model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in model_vars if 'g_B' in var.name]

        d_A_trainer = optimizer.minimize(self.loss_d_a, var_list=d_A_vars)
        d_B_trainer = optimizer.minimize(self.loss_d_b, var_list=d_B_vars)
        g_A_trainer = optimizer.minimize(self.loss_g_a, var_list=g_A_vars)
        g_B_trainer = optimizer.minimize(self.loss_g_b, var_list=g_B_vars)

        tf.summary.histogram('D_Y/true', self.dis_b)
        tf.summary.histogram('D_Y/fake', self.dis_fake_a)
        tf.summary.histogram('D_X/true', self.dis_a)
        tf.summary.histogram('D_X/fake', self.dis_fake_b)

        tf.summary.scalar("g_A_loss", self.loss_g_a)
        tf.summary.scalar("g_B_loss", self.loss_g_b)
        tf.summary.scalar("d_A_loss", self.loss_d_a)
        tf.summary.scalar("d_B_loss", self.loss_d_b)

        tf.summary.image('input_a', tf.image.convert_image_dtype((self.input_A+1.0)/2.0, tf.uint8))
        tf.summary.image('input_b', tf.image.convert_image_dtype((self.input_B+1.0)/2.0, tf.uint8))
        tf.summary.image('fake_a', tf.image.convert_image_dtype((self.fake_a+1.0)/2.0, tf.uint8))
        tf.summary.image('fake_b', tf.image.convert_image_dtype((self.fake_b+1.0)/2.0, tf.uint8))


        with tf.control_dependencies([g_A_trainer, d_B_trainer, g_B_trainer, d_A_trainer]):
              self.optimizers = tf.no_op(name='optimizers')
        #Summary variables for tensorboard
