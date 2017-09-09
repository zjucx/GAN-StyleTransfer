import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from cyclegan import *
import os


img_height = 256
img_width = 256
img_layer = 3
batch_size = 1
img_size = img_height * img_width
max_images = 100
cp_dir = "./output/checkpoints/"
to_restore = True
global_step = tf.Variable(0, name="global_step", trainable=False)


def gen_image_pool(num_gens, genimg, gen_pool):
    ''' This function saves the generated image to corresponding pool of images.
    In starting. It keeps on feeling the pool till it is full and then randomly selects an
    already stored image and replace it with new one.'''
    pool_size = 50
    if(num_gens < pool_size):
        gen_pool[num_gens] = genimg
        return genimg
    else :
        p = random.random()
        if p > 0.5:
            random_id = random.randint(0,pool_size-1)
            temp = gen_pool[random_id]
            gen_pool[random_id] = genimg
            return temp
        else :
            return genimg

def main():

    # set up img queue
    image_reader = tf.WholeFileReader()
    train_a_files = tf.train.match_filenames_once("/Users/zjucx/Documents/Github/GAN/dataset/input/monet2photo/trainA/*.jpg")
    train_b_files = tf.train.match_filenames_once("/Users/zjucx/Documents/Github/GAN/dataset/input/monet2photo/trainB/*.jpg")
    train_a_queue = tf.train.string_input_producer(train_a_files)
    train_b_queue = tf.train.string_input_producer(train_b_files)
    _, raw_image_a = image_reader.read(train_a_queue)
    _, raw_image_b = image_reader.read(train_b_queue)
    image_a = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(raw_image_a),[256,256]),127.5),1)
    image_b = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(raw_image_b),[256,256]),127.5),1)

    # set up fake img pool
    fake_b_pool = np.zeros((50, batch_size, img_height, img_width, img_layer))
    fake_a_pool = np.zeros((50, batch_size, img_height, img_width, img_layer))

    # init cycle model
    model = CycleGAN()
    model.init_model()
    model.init_loss()

    # set up saver and summary for tf broad
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(cp_dir)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        #Restore the model to run the model from last checkpoint
        if to_restore:
            cp_file = tf.train.latest_checkpoint(cp_dir)
            saver.restore(sess, cp_file)

        if not os.path.exists(cp_dir):
            os.makedirs(cp_dir)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0

            while not coord.should_stop():
                imga = sess.run(image_a).reshape((batch_size, img_height, img_width, img_layer))
                imgb = sess.run(image_b).reshape((batch_size, img_height, img_width, img_layer))
                # get previously generated images
                fake_a, fake_b = sess.run([model.fake_a, model.fake_b],feed_dict={model.input_A:imga, model.input_B:imgb})

                # train
                _, a, b, c, d , summary = (
                      sess.run(
                          [model.optimizers, model.loss_g_a, model.loss_d_b, model.loss_g_b, model.loss_d_a, summary_op],
                          feed_dict={model.input_A:imga, model.input_B:imgb,
                                     model.pool_fake_b: gen_image_pool(step, fake_a, fake_a_pool),
                                     model.pool_fake_a: gen_image_pool(step, fake_b, fake_b_pool)}
                      )
                )
                print("step:%d g_A_trainer:%f d_B_trainer:%f g_B_trainer:%f d_A_trainer%f"%(step, a, b, c, d))

                train_writer.add_summary(summary, step)
                train_writer.flush()

                if step % 10000 == 0:
                  saver.save(sess,os.path.join(cp_dir,"cyclegan"), global_step=step)
                step += 1

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            path = saver.save(sess,os.path.join(cp_dir,"cyclegan"), global_step=step)
            print(path)

            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)



if __name__ == '__main__':
    main()
