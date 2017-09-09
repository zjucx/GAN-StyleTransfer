import tensorflow as tf
import matplotlib.pyplot as plt
from cyclegan import *

img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width
max_images = 100
cp_dir = "./output/checkpoints/"
to_restore = False
global_step = tf.Variable(0, name="global_step", trainable=False)


def train_reader(sess):
    train_a_files = tf.train.match_filenames_once("/Users/zjucx/Documents/Github/GAN/dataset/input/monet2photo/trainA/*.jpg")
    train_b_files = tf.train.match_filenames_once("/Users/zjucx/Documents/Github/GAN/dataset/input/monet2photo/trainB/*.jpg")

    train_a_queue = tf.train.string_input_producer(train_a_files)
    train_b_queue = tf.train.string_input_producer(train_b_files)

    image_reader = tf.WholeFileReader()
    _, image_a = image_reader.read(train_a_queue)
    _, image_b = image_reader.read(train_b_queue)

    image_a = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_a),[256,256]),127.5),1)
    image_b = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_b),[256,256]),127.5),1)

    # Loading images into the tensors
    input_A = np.zeros((max_images, batch_size, img_height, img_width, img_layer))
    input_B = np.zeros((max_images, batch_size, img_height, img_width, img_layer))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(max_images):
        image_tensor = sess.run(image_a)
        if(image_tensor.size() == img_size*batch_size*img_layer):
            input_A[i] = image_tensor.reshape((batch_size, img_height, img_width, img_layer))

    for i in range(max_images):
        image_tensor = sess.run(image_b)
        if(image_tensor.size() == img_size*batch_size*img_layer):
            input_B[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layer))

    coord.request_stop()
    coord.join(threads)
    return input_A, input_B

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

    # set up fake img pool
    gen_A_pool = tf.placeholder(tf.float32, [None, img_height, img_width, img_layer], name="gen_A_pool")
    gen_B_pool = tf.placeholder(tf.float32, [None, img_height, img_width, img_layer], name="gen_B_pool")

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
        input_A, input_B = train_reader(sess)

        #Restore the model to run the model from last checkpoint
        if to_restore:
            cp_file = tf.train.latest_checkpoint(cp_dir)
            saver.restore(sess, cp_file)

        if not os.path.exists(cp_dir):
                os.makedirs(cp_dir)

        for epoch in range(sess.run(global_step), max_epoll):
            for idx in range(0, max_images):
                fake_a, fake_b = sess.run([fake_a, fake_b],feed_dict={input_A:input_A[idx], input_B:input_B[idx]})

                # train
                _, a, b, c, d , summary = (
                      sess.run(
                          [optimizers, g_loss_a, d_loss_b, g_loss_b, d_loss_a, summary_op],
                          feed_dict={input_A:A_input[idx], input_B:B_input[idx],
                                     pool_fake_b: gen_image_pool(epoch*max_epoll+idx, fake_b, fake_b_pool),
                                     pool_fake_a: gen_image_pool(epoch*max_epoll+idx, fake_a, fake_a_pool)}
                      )
                )
                print("epoch:%d idx:%d g_A_trainer:%f d_B_trainer:%f g_B_trainer:%f d_A_trainer%f"%(epoch, idx, a, b, c, d))

                train_writer.add_summary(summary, epoch*max_epoll+idx)
                train_writer.flush()

            saver.save(sess,os.path.join(cp_dir,"cyclegan"), global_step=epoch)

if __name__ == '__main__':
    main()
