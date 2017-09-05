import tensorflow as tf
import matplotlib.pyplot as plt
from cyclegan import *

img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width
max_images = 100
global_step = tf.Variable(0, name="global_step", trainable=False)

def train_reader(sess):
    train_a_files = tf.train.match_filenames_once("/Users/zjucx/Documents/Github/GAN/dataset/input/monet2photo/trainA/*.jpg")
    train_b_files = tf.train.match_filenames_once("/Users/zjucx/Documents/Github/GAN/dataset/input/monet2photo/trainB/*.jpg")

    train_a_queue = tf.train.string_input_producer(train_a_files)
    train_b_queue = tf.train.string_input_producer(train_b_files)

    image_reader = tf.WholeFileReader()
    _, image_a = image_reader.read(train_a_queue)
    _, image_b = image_reader.read(train_b_queue)

    image_A = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_a),[256,256]),127.5),1)
    image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_b),[256,256]),127.5),1)

    # Loading images into the tensors
    input_A = np.zeros((max_images, batch_size, img_height, img_width, img_layer))
    input_B = np.zeros((max_images, batch_size, img_height, img_width, img_layer))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(max_images):
        image_tensor = sess.run(image_A)
        if(image_tensor.size() == img_size*batch_size*img_layer):
            input_A[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layer))

    for i in range(max_images):
        image_tensor = sess.run(image_B)
        if(image_tensor.size() == img_size*batch_size*img_layer):
            input_B[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layer))
    coord.request_stop()
    coord.join(threads)
    return input_A, input_B


def main():
    model = CycleGAN()
    model.init_model()
    model.init_loss()

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        input_A, input_B = train_reader(sess)
        for epoch in range(sess.run(global_step),100):
            # Dealing with the learning rate as per the epoch number
            if(epoch < 100) :
                curr_lr = 0.0002
            else:
                curr_lr = 0.0002 - 0.0002*(epoch-100)/100

            for ptr in range(0, max_images):
        sess.run(tf.assign(global_step, epoch + 1))

if __name__ == '__main__':
    main()
