import tensorflow as tf
import matplotlib.pyplot as plt

def train_reader(sess, input_a, input_b):
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
    input_a = np.zeros((max_images, batch_size, img_height, img_width, img_layer))
    input_b = np.zeros((max_images, batch_size, img_height, img_width, img_layer))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(max_images):
        image_tensor = sess.run(image_A)
        if(image_tensor.size() == img_size*batch_size*img_layer):
            input_a[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layer))

    for i in range(max_images):
        image_tensor = sess.run(image_B)
        if(image_tensor.size() == img_size*batch_size*img_layer):
            input_b[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layer))
    coord.request_stop()
    coord.join(threads)
    return input_a, input_b


def main():
    model = CycleGAN()
    if to_train:
        model.train()
    elif to_test:
        model.test()

if __name__ == '__main__':
    main()
