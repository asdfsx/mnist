# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import os
import subprocess
import gzip

import tempfile
import argparse

FLAGS = None

def load_data(data_dir):
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    
    paths = []
    for fname in files:
        paths.append(os.path.join(data_dir, fname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)

def main():
    tf.logging.set_verbosity(tf.logging.ERROR)
    print(tf.__version__)

    (train_images, train_labels), (test_images, test_labels) = load_data(FLAGS.data_dir)

    # scale the values to 0.0 to 1.0
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    # reshape for feeding into the model
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
    print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3, 
                            strides=2, activation='relu', name='Conv1'),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')
    ])
    model.summary()
    
    testing = False
    epochs = 5

    model.compile(optimizer=tf.train.AdamOptimizer(), 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=epochs)
    
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('\nTest accuracy: {}'.format(test_acc))

    MODEL_DIR = FLAGS.modle_dir
    version = 1
    export_path = os.path.join(MODEL_DIR, str(version))
    print('export_path = {}\n'.format(export_path))
    if os.path.isdir(export_path):
        print('\nAlready saved a model, cleaning up\n')
        os.removedirs(export_path)
    
    tf.saved_model.simple_save(
        keras.backend.get_session(),
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name:t for t in model.outputs})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
  
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/mnist/input_data'),
        help='Directory for storing input data')
    parser.add_argument(
        '--modle_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/mnist/logs/mnist_with_summaries'),
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
