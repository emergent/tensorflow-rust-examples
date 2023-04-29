import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os

print('tensorflow version is {}'.format(tf.__version__))

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

class_names = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']


def save_img(filename, data):
    img = Image.new('L', (28, 28))
    pix = img.load()
    for i in range(28):
        for j in range(28):
            pix[i, j] = int(data[j][i])
    img2 = img.resize((28, 28))
    img2.save(filename)


def dump(data, dhead):
    for i in range(10):
        dname = "{}/{}".format(dhead, class_names[i])
        if os.path.isdir(dname) is False:
            os.makedirs(dname)
    images, labels = data
    count = [0]*10
    for i in range(len(images)):
        index = labels[i]
        filename = '{}/{}/{}.png'.format(dhead,
                                         class_names[index], count[index])
        save_img(filename, images[i])
        count[index] += 1
        print('{}     \r'.format(filename), end='')
    print('done                   ')


# thanks from https://github.com/kaityo256/fashion_mnist_dump
print('saving test images')
dump((test_images, test_labels), "images")


train_images = train_images / 255.0
test_images = test_images / 255.0

# build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=5)
model.save('models')
