"""
Project 2 Traffic sign classifier
"""
import cv2
import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import scipy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

########### STEP 0.

training_file = './traffic-signs-data/train.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

signs_names = pd.read_csv('./signnames.csv')

########### STEP 1.

n_train = len(train['features'])
n_test = len(test['features'])
image_shape = '{}x{}'.format(len(train['features'][0]), len(train['features'][0][0]))
n_classes = len(set(train['labels']))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# * Plot histogram of classes occurrence
def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom', rotation='vertical')

train_counts = np.bincount(y_train)
test_counts = np.bincount(y_test)

fig, ax = plt.subplots()
rects1 = ax.bar(np.arange(n_classes) + 0.35, test_counts, 0.35, color='r')
rects2 = ax.bar(np.arange(n_classes), train_counts, 0.35, color='g')

ax.set_ylabel('Occurrence count')
ax.set_title('Traffic sign type occurrence count')
ax.legend((rects1[0], rects2[0]), ('Training set', 'Test set'))
ax.set_xticks(np.arange(n_classes) + 0.35)
ax.set_xticklabels(signs_names['SignName'], rotation='vertical')
autolabel(rects1)
autolabel(rects2)
plt.subplots_adjust(top=0.95, bottom=0.35)


# * Plot gallery of traffic signs examples
def iter_axes(ax):
    for row in ax:
        for col in row:
            yield col
unique_signs = []

for sign_type in range(n_classes):
    sign_index = np.where(y_train == sign_type)[0][0]
    unique_signs.append([sign_type, X_train[sign_index]])

n_cols = 5
n_rows = math.ceil(n_classes/n_cols)

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(8, 9))
fig.tight_layout()

axes_iterator = iter_axes(axes)
for sign, subplot in zip(unique_signs, axes_iterator):
    subplot.axis('off')
    subplot.set_title(signs_names[signs_names.ClassId == sign[0]].SignName.values[0], y=-0.45)
    subplot.imshow(sign[1])
for remaining_subplot in axes_iterator:
    remaining_subplot.axis('off')
plt.show()


########### STEP 2.

def add_more_training_examples(X, y):
    numbers_to_equalize = [3000 - i for i in train_counts]

    additional_X = []
    additional_y = []

    for index in range(len(numbers_to_equalize)):
        for sign, label in zip(X, y):
            if numbers_to_equalize[index] > 0 and label == index:
                for angle in [-10, -8, -6, -2, -1, 1, 2, 4, 8, 10]:
                    (h, w) = sign.shape[:2]
                    center = (w / 2, h / 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(sign, M, (w, h))
                    additional_X.append(rotated)
                    additional_y.append(label)
                    numbers_to_equalize[index] -= 1
                    if numbers_to_equalize[index] > 0:
                        continue

    return np.concatenate((np.asarray(additional_X), X)), np.concatenate((np.asarray(additional_y), y))

X_train, y_train = add_more_training_examples(X_train, y_train)

########### Step 2.1 - Binarize labels

encoder = LabelBinarizer()
encoder.fit(y_train)
train_labels = encoder.transform(y_train)
test_labels = encoder.transform(y_test)
train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

########### Step 2.2 - Split training dataset into training and validation set.

X_train, X_validation, train_labels, validation_labels = train_test_split(X_train, train_labels, test_size=0.25)

def preprocess(dataset):
    preprocessed = []
    dst = np.zeros(shape=(32, 32))
    for traffic_sign in dataset:
        yuv_image = cv2.cvtColor(traffic_sign, cv2.COLOR_BGR2YUV)
        grayscale = -0.5 + yuv_image[:, :, 0] / 255.
        preprocessed.append(np.reshape(grayscale, (32, 32, 1)))
    return preprocessed

X_train = preprocess(X_train)
X_validation = preprocess(X_validation)
X_test = preprocess(X_test)

additional_images = []
additional_image_labels = [40, 39, 17, 2, 9]
binarized_additional_image_labels = encoder.transform(additional_image_labels)

for image in os.listdir('./additional_images'):
    read_image = scipy.misc.imread('./additional_images/' + image)
    additional_images.append(read_image)

additional_images = np.asarray(additional_images)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 9))
fig.tight_layout()

axes_iterator = iter_axes(axes)
i = 0
for sign, subplot in zip(additional_images, axes_iterator):
    subplot.axis('off')
    subplot.set_title(signs_names[signs_names.ClassId == additional_image_labels[i]].SignName.values[0], y=-0.45)
    subplot.imshow(sign)
    i += 1
for remaining_subplot in axes_iterator:
    remaining_subplot.axis('off')
plt.show()

preprocessed_additional_images = preprocess(additional_images)


def LeNet(x):
    a_mean = 0
    a_sigma = 0.1

    # Convolution layer 1. 32x32x1 to 28x28x6.
    conv_1 = tf.nn.conv2d(
        x,
        tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean=a_mean, stddev=a_sigma)),
        strides=[1, 1, 1, 1],
        padding='VALID'
    )
    conv_1 = tf.nn.bias_add(conv_1, tf.Variable(tf.zeros(6)))
    conv_1 = tf.nn.relu(conv_1)

    # Pooling layer 28x28x6 to 14x14x6
    pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Convolution layer 2. 14x14x6 to 10x10x16
    conv_2 = tf.nn.conv2d(
        pool_1,
        tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=a_mean, stddev=a_sigma)),
        strides=[1, 1, 1, 1],
        padding='VALID'
    )
    conv_2 = tf.nn.bias_add(conv_2, tf.Variable(tf.zeros(16)))
    conv_2 = tf.nn.relu(conv_2)

    # Pooling layer 2. 10x10x16 to 5x5x16
    pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten 5x5x16 to 400
    flattened = flatten(pool_2)

    fully_connected_1 = tf.add(
        tf.matmul(flattened, tf.Variable(tf.truncated_normal([400, 120], mean=a_mean, stddev=a_sigma))),
        tf.Variable(tf.zeros(120))
    )
    fully_connected_1 = tf.nn.dropout(fully_connected_1, 0.5)
    fully_connected_1 = tf.nn.relu(fully_connected_1)

    fully_connected_2 = tf.add(
        tf.matmul(fully_connected_1, tf.Variable(tf.truncated_normal([120, 84], mean=a_mean, stddev=a_sigma))),
        tf.Variable(tf.zeros(84))
    )
    fully_connected_2 = tf.nn.dropout(fully_connected_2, 0.5)
    fully_connected_2 = tf.nn.relu(fully_connected_2)

    out = tf.add(
        tf.matmul(fully_connected_2, tf.Variable(tf.truncated_normal([84, n_classes], mean=a_mean, stddev=a_sigma))),
        tf.Variable(tf.zeros(n_classes))
    )
    return out


train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None, n_classes))
fc2 = LeNet(x)

logits = tf.nn.softmax_cross_entropy_with_logits(fc2, y)
loss_op = tf.reduce_mean(logits)
opt = tf.train.AdamOptimizer(learning_rate=0.0001)
train_op = opt.minimize(loss_op)

EPOCHS = 120
BATCH_SIZE = 50

correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def eval_data(sess, X_eval, y_eval):
    num_examples = len(X_eval)
    total_acc = 0
    for i in range(0, num_examples, BATCH_SIZE):
        current_batch_X = X_eval[i: i + BATCH_SIZE]
        current_batch_y = y_eval[i: i + BATCH_SIZE]
        batch_acc = sess.run(accuracy_op, feed_dict={x: current_batch_X, y:current_batch_y})
        total_acc += (batch_acc * len(current_batch_X))
    return total_acc/num_examples

if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    num_examples = len(X_train)
    for epoch in range(EPOCHS):
        X_train, train_labels = shuffle(X_train, train_labels)
        for i in range(0, num_examples, BATCH_SIZE):
            batch_x = X_train[i: i + BATCH_SIZE]
            batch_y = train_labels[i: i + BATCH_SIZE]
            loss = sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

        val_acc = eval_data(sess, X_validation, validation_labels)
        print("EPOCH {} ...".format(epoch + 1))
        print("Validation accuracy = {:.3f}".format(val_acc))
        print()
    test_acc = eval_data(sess, X_test, test_labels)
    print('Test set accuracy: ', test_acc)
    print('My images accuracy: ', sess.run(accuracy_op, feed_dict={x:preprocessed_additional_images, y: binarized_additional_image_labels}))
    print('My images correct predictions', sess.run(correct_prediction, feed_dict={x:preprocessed_additional_images, y: binarized_additional_image_labels}))
    print('My images top 3 probabilities', sess.run(tf.nn.top_k(tf.nn.softmax(fc2), k=3), feed_dict={x:preprocessed_additional_images, y: binarized_additional_image_labels}))
