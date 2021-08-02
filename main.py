# from https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification

import nest_asyncio
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from matplotlib import pyplot as plt
import wandb

import collections
from functools import partial

NUM_CLIENTS = 30        # number of clients to sample on each round
NUM_EPOCHS = 2          # number of times to train for each selected client subset
NUM_MEGAPOCHS = 2000    # number of times to reselect clients
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

IMG_WIDTH = 84
IMG_HEIGHT = 84

def preprocess(dataset):
    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            # x=tf.reshape(element['image'], [1, 21168]),
            x = element['image'],
            y = tf.reshape(int(element['attractive']), [-1, 1]))

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

def make_federated_data(data, ids):
    return [ preprocess(data.create_tf_dataset_for_client(i)) for i in ids ]

def model_factory(spec):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(18, (21, 21), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5),

        tf.keras.layers.Conv2D(32, (6, 6), activation='relu'),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2)

        # tf.keras.layers.Conv2D(32, (21, 21), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        # tf.keras.layers.MaxPooling2D((2, 2)),
        # tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5),

        # tf.keras.layers.InputLayer(input_shape=(21168,)),
        # tf.keras.layers.Dense(1000, kernel_initializer='zeros'),
        # tf.keras.layers.Dense(2, kernel_initializer='zeros'),
        # tf.keras.layers.Softmax(),
    ])

    return tff.learning.from_keras_model(
        model,
        input_spec=spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

if __name__ == '__main__':
    np.random.seed(1336)
    celeba_train, celeba_test = tff.simulation.datasets.celeba.load_data()
    dataset_spec = preprocess(celeba_train.create_tf_dataset_for_client(
        celeba_train.client_ids[0])).element_spec

    # # print(len(celeba_train.client_ids), "\n".join([x for x in celeba_train.element_type_structure]))
    #
    dataset = celeba_train.create_tf_dataset_for_client(celeba_train.client_ids[0])
    # print(dataset.element_spec)

    # figure = plt.figure(figsize=(20, 4))
    # j = 0
    # for example in dataset.take(40):
    #     plt.subplot(4, 10, j+1)
    #     plt.imshow(example['image'].numpy(), cmap='gray', aspect='equal')
    #     plt.axis('off')
    #     j += 1
    # plt.show()

    dataset = preprocess(dataset)
    batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(dataset)))
    #
    # print(list(batch.items())[0][1].shape)

    wandb.init(project="federated-celeba-vanilla", entity="exr0nprojects")

    iterative_process = tff.learning.build_federated_averaging_process(
        partial(model_factory, dataset_spec),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.002),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.001))

    state = iterative_process.initialize()

    # TODO: move sampled clients into megapochs loop
    sampled_clients = np.random.choice(celeba_train.client_ids, NUM_CLIENTS)
    dataset = make_federated_data(celeba_train, sampled_clients)
    for round_num in range(0, NUM_MEGAPOCHS):
        state, metrics = iterative_process.next(state, dataset)
        print('round {:2d}, metrics={}'.format(round_num+1, metrics))
        wandb.log(metrics['train'])
