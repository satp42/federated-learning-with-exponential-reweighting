import nest_asyncio
nest_asyncio.apply()

import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import wandb
from tqdm import trange

import collections
from functools import partial
from datetime import datetime
from time import time
import gc

np.random.seed(0)

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])
NUM_CLIENTS = 50       # number of clients to sample on each round
NUM_EPOCHS = 100         # number of times to train for each selected client subset
NUM_MEGAPOCHS = 20000    # number of times to reselect clients
BATCH_SIZE = 32
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

CLIENT_LR = 0.001
CENTRAL_LR = 0.003

IMG_WIDTH = 84
IMG_HEIGHT = 84


def preprocess(dataset):
    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            x=tf.reshape(element['pixels'], [-1, 784]),
            y=tf.reshape(element['label'], [-1, 1]))

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE
                    ).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


preprocessed_example_dataset = preprocess(example_dataset)

def make_federated_data(client_data, client_ids):
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]

def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])
def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

federated_eval = tff.learning.build_federated_evaluation(model_fn)  # https://stackoverflow.com/a/56811627/10372825
eval_dataset = make_federated_data(emnist_test, emnist_test.client_ids[:50])

state = iterative_process.initialize()

seen_ids = set()
print(f"total clients: {len(emnist_train.client_ids)} train, {len(emnist_test.client_ids)} test")

for round_num in trange(0, NUM_MEGAPOCHS):
    try:
        start_time = time()
        sampled_clients = np.random.choice(emnist_train.client_ids, NUM_CLIENTS)
        for client in sampled_clients:
            seen_ids.add(client)
        dataset = make_federated_data(emnist_train, sampled_clients)
        state, metrics = iterative_process.next(state, dataset)
        gc.collect()
        eval_metrics = federated_eval(state.model, eval_dataset)
        # print('round {:2d}, metrics={}, evalmetrics={}'.format(round_num+1, metrics['train'], eval_metrics))
        wandb.log({
            **metrics['train'],
            'step': round_num * NUM_EPOCHS,
            'test_accuracy': eval_metrics['eval']['sparse_categorical_accuracy'],
            'test_loss': eval_metrics['eval']['loss'],
            'client_coverage': len(seen_ids)/len(emnist_train.client_ids),
            'time_taken': time() - start_time
        })
        # if round_num % 30 == 29:
        #     fcm.save_checkpoint(state, round_num+1)
        #     print('saved checkpoint', run.name, round_num+1)
    except KeyboardInterrupt:
        print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\ninterrupted at', datetime.now().strftime("%T"))
        input('press enter to continue...')


