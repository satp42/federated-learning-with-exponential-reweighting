# from https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification

import nest_asyncio
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from matplotlib import pyplot as plt
import wandb

import collections
from functools import partial
from datetime import datetime
from time import time
import gc

NUM_MEGAPOCHS = 20000    # number of times to reselect clients
NUM_CLIENTS = 50         # number of clients to sample on each round
CENTRAL_LR = 0.006

NUM_EPOCHS = 100          # number of times to train for each selected client subset
BATCH_SIZE = 32
CLIENT_LR = 0.001

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
        tf.keras.layers.Dense(685, activation='relu'),
        # tf.keras.layers.Dense(303, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    return tff.learning.from_keras_model(
        model,
        input_spec=spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

if __name__ == '__main__':
    np.random.seed(3091)
    nest_asyncio.apply()
    fcm = tff.simulation.FileCheckpointManager('checkpoint/')

    celeba_train, celeba_test = tff.simulation.datasets.celeba.load_data()
    dataset_spec = preprocess(celeba_train.create_tf_dataset_for_client(
        celeba_train.client_ids[0])).element_spec

    run = wandb.init(project="federated-celeba-vanilla", entity="exr0nprojects")
    print('running', run.name)

    NUM_MEGAPOCHS = run.config.central_epochs
    NUM_CLIENTS =   run.config.central_batch
    CENTRAL_LR =    run.config.central_lr

    NUM_EPOCHS =    run.config.client_epochs
    BATCH_SIZE =    run.config.client_batch
    CLIENT_LR =     run.config.client_lr

    iterative_process = tff.learning.build_federated_averaging_process(
        partial(model_factory, dataset_spec),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=CLIENT_LR),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=CENTRAL_LR),
        use_experimental_simulation_loop=True)

    federated_eval = tff.learning.build_federated_evaluation(partial(model_factory, dataset_spec))  # https://stackoverflow.com/a/56811627/10372825
    eval_dataset = make_federated_data(celeba_test, celeba_test.client_ids[:50])

    state = iterative_process.initialize()
    seen_ids = set()
    print(f"total clients: {len(celeba_train.client_ids)} train, {len(celeba_test.client_ids)} test")

    # sampled_clients = np.random.choice(celeba_train.client_ids, NUM_CLIENTS)
    sampled_clients = celeba_train.client_ids[:NUM_CLIENTS]
    for client in sampled_clients:
        seen_ids.add(client)
    dataset = make_federated_data(celeba_train, sampled_clients)


    for round_num in range(0, NUM_MEGAPOCHS):
        try:
            start_time = time()
            # sampled_clients = np.random.choice(celeba_train.client_ids, NUM_CLIENTS)
            # for client in sampled_clients:
            #     seen_ids.add(client)
            # dataset = make_federated_data(celeba_train, sampled_clients)
            state, metrics = iterative_process.next(state, dataset)
            gc.collect()
            eval_metrics = federated_eval(state.model, eval_dataset)
            overfit_eval_metrics = federated_eval(state.model, dataset)
            print('round {:3d}, time {}, metrics={}, evalmetrics={}'.format(round_num+1, time() - start_time, metrics['train'], eval_metrics))
            wandb.log({
                **metrics['train'],
                'step': round_num * NUM_EPOCHS,
                'test_accuracy': eval_metrics['eval']['sparse_categorical_accuracy'],
                'test_loss': eval_metrics['eval']['loss'],
                'train_acc': overfit_eval_metrics['eval']['sparse_categorical_accuracy'],
                'train_loss': overfit_eval_metrics['eval']['loss'],
                'client_coverage': len(seen_ids)/len(celeba_train.client_ids),
                'time_taken': time() - start_time
            })
            if round_num % 30 == 29:
            # if True:
                fcm.save_checkpoint(state, round_num+1)
                print('saved checkpoint', run.name, round_num+1)
        except KeyboardInterrupt:
            print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\ninterrupted at', datetime.now().strftime("%T"))
            input('press enter to continue...')
