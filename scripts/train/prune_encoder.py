#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
import sys

import keras.backend as K
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from kerassurgeon.identify import get_apoz
from kerassurgeon import Surgeon
import numpy as np
import pandas as pd
import tensorflow as tf

from colormotion.argparse import directory_path
from colormotion.nn.layers import load_weights

base_dir = Path(__file__).resolve().parent
sys.path.append(base_dir)
from filters_optical_flow_split_encoder_decoder import encoder_model
from user_guided import data_generators

START = None
END = None


def args_parser():
    parser = argparse.ArgumentParser(description='Prune weights and compress model.')
    parser.add_argument('weights', type=Path, help='weights file')
    parser.add_argument('dataset', type=directory_path, help='dataset path')
    return parser


def prune_model(model, apoz_df, n_channels_delete):
    sorted_apoz_df = apoz_df.sort_values('apoz', ascending=False)
    high_apoz_index = sorted_apoz_df.iloc[0:n_channels_delete, :]

    # Create the Surgeon and add a 'delete_channels' job for each layer
    # whose channels are to be deleted.
    surgeon = Surgeon(model, copy=True)
    for name in high_apoz_index.index.unique().values:
        channels = list(pd.Series(high_apoz_index.loc[name, 'index'], dtype=np.int64).values)
        surgeon.add_job('delete_channels', model.get_layer(name), channels=channels)
    return surgeon.operate()


def get_model_apoz(model, generator):
    # Get APoZ
    apoz = []
    for layer in model.layers[START:END]:
        if layer.__class__.__name__ == 'Conv2D':
            print(layer.name)
            apoz.extend([(layer.name, i, value) for (i, value)
                          in enumerate(get_apoz(model, layer, generator))])

    layer_name, index, apoz_value = zip(*apoz)
    apoz_df = pd.DataFrame({'layer': layer_name, 'index': index,
                            'apoz': apoz_value})
    apoz_df = apoz_df.set_index('layer')
    return apoz_df


def get_total_channels(model):
    channels = 0
    for layer in model.layers[START:END]:
        if layer.__class__.__name__ == 'Conv2D':
            channels += layer.filters
    return channels


class GeneratorToObject(object):
    def __init__(self, generator):
        self.generator = generator

    def __call__(self):
        for x, y in self.generator:
            yield np.array(x), y

    def __iter__ (self):
        return self

    def __next__ (self):
        x, y = next(self.generator)
        print([i.shape for i in x])
        print(y.shape)
        return x[::-1], y
        # return np.array(x), y


def prune(args):
    _, validation_generator = data_generators(args.dataset)
    validation_generator = GeneratorToObject(validation_generator)
    validation_generator.n = 200
    validation_generator.batch_size = 8
    encoder = encoder_model()
    load_weights(encoder, args.weights, by_name=True)
    percent_pruned = 0
    percent_pruning = 20
    # while percent_pruned <= .3:
    if True:
        total_channels = get_total_channels(encoder)
        n_channels_delete = int(percent_pruning // 100 * total_channels)
        # Prune the model
        apoz_df = get_model_apoz(encoder, validation_generator)
        # percent_pruned += percent_pruning
        print('pruning up to ', str(percent_pruned),
              '% of the original model weights')
        model = prune_model(encoder, apoz_df, n_channels_delete)

        # Clean up tensorflow session after pruning and re-load model
        # checkpoint_name = ('inception_flowers_pruning_' + str(percent_pruned)
        #                    + 'percent')
        output = str(args.weights.parent / 'compressed_{}'.format(args.weights.name))
        print('Saving to {}'.format(output))
        model.save(output)
        # del model
        # K.clear_session()
        # tf.reset_default_graph()
        # model = load_model(output_dir + checkpoint_name + '.h5')

        # # Re-train the model
        # model.compile(loss='categorical_crossentropy',
        #               optimizer=SGD(lr=1e-4, momentum=0.9),
        #               metrics=['accuracy'])
        # checkpoint_name = ('inception_flowers_pruning_' + str(percent_pruned)
        #                    + 'percent')
        # csv_logger = CSVLogger(output_dir + checkpoint_name + '.csv')
        # model.fit_generator(train_generator,
        #                     steps_per_epoch=train_steps,
        #                     epochs=epochs,
        #                     validation_data=validation_generator,
        #                     validation_steps=val_steps,
        #                     workers=4,
        #                     callbacks=[csv_logger])

    # Evaluate the final model performance
    # loss = model.evaluate_generator(validation_generator,
    #                                 validation_generator.n //
    #                                 validation_generator.batch_size)
    # print('pruned model loss: ', loss[0], ', acc: ', loss[1])


def main(args):
    prune(args)

if __name__ == '__main__':
    main(args_parser().parse_args())
