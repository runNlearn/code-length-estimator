import os
import logging
import functools

from collections import namedtuple

import tensorflow_datasets as tfds

from preprocess import *
from model import build_model

__all__ = ['run']

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

def main(argv):
    del argv

    run(lr=FLAGS.lr,
        bs=FLAGS.bs,
        data_dir=FLAGS.data_dir,
        dataset=FLAGS.dataset,
        round=FLAGS.round,
        save=FLAGS.save)



def run(**kwargs):
    Config = namedtuple('Config', ' '.join(kwargs.keys()))
    config = Config(**kwargs)

    saving_path_template = ('gs://iris-us/jsm/research/code-length-estimator/'
                            '{}-bs{}-lr{}').format(config.dataset, config.bs, config.lr) 
    saving_path_template = saving_path_template + '-round' if config.round else saving_path_template
    saving_path_template = saving_path_template + '-steps{:08d}'


    train_ds = tfds.load(config.dataset,
                         data_dir=config.data_dir,
                         as_supervised=True,
                         decoders={'image': tfds.decode.SkipDecoding()},
                         split='train')
    
    valid_ds = tfds.load(config.dataset,
                         data_dir=config.data_dir,
                         as_supervised=True,
                         decoders={'image': tfds.decode.SkipDecoding()},
                         split='validation')
    
    train_ds = train_ds.map(lambda x, _: tf_process(x), -1).repeat().batch(config.bs).prefetch(-1)
    valid_ds = valid_ds.map(lambda x, _: tf_process(x), -1).repeat().batch(config.bs).prefetch(-1)

    model = build_model(config.lr)

    np_train_iter = train_ds.as_numpy_iterator()
    np_valid_iter = valid_ds.as_numpy_iterator()
    preprocess = functools.partial(np_process, round=config.round)
    
    steps = 0

    logging.info("Training Start")
    while True:
        steps += 1
        train_data = next(np_train_iter) 
    
        train_blks, train_cls = zip(*map(preprocess, train_data))
        train_blks = np.stack(train_blks, axis=0)
        train_cls = np.stack(train_cls, axis=0)
        tloss, tmetric1, tmetric2 = model.train_on_batch(train_blks, train_cls)
    
        if steps % 100 == 0:
            logging.info("step: {:5d}  {}: {:8.4f}  {}: {:8.4f}  {}: {:8.4f}".format(
                    steps,
                    model.metrics_names[0], tloss,
                    model.metrics_names[1], tmetric1,
                    model.metrics_names[2], tmetric2))
        if steps % 2000 == 0:
            vsteps = 0
            vlosses = []
            vmetric1s = []
            vmetric2s = []
            while vsteps < 1000 :
                vsteps += 1
                valid_data = next(np_valid_iter)
                valid_blks, valid_cls = zip(*map(preprocess, valid_data))
                valid_blks = np.stack(valid_blks, axis=0)
                valid_cls = np.stack(valid_cls, axis=0)
                vloss, vmetric1, vmetric2 = model.test_on_batch(valid_blks, valid_cls)
                vlosses.append(vloss)
                vmetric1s.append(vmetric1)
                vmetric2s.append(vmetric2)
            logging.info("Validation {}: {:8.4f}  {}: {:8.4f}  {}: {:8.4f}". format(
                model.metrics_names[0], sum(vlosses) / len(vlosses),
                model.metrics_names[1], sum(vmetric1s) / len(vmetric1s),
                model.metrics_names[2], sum(vmetric2s) / len(vmetric2s),
            ))
            if config.save:
                saving_path = saving_path_template.format(steps)
                model.save(saving_path, overwrite=True, include_optimizer=True,
                           save_format='tf')


if __name__ == '__main__':
    from absl import app
    from absl import flags

    FLAGS = flags.FLAGS
    flags.DEFINE_float('lr', 1e-4, 'Learning rate')
    flags.DEFINE_integer('bs', 64, 'Batch size')
    flags.DEFINE_string('data_dir', 'gs://iris-us/tfds_datasets', 'Path for tfds dataset')
    flags.DEFINE_string('dataset', 'imagenet2012', 'Name of dataset')
    flags.DEFINE_boolean('round', False, 'Round after scale to log2 regime')
    flags.DEFINE_boolean('save', False, 'Saving option')

    app.run(main)
