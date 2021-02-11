import tensorflow as tf
import tensorflow_datasets as tfds

from absl import app
from absl import flags
from absl import logging

from preprocess import *

FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 1e-4, 'learning rate')
flags.DEFINE_integer('bs', 64, 'batch size')


def main(argv):
    del argv

    lr = FLAGS.lr
    bs = FLAGS.bs

    train_ds = tfds.load('imagenet2012',
                         data_dir='gs://iris-us/tfds_datasets',
                         as_supervised=True,
                         decoders={'image': tfds.decode.SkipDecoding()},
                         split='train')
    
    valid_ds = tfds.load('imagenet2012',
                         data_dir='gs://iris-us/tfds_datasets',
                         as_supervised=True,
                         decoders={'image': tfds.decode.SkipDecoding()},
                         split='validation')
    
    train_ds = train_ds.map(lambda x, _: tf_process(x), -1).repeat().batch(bs).prefetch(-1)
    valid_ds = valid_ds.map(lambda x, _: tf_process(x), -1).repeat().batch(bs).prefetch(-1)

    model = build_model(lr)

    np_train_iter = train_ds.as_numpy_iterator()
    np_valid_iter = valid_ds.as_numpy_iterator()
    
    steps = 0

    while True:
        steps += 1
        train_data = next(np_train_iter) 
    
        train_blks, train_cls = zip(*map(np_process, train_data))
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
                valid_blks, valid_cls = zip(*map(np_process, valid_data))
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



def build_model(lr):
    dc_predictor = tf.keras.Sequential([
        tf.keras.layers.Reshape((1,)),
        tf.keras.layers.Dense(16, 'relu'),
        tf.keras.layers.Dense(16, 'relu'),
        tf.keras.layers.Dense(16, 'relu'),
        tf.keras.layers.Dense(1, 'linear'),
    ])

    ac_predictor = tf.keras.Sequential([
        tf.keras.layers.Reshape((63, 1)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, 'relu'),
        tf.keras.layers.Dense(16, 'relu'),
        tf.keras.layers.Dense(16, 'relu'),
        tf.keras.layers.Dense(1, 'linear'),
    ])

    inputs = tf.keras.Input(shape=(64,))
    dc_cl = dc_predictor(inputs[..., 0])
    ac_cl = ac_predictor(inputs[..., 1:])
    outputs = dc_cl + ac_cl

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(tf.keras.optimizers.SGD(lr, 0.9, True),
                  tf.keras.losses.Huber(),
                  [tf.keras.metrics.MeanAbsoluteError(),
                   tf.keras.metrics.MeanAbsolutePercentageError()])
    return model


if __name__ == '__main__':
    app.run(main)
