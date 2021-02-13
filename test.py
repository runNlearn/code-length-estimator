import functools

import tensorflow.keras as tfk
import tensorflow_datasets as tfds

from absl import app
from absl import flags

from preprocess import *


def test(argv):
    del argv

    model = tfk.models.load_model(FLAGS.saving_path, compile=False)
    test_ds = tfds.load(FLAGS.dataset,
                        data_dir=FLAGS.data_dir,
                        as_supervised=True,
                        decoders={'image': tfds.decode.SkipDecoding()},
                        split='validation')
    
    test_ds = test_ds.map(lambda x, _: tf_process(x), -1).prefetch(-1)
    np_test_iter = test_ds.as_numpy_iterator()


    for test_data in np_test_iter:
        blks, cl, clwh = np_test_process(test_data, False)
        pred = np.sum(model(blks, training=False).numpy()) // 8
        print(('Real Code Length: {}\n'
               'Real Code Length with Header: {}\n'
               'Predicted Code Length: {:.0f}\n'
               .format(cl, clwh, pred)))


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', 'imagenet2012', 'Dataset')
    flags.DEFINE_string('data_dir', 'gs://iris-us/tfds_datasets',
                        'Path of tfds')
    flags.DEFINE_string('saving_path', None, 'Path of saved model')
    app.run(test)
