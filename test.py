import functools

import tensorflow.keras as tfk
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

from absl import app
from absl import flags

from preprocess import *


def percentage_error(r, p):
    return (abs(r - p) / r) * 100

def test(argv):
    del argv

    model = tfk.models.load_model(FLAGS.saving_path, compile=False)
    test_ds = tfds.load(FLAGS.dataset,
                        data_dir=FLAGS.data_dir,
                        as_supervised=True,
                        decoders={'image': tfds.decode.SkipDecoding()},
                        split='validation')
    
    test_ds = (test_ds.map(lambda x, _: tf_process(x), -1)
                      .take(FLAGS.num)
                      .prefetch(-1))
    np_test_iter = test_ds.as_numpy_iterator()

    cls, preds, pes = [], [], []
    for test_data in np_test_iter:
        blks, cl, clwh = np_test_process(test_data, False)
        pred = np.sum(model(blks, training=False).numpy()) // 8
        pe = percentage_error(cl, pred)
        print(('Real Code Length:             {}\n'
               'Real Code Length with Header: {}\n'
               'Predicted Code Length:        {:.0f}\n'
               'Absolute Percentage Error: {}\n'
               .format(cl, clwh, pred, pe)))
        cls.append(cl)
        preds.append(pred)
        pes.append(pe)
    
    fontdict = {'size': 16}
    plt.figure(figsize=(10, 10))
    plt.scatter(gts, preds, alpha=0.6, color='orange')
    plt.grid(True)
    plt.xlabel('Real Code Length', fontdict=fontdict)
    plt.ylabel('Predicted Code Length', fontdict=fontdict)
    plt.savefig('result.png', dpi=300)
#    plt.show()

    print('MAPE: {.2f}'.format(sum(pes) / len(pes)))
 
    

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', 'imagenet2012', 'Dataset')
    flags.DEFINE_string('data_dir', 'gs://iris-us/tfds_datasets',
                        'Path of tfds')
    flags.DEFINE_string('saving_path', None, 'Path of saved model')
    flags.DEFINE_integer('num', 1000, 'Number of test data')
    app.run(test)
