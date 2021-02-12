import tensorflow.keras as tfk

def build_model(lr):
    dc_predictor = tfk.Sequential([
        tfk.layers.Reshape((1,)),
        tfk.layers.Dense(16, 'relu'),
        tfk.layers.Dense(16, 'relu'),
        tfk.layers.Dense(16, 'relu'),
        tfk.layers.Dense(1, 'linear'),
    ])

    ac_predictor = tfk.Sequential([
        tfk.layers.Reshape((63, 1)),
        tfk.layers.Bidirectional(tfk.layers.LSTM(16, return_sequences=True)),
        tfk.layers.Flatten(),
        tfk.layers.Dense(16, 'relu'),
        tfk.layers.Dense(16, 'relu'),
        tfk.layers.Dense(16, 'relu'),
        tfk.layers.Dense(1, 'linear'),
    ])

    inputs = tfk.Input(shape=(64,))
    dc_cl = dc_predictor(inputs[..., 0])
    ac_cl = ac_predictor(inputs[..., 1:])
    outputs = dc_cl + ac_cl

    model = tfk.Model(inputs=inputs, outputs=outputs)
    model.compile(tfk.optimizers.SGD(lr, 0.9, True),
                  tfk.losses.Huber(),
                  [tfk.metrics.MeanAbsoluteError(),
                   tfk.metrics.MeanAbsolutePercentageError()])
    return model
