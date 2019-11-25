from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, LSTM
# from tensorflow.python.training.rmsprop import RMSPropOptimizer
from tensorflow.python.training import AdamOptimizer, RMSPropOptimizer


NUM_CLASSES = 2

INPUT_TENSOR_NAME = "inputs_input" # According to Amazon, needs to match the name of the first layer + "_input"
                                   # Workaround for actual known bugs

def keras_model_fn(input_dim, output_dim, return_sequences, hyperparameters):
    model = Sequential()
    model.add(LSTM(input_shape=(None, input_dim),
                   units=output_dim,
                   return_sequences=return_sequences, 
                   name="inputs"))

    model.add(Dropout(0.2))

    model.add(LSTM(128,
                   return_sequences=False))

    model.add(Dropout(0.2))

    model.add(Dense(units=1))
    model.add(Activation('linear'))
    
    opt = hyperparameters['optimizer'],
    loss= hyperparameters['loss'],
    eval_metric = hyperparameters['eval_metric'],
    
    model.compile(loss=loss, optimizer=opt, metrics=eval_metric)
    
    print(model.summary())

    return model

def serving_input_fn(hyperparameters):
    tensor = tf.placeholder(tf.float32, shape=[None,input_dim])
    inputs = {INPUT_TENSOR_NAME: tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def train_input_fn(training_dir, hyperparameters):
    return _input(tf.estimator.ModeKeys.TRAIN, batch_size=hyperparameters['batch_size'], data_dir=training_dir)


def eval_input_fn(training_dir, hyperparameters):
    return _input(tf.estimator.ModeKeys.EVAL, batch_size=hyperparameters['batch_size'], data_dir=training_dir)


def _input(mode, batch_size, data_dir):
    assert os.path.exists(data_dir), ("Unable to find images resources for input, are you sure you downloaded them ?")

    if mode == tf.estimator.ModeKeys.TRAIN:
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
    else:
        datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(data_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size)
    images, labels = generator.next()

    return {INPUT_TENSOR_NAME: images}, labels