import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, SpatialDropout2D
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D, BatchNormalization



HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_GPUS = 1

BATCH_SIZE = 128
NUM_EPOCHS = 100
NUM_TRAIN_SAMPLES = 50000

EXPERIMENT_NAME = "cifar_basic_normalize_aug_spatial_and_normal_dropout_V5_100_epochs_"

(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()


train_dataset = tf.data.Dataset.from_tensor_slices((x, y))

def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(
        x, HEIGHT + 8, WIDTH + 8)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
#     x = tf.image.random_flip_left_right(x)
    return x, y


def normalize(x, y):
  x = tf.cast(x, tf.float32)
  x /= 255.0  # normalize to [0,1] range
  return x, y

train_dataset = (train_dataset.map(normalize)
                 .shuffle(50000)
                 .batch(128, drop_remainder=True))

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = (test_dataset.map(normalize).batch(128, drop_remainder=True))



i = Input(shape=(HEIGHT, WIDTH, NUM_CHANNELS))
x = Conv2D(128, (3, 3), activation='relu')(i)
x = MaxPooling2D((2, 2))(x)
x = SpatialDropout2D(rate=0.2)(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = SpatialDropout2D(rate=0.2)(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(512, (3, 3), activation='relu')(x)
x = SpatialDropout2D(rate=0.3)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)

x = Dropout(rate=0.3)(x)
# last hidden layer i.e.. output layer
x = Dense(NUM_CLASSES)(x)
 
model = Model(i, x)


model.compile(
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam'),
          metrics=['accuracy'])


tmp_dir = '/raid/developers/uib49306/tmp/'

log_dir= tmp_dir + EXPERIMENT_NAME +  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = TensorBoard(
  log_dir=log_dir,
  update_freq='batch',
  histogram_freq=1)



filepath= log_dir + "/model_checkpoints"
checkpoint_filepath = filepath + "/saved-model-{epoch:02d}-{val_accuracy:.2f}.h5"


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


model.fit(train_dataset,
          epochs=NUM_EPOCHS,
          validation_data=test_dataset,
          validation_freq=1,
          callbacks=[tensorboard_callback, model_checkpoint_callback])






