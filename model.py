import errno
from datetime import datetime
import os
from abc import ABC
import csv

import tensorflow as tf
import tensorboard as tb

# from keras.applications import ResNet50 as RN
# from keras.applications.inception_resnet_v2 import InceptionResNetV2 as IRNV2
# from keras.applications.inception_v3 import InceptionV3 as INS3
# from keras.applications.mobilenet import MobileNet as MOB
# from keras.applications.mobilenetv2 import MobileNetV2 as MOBv2
# from keras.applications.densenet import DenseNet121 as DENS121
# from keras.applications.densenet import DenseNet169 as DENS169
# from keras.applications.densenet import DenseNet201 as DENS201
#
# from keras.callbacks import ModelCheckpoint, TensorBoard
# from keras.models import Model
# from keras.layers import Flatten, Dense, Dropout
# from keras.optimizers import Adam
#
# from keras.preprocessing.image import ImageDataGenerator

FREEZE_LAYERS = 2  # freeze the first this many layers for training
IMAGE_SIZE = (224, 224)
DEFAULT_DATE_TIME_FORMAT = "%Y%m%d-%H%M%S.%s"

NUM_CLASSES = 2
CLASS_LABEL = ["cat", "dog"]
BATCH_SIZE = 8  # try reducing batch size or freeze more layers if your GPU runs out of memory
DATASET_PATH = './data/sample'
NUM_EPOCHS = 1


def get_difference_in_seconds(t_start, t_end):
    difference = t_end - t_start
    difference_in_seconds = difference.total_seconds()
    return difference_in_seconds


def append_row_to_csv(filename, row):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # race condition if path was created after path.exists()
            if exc.errno != errno.EEXIST:
                raise

    with open(filename, 'a+') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(row)


def model(network, num_classes):
    x = network.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax', name='softmax')(x)

    net_final = tf.keras.models.Model(inputs=network.input, outputs=output_layer)

    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in net_final.layers[FREEZE_LAYERS:]:
        layer.trainable = True

    net_final.compile(optimizer=tf.optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    return net_final


class BaseModel(ABC):
    def __init__(self, num_classes, per_epoch_metrics_file, per_batch_metrics_file, log_dir_postfix):
        self.model = model(self.net, num_classes)

        if os.path.isfile(self.file_checkpoints):
            print("Loading existing weights from {}".format(self.file_checkpoints))
            self.model.load_weights(self.file_checkpoints)

        self.batch_size = BATCH_SIZE
        self.dataset_path = DATASET_PATH
        self.num_epochs = NUM_EPOCHS

        self.per_epochs_metric_file = per_epoch_metrics_file
        self.per_batch_metrics_file = per_batch_metrics_file
        self.log_dir_postfix = log_dir_postfix

        self.train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
                                                                             width_shift_range=0.2,
                                                                             height_shift_range=0.2,
                                                                             shear_range=0.2,
                                                                             zoom_range=0.2,
                                                                             channel_shift_range=10,
                                                                             horizontal_flip=True,
                                                                             fill_mode='nearest')

        self.train_batches = self.train_datagen.flow_from_directory(self.dataset_path + '/train',
                                                                    target_size=IMAGE_SIZE,
                                                                    interpolation='bicubic',
                                                                    class_mode='categorical',
                                                                    shuffle=True,
                                                                    batch_size=self.batch_size)

        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        self.valid_batches = valid_datagen.flow_from_directory(self.dataset_path + '/valid',
                                                               target_size=IMAGE_SIZE,
                                                               interpolation='bicubic',
                                                               class_mode='categorical',
                                                               shuffle=False,
                                                               batch_size=self.batch_size)

    def train(self, t_start, epochs, batch_size, training, validation):
        tensorboard_metrics_callback = TrainValTensorBoard(self.log_dir_postfix, t_start, self.per_epochs_metric_file,
                                                           self.per_batch_metrics_file,
                                                           log_dir='./tensorboard',
                                                           write_graph=False)

        # Use Checkpoint model to save model weights after an epoch
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(self.file_checkpoints, monitor='val_acc', verbose=1, save_best_only=True,
                                                                 mode='max')

        # Configure Tensorboard Callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./tensorboard', histogram_freq=0, write_graph=True,
                                                              write_images=False)

        self.model.fit_generator(training,
                                 steps_per_epoch=training.samples // batch_size,
                                 validation_data=validation,
                                 validation_steps=validation.samples // batch_size,
                                 epochs=epochs,
                                 callbacks=[checkpoint_callback, tensorboard_callback, tensorboard_metrics_callback])


class ResNet50(BaseModel):
    def __init__(self):
        self.net = tf.keras.applications.ResNet50(include_top=False, input_tensor=None, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        self.file_checkpoints = "resnet50-weights-improvement.hdf5"
        self.file_weights = "model-resnet50-final.h5"
        self.file_architecture = 'model_Resnet50_architecture.json'

        # todo: unused
        # self.per_epoch_log_dir = '/ResNet50/' + datetime.now().strftime(DEFAULT_DATE_TIME_FORMAT)

        self.per_epoch_metrics_file_name = "ResNet50_PerEpochMetrics_" + datetime.now().strftime(
            DEFAULT_DATE_TIME_FORMAT) + ".csv"
        self.per_batch_metrics_file_name = "ResNet50_PerBatchMetrics_" + datetime.now().strftime(
            DEFAULT_DATE_TIME_FORMAT) + ".csv"
        self.per_epoch_metrics_file = "./trainingMetrics/" + self.per_epoch_metrics_file_name
        self.per_batch_metrics_file = "./trainingMetrics/" + self.per_batch_metrics_file_name
        self.log_dir_postfix = '_ResNet50'

        super(ResNet50, self).__init__(NUM_CLASSES, self.per_epoch_metrics_file,
                                       self.per_batch_metrics_file, self.log_dir_postfix)


class DenseNet121(BaseModel):
    def __init__(self):
        self.net = tf.keras.applications.DenseNet121(include_top=False, input_tensor=None, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        self.file_checkpoints = "DenseNet121-weights-improvement.hdf5"
        self.file_weights = "model-DenseNet121-final.h5"
        self.file_architecture = 'model_DenseNet121_architecture.json'
        self.per_epoch_log_dir = '/DenseNet121/' + datetime.now().strftime(DEFAULT_DATE_TIME_FORMAT)

        self.log_dir_postfix = '_DenseNet121'
        self.per_epoch_metrics_file_name = "DenseNet121_PerEpochMetrics_" + datetime.now().strftime(
            DEFAULT_DATE_TIME_FORMAT) + ".csv"
        self.per_batch_metrics_file_name = "DenseNet121_PerBatchMetrics_" + datetime.now().strftime(
            DEFAULT_DATE_TIME_FORMAT) + ".csv"
        self.per_epoch_metrics_file = "./trainingMetrics/" + self.per_epoch_metrics_file_name
        self.per_batch_metrics_file = "./trainingMetrics/" + self.per_batch_metrics_file_name
        super(DenseNet121, self).__init__(NUM_CLASSES, self.per_epoch_metrics_file, self.per_batch_metrics_file,
                                          self.log_dir_postfix)


class DenseNet169(BaseModel):
    def __init__(self):
        self.net = tf.keras.applications.DenseNet169(include_top=False, input_tensor=None, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        self.file_checkpoints = "DenseNet169-weights-improvement.hdf5"
        self.file_weights = "model-DenseNet169-final.h5"
        self.file_architecture = 'model_DenseNet169_architecture.json'
        self.per_epoch_log_dir = '/DenseNet169/' + datetime.now().strftime(DEFAULT_DATE_TIME_FORMAT)

        self.log_dir_postfix = '_DenseNet169'
        self.per_epoch_metrics_file_name = "DenseNet169_PerEpochMetrics_" + datetime.now().strftime(
            DEFAULT_DATE_TIME_FORMAT) + ".csv"
        self.per_batch_metrics_file_name = "DenseNet169_PerBatchMetrics_" + datetime.now().strftime(
            DEFAULT_DATE_TIME_FORMAT) + ".csv"
        self.per_epoch_metrics_file = "./trainingMetrics/" + self.per_epoch_metrics_file_name
        self.per_batch_metrics_file = "./trainingMetrics/" + self.per_batch_metrics_file_name

        super(DenseNet169, self).__init__(NUM_CLASSES, self.per_epoch_metrics_file,
                                          self.per_batch_metrics_file, self.log_dir_postfix)


class DenseNet201(BaseModel):
    def __init__(self):
        self.net = tf.keras.applications.DenseNet201(include_top=False, input_tensor=None, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        self.file_checkpoints = "DenseNet201-weights-improvement.hdf5"
        self.file_weights = "model-DenseNet201-final.h5"
        self.file_architecture = 'model_DenseNet201_architecture.json'
        self.per_epoch_log_dir = '/DenseNet201/' + datetime.now().strftime(DEFAULT_DATE_TIME_FORMAT)

        self.log_dir_postfix = '_DenseNet201'
        self.per_epoch_metrics_file_name = "DenseNet201_PerEpochMetrics_" + datetime.now().strftime(
            DEFAULT_DATE_TIME_FORMAT) + ".csv"
        self.per_batch_metrics_file_name = "DenseNet201_PerBatchMetrics_" + datetime.now().strftime(
            DEFAULT_DATE_TIME_FORMAT) + ".csv"
        self.per_epoch_metrics_file = "./trainingMetrics/" + self.per_epoch_metrics_file_name
        self.per_batch_metrics_file = "./trainingMetrics/" + self.per_batch_metrics_file_name

        super(DenseNet201, self).__init__(NUM_CLASSES, self.per_epoch_metrics_file,
                                          self.per_batch_metrics_file, self.log_dir_postfix)


class InceptionV3(BaseModel):
    def __init__(self):
        self.net = tf.keras.applications.InceptionV3(include_top=False, input_tensor=None, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        self.file_checkpoints = "inceptionV3-weights-improvement.hdf5"
        self.file_weights = "model-inceptionV3-final.h5"
        self.file_architecture = 'model_InceptionV3_architecture.json'
        self.per_epoch_log_dir = '/InceptionV3/' + datetime.now().strftime(DEFAULT_DATE_TIME_FORMAT)

        self.log_dir_postfix = '_InceptionV3'
        self.per_epoch_metrics_file_name = "InceptionV3_PerEpochMetrics_" + datetime.now().strftime(
            DEFAULT_DATE_TIME_FORMAT) + ".csv"
        self.per_batch_metrics_file_name = "InceptionV3_PerBatchMetrics_" + datetime.now().strftime(
            DEFAULT_DATE_TIME_FORMAT) + ".csv"
        self.per_epoch_metrics_file = "./trainingMetrics/" + self.per_epoch_metrics_file_name
        self.per_batch_metrics_file = "./trainingMetrics/" + self.per_batch_metrics_file_name

        super(InceptionV3, self).__init__(NUM_CLASSES, self.per_epoch_metrics_file,
                                          self.per_batch_metrics_file, self.log_dir_postfix)


class MobileNet(BaseModel):
    def __init__(self):
        self.net = tf.keras.applications.MobileNet(include_top=False, input_tensor=None, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        self.file_checkpoints = "MobileNet-weights-improvement.hdf5"
        self.file_weights = "model-MobileNet-final.h5"
        self.file_architecture = 'model_MobileNet_architecture.json'
        self.per_epoch_log_dir = '/MobileNet/' + datetime.now().strftime(DEFAULT_DATE_TIME_FORMAT)

        self.log_dir_postfix = '_MobileNet'
        self.per_epoch_metrics_file_name = "MobileNet_PerEpochMetrics_" + datetime.now().strftime(
            DEFAULT_DATE_TIME_FORMAT) + ".csv"
        self.per_batch_metrics_file_name = "MobileNet_PerBatchMetrics_" + datetime.now().strftime(
            DEFAULT_DATE_TIME_FORMAT) + ".csv"
        self.per_epoch_metrics_file = "./trainingMetrics/" + self.per_epoch_metrics_file_name
        self.per_batch_metrics_file = "./trainingMetrics/" + self.per_batch_metrics_file_name

        super(MobileNet, self).__init__(NUM_CLASSES, self.per_epoch_metrics_file,
                                        self.per_batch_metrics_file, self.log_dir_postfix)


class MobileNetV2(BaseModel):
    def __init__(self):
        self.net = tf.keras.applications.MobileNetV2(include_top=False, input_tensor=None, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        self.file_checkpoints = "MobileNetV2-weights-improvement.hdf5"
        self.file_weights = "model-MobileNetV2-final.h5"
        self.file_architecture = 'model_MobileNetV2_architecture.json'
        self.per_epoch_log_dir = '/MobileNetV2/' + datetime.now().strftime(DEFAULT_DATE_TIME_FORMAT)

        self.log_dir_postfix = '_MobileNetV2'
        self.per_epoch_metrics_file_name = "MobileNetV2_PerEpochMetrics_" + datetime.now().strftime(
            DEFAULT_DATE_TIME_FORMAT) + ".csv"
        self.per_batch_metrics_file_name = "MobileNetV2_PerBatchMetrics_" + datetime.now().strftime(
            DEFAULT_DATE_TIME_FORMAT) + ".csv"
        self.per_epoch_metrics_file = "./trainingMetrics/" + self.per_epoch_metrics_file_name
        self.per_batch_metrics_file = "./trainingMetrics/" + self.per_batch_metrics_file_name

        super(MobileNetV2, self).__init__(NUM_CLASSES, self.per_epoch_metrics_file,
                                          self.per_batch_metrics_file, self.log_dir_postfix)


class InceptionResNetV2(BaseModel):
    def __init__(self):
        self.net = tf.keras.applications.InceptionResNetV2(include_top=False, input_tensor=None, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        self.file_checkpoints = "InceptionResNetV2-weights-improvement.hdf5"
        self.file_weights = 'model-InceptionResNetV2-final.h5'
        self.file_architecture = 'model_InceptionResNetV2_architecture.json'
        self.per_epoch_log_dir = '/InceptionResNetV2/' + datetime.now().strftime(DEFAULT_DATE_TIME_FORMAT)

        self.log_dir_postfix = '_InceptionResNetV2'
        self.per_epoch_metrics_file_name = "InceptionResNetV2_PerEpochMetrics_" + datetime.now().strftime(
            DEFAULT_DATE_TIME_FORMAT) + ".csv"
        self.per_batch_metrics_file_name = "InceptionResNetV2_PerBatchMetrics_" + datetime.now().strftime(
            DEFAULT_DATE_TIME_FORMAT) + ".csv"
        self.per_epoch_metrics_file = "./trainingMetrics/" + self.per_epoch_metrics_file_name
        self.per_batch_metrics_file = "./trainingMetrics/" + self.per_batch_metrics_file_name

        super(InceptionResNetV2, self).__init__(NUM_CLASSES, self.per_epoch_metrics_file, self.per_batch_metrics_file,
                                                self.log_dir_postfix)


class TrainValTensorBoard(tf.keras.callbacks.TensorBoard):

    def __init__(self, log_dir_postfix, tstart, per_epoch_metrics_file, per_batch_metrics_file, log_dir='./logs',
                 **kwargs):
        self.per_epoch_metrics_file = per_epoch_metrics_file
        self.per_batch_metrics_file = per_batch_metrics_file
        self.epoch_counter = 1
        self.previous_epoch_time = tstart
        # Make the original `TensorBoard` log to a subdirectory 'training'
        # training_log_dir = os.path.join(log_dir, 'training' + log_dir_postfix)
        super(TrainValTensorBoard, self).__init__(**kwargs)  # training_log_dir,

        self.acc = []
        self.loss = []
        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation' + log_dir_postfix)

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.create_file_writer(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        current_time = datetime.now()

        row = [self.epoch_counter, current_time.strftime(DEFAULT_DATE_TIME_FORMAT),
               get_difference_in_seconds(self.previous_epoch_time, current_time)]

        self.previous_epoch_time = current_time

        for name, value in val_logs.items():
            with self.val_writer.as_default():
                tf.summary.scalar(name, value.item(), epoch)
            row.append(value)
        self.val_writer.flush()
        append_row_to_csv(self.per_epoch_metrics_file, row)
        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.totals = {}
        self.write_batch_performance = True
        self.batch_counter = 1

        header_batch = ["Epoch Number", "Batch Number", "Loss", "Accuracy", "Average Loss", "Average Accuracy"]
        header_epoch = ["Epoch Number", "Current Time", "Duration of Epoch", "Loss", "Accuracy"]

        append_row_to_csv(self.per_batch_metrics_file, header_batch)
        append_row_to_csv(self.per_epoch_metrics_file, header_epoch)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        if self.write_batch_performance:

            row = []
            row.append(str(self.epoch_counter))
            row.append(str(self.batch_counter))

            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                if name is 'loss':
                    self.loss.append(value)
                if name is 'accuracy':
                    self.acc.append(value)
                with self.val_writer.as_default():
                    tf.summary.scalar(name, value.item(), batch)
                row.append(str(value))

            if float(len(self.loss)) != 0:
                row.append(round(sum(self.loss) / float(len(self.loss)), 4))

            if float(len(self.acc)) != 0:
                row.append(round(sum(self.acc) / float(len(self.acc)), 4))

            append_row_to_csv(self.per_batch_metrics_file, row)

            self.val_writer.flush()

            self.batch_counter = self.batch_counter + 1
            self.epoch_counter = self.epoch_counter + 1

        self.seen += BATCH_SIZE
