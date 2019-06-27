from datetime import datetime
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

FREEZE_LAYERS = 2  # freeze the first this many layers for training
IMAGE_SIZE = (224, 224)
DEFAULT_DATE_TIME_FORMAT = "%Y%m%d-%H%M%S.%s"

NUM_CLASSES = 2
CLASS_LABEL = ["cat", "dog"]
BATCH_SIZE = 8  # try reducing batch size or freeze more layers if your GPU runs out of memory
DATASET_PATH = '../data/sample'
NUM_EPOCHS = 1

file_checkpoints = "MobileNet-weights-improvement.hdf5"
file_weights = "model-MobileNet-final.h5"
file_architecture = 'model_MobileNet_architecture.json'


def prepare_model(network, num_classes):
    x = network.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation='softmax', name='softmax')(x)

    net_final = Model(inputs=network.input, outputs=output_layer)

    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in net_final.layers[FREEZE_LAYERS:]:
        layer.trainable = True

    net_final.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    return net_final


# super(MobileNet, self).__init__(NUM_CLASSES, self.per_epoch_metrics_file, self.per_batch_metrics_file, self.log_dir_postfix)


def train(network, epochs, batch_size, training, validation):
    # Use Checkpoint model to save model weights after an epoch
    checkpoint_callback = ModelCheckpoint(file_checkpoints, monitor='val_acc', verbose=1, save_best_only=True,
                                          mode='max')

    # Configure Tensorboard Callback
    tensorboard_callback = TensorBoard(log_dir='./tensorboard', histogram_freq=0, write_graph=True,
                                       write_images=False)

    network.fit_generator(training, steps_per_epoch=training.samples // batch_size,
                          validation_data=validation,
                          validation_steps=validation.samples // batch_size,
                          epochs=epochs,
                          callbacks=[checkpoint_callback, tensorboard_callback])


def main():
    """
    Script entrypoint
    """

    train_datagen = ImageDataGenerator(rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       channel_shift_range=10,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                      target_size=IMAGE_SIZE,
                                                      interpolation='bicubic',
                                                      class_mode='categorical',
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE)

    valid_datagen = ImageDataGenerator()
    valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                                      target_size=IMAGE_SIZE,
                                                      interpolation='bicubic',
                                                      class_mode='categorical',
                                                      shuffle=False,
                                                      batch_size=BATCH_SIZE)

    dnn = MobileNet(include_top=False, input_tensor=None, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    final_dnn = prepare_model(dnn, NUM_CLASSES)

    # show class indices
    print('****************')
    for cls, idx in train_batches.class_indices.items():
        print('Class #{} = {}'.format(idx, cls))
    print('****************')

    print(dnn.summary())

    train(network=final_dnn, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, training=train_batches, validation=valid_batches)

    # save trained weights
    # final_dnn.save(file_weights)

    final_dnn.save_weights(file_weights)
    with open(file_architecture, 'w') as f:
        f.write(final_dnn.to_json())


if __name__ == "__main__":
    main()
