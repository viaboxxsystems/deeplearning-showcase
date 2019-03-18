#!/usr/bin/python
from argparse import ArgumentParser

import keras
from keras.models import model_from_json
import numpy as np
from model import *
from keras.preprocessing import image as kerasImage
import logging
from datetime import datetime
from keras.utils.generic_utils import CustomObjectScope

# models = ['ResNet50', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'InceptionV3', 'InceptionResNetV2']
models = ['MobileNet', 'MobileNetV2']
directories = {"data/sample/valid/cats", "data/sample/valid/dogs"}

DEFAULT_DATE_TIME_FORMAT = "%Y%m%d-%H%M%S.%s"
classification_time_file = "./classificationTime.txt"
classification_results = "./classificationresults.txt"
header2 = ["Model Name", "Image Name", "Actual Class", "Classification Time", "Class", "Probability"]
append_row_to_csv(classification_results, header2)
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s - %(message)s', )
#  format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
logger = logging.getLogger(__name__)


def load_model(model_name: str):
    with open('model_' + model_name + '_architecture.json', 'r') as f:
        net_models = model_from_json(f.read())
    net_models.load_weights('model-' + model_name + '-final.h5')
    return net_models


for model in models:
    if model.startswith("Mobile"):
        logger.info('Loaded model' + model)
        with CustomObjectScope({'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D':
            keras.layers.DepthwiseConv2D}):
            net_model = load_model(model)
    else:
        with open('model_' + model + '_architecture.json', 'r') as f:
            net_model = model_from_json(f.read())
        logger.info('Loaded model %s', model)
        net_model.load_weights('model-' + model + '-final.h5')
        logger.info('Loaded weights')

    for d in directories:
        directory = os.fsencode(d)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            t_start = datetime.now()

            # preprocess input
            raw_image = kerasImage.load_img(directory.decode('utf-8') + "/" + filename, target_size=IMAGE_SIZE)
            x = kerasImage.img_to_array(raw_image)
            x = np.expand_dims(x, axis=0)
            post_processed_input_images = np.vstack([x])

            output_probability = net_model.predict(post_processed_input_images)
            output_classes = output_probability.argmax(axis=-1)

            t_end = datetime.now()
            difference_in_seconds = get_difference_in_seconds(t_start, t_end)

            actual_type = filename.split(".")[0]
            for idx, output_class in enumerate(output_classes):
                row2 = []
                row2.append(', '.join(
                    [model, filename, actual_type, str(difference_in_seconds), CLASS_LABEL[output_class],
                     str(output_probability[idx][output_class])]))
                append_row_to_csv(classification_results, row2)
