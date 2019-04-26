#!/usr/bin/python
import tensorflow as tf
from argparse import ArgumentParser
# from  import model_from_json
import numpy as np
from model import *
# from keras.preprocessing import image as kerasImage
import logging
from datetime import datetime

DEFAULT_DATE_TIME_FORMAT = "%Y%m%d-%H%M%S.%s"
classification_time_file = "./classificationTime.txt"
classification_results = "./classificationresults.txt"
header = ["Start Time", "End Time", "Duration (s)"]
header2 = ["Image Name", "Category", "Percentage"]
append_row_to_csv(classification_results, header2)

logger = logging.getLogger(__name__)
logging.basicConfig()
logger = logging.getLogger("classify_image_using_trained_net")
logger.setLevel(logging.INFO)


def runCNN(cnn_name, image):
    logger.info('Running Against {} neural network'.format(cnn_name))

    with open('model_' + cnn_name + '_architecture.json', 'r') as f:
        net_model = tf.keras.models.model_from_json(f.read())
    logger.info('Loaded model')
    net_model.load_weights('model-' + cnn_name + '-final.h5')
    logger.info('Loaded weights')

    # preprocess input
    raw_image = tf.keras.preprocessing.image.load_img(image, target_size=IMAGE_SIZE)
    x = tf.keras.preprocessing.image.img_to_array(raw_image)
    x = np.expand_dims(x, axis=0)
    post_processed_input_images = np.vstack([x])

    # predict output
    t_start = datetime.now()
    row = [t_start.strftime(DEFAULT_DATE_TIME_FORMAT)]

    output_probability = net_model.predict(post_processed_input_images)
    output_classes = output_probability.argmax(axis=-1)

    logger.debug("Output classes: %s", output_classes)
    logger.debug("Output probabilities: %s", output_probability)

    for idx, output_class in enumerate(output_classes):
        logger.info("Image {} was a {}".format(image, CLASS_LABEL[output_class]))
        logger.info("Probability: {} ".format(output_probability[idx][output_class]))
        row2 = []
        row2.append(image + "," + CLASS_LABEL[output_class] + "," + str(output_probability[idx][output_class]))
        append_row_to_csv(classification_results, row2)

    t_end = datetime.now()
    difference_in_seconds = get_difference_in_seconds(t_start, t_end)
    row.append(t_end.strftime(DEFAULT_DATE_TIME_FORMAT))
    row.append(str(difference_in_seconds))

    append_row_to_csv(classification_time_file, header)
    append_row_to_csv(classification_time_file, row)


if __name__ == "__main__":
    models = [cls.__name__ for cls in vars()[BaseModel.__name__].__subclasses__()]

    parser = ArgumentParser()
    parser.add_argument("-n", "--net", required=True, choices=models,
                        help="Neural network model that should be used for classification")
    parser.add_argument("-i", "--image", required=True, help="Path to image that should be classified")
    args = parser.parse_args()

    logger.info('CNN_TO_RUN: %s', args.net)
    logger.info('TEST_IMAGE: %s', args.image)

    runCNN(args.net, args.image)
