#!/usr/bin/python
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image as kerasImage
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s - %(message)s', )
#  format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
logger = logging.getLogger(__name__)
model_arch_file_name = 'model_MobileNet_architecture.json'
model_wights_file_name = 'model-MobileNet-final.h5'
IMAGE_SIZE = (224, 224)
CLASS_LABEL = ["cat", "dog"]


def runCNN(image):
    logger.info('Running Against MobileNet neural network')

    with open(model_arch_file_name, 'r') as f:
        net_model = model_from_json(f.read())
    logger.info('Loaded model architecture')
    net_model.load_weights(model_wights_file_name)
    logger.info('Loaded model weights')

    # preprocess input
    raw_image = kerasImage.load_img(image, target_size=IMAGE_SIZE)
    x = kerasImage.img_to_array(raw_image)
    x = np.expand_dims(x, axis=0)
    post_processed_input_images = np.vstack([x])

    # predict output
    output_probability = net_model.predict(post_processed_input_images)
    output_classes = output_probability.argmax(axis=-1)

    logger.debug("Output classes: %s", output_classes)
    logger.debug("Output probabilities: %s", output_probability)

    for idx, output_class in enumerate(output_classes):
        logger.info("Image {} was a {}".format(image, CLASS_LABEL[output_class]))
        logger.info("Probability: {} ".format(output_probability[idx][output_class]))


if __name__ == "__main__":
    image_file = "../data/sample/valid/dogs/dog.1019.jpg"
    logger.info('TEST_IMAGE: %s', image_file)

    runCNN(image_file)
