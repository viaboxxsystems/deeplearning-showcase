#!/usr/bin/env python
import base64
import io

from PIL import Image
from keras.models import model_from_json
import numpy as np
from keras.utils.generic_utils import CustomObjectScope
from model import *
from keras.preprocessing import image as kerasImage
import logging
from datetime import datetime
import tensorflow as tf
import keras
from flask import Flask, request, Response, jsonify, flash, redirect

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s - %(message)s', )
logger = logging.getLogger(__name__)
DEFAULT_DATE_TIME_FORMAT = "%Y%m%d-%H%M%S.%s"
header = ["Start Time", "End Time", "Duration (s)"]
IMAGE_SIZE = (224, 224, 3)
net_models = dict()

global graph, model
graph = tf.compat.v1.get_default_graph()


def init_models():
    models = ['MobileNet', 'MobileNetV2', 'ResNet50', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'InceptionV3',
              'InceptionResNetV2', 'InceptionV3']
    filtered_models = [value for value in models if not value.startswith("Mobile")]
    # added 123123 to not filter MobileNet and MobileNetV2

    print('Loading all models...')
    print(filtered_models)

    for model_name in filtered_models:
        if model_name.startswith("Mobile"):
            logger.info('Loaded model' + model_name)
            with CustomObjectScope({'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D':
                keras.layers.DepthwiseConv2D}):
                net_models[model_name] = load_model(model_name)
        else:
            logger.info('Loaded model' + model_name)
            net_models[model_name] = load_model(model_name)
        logger.info('Loaded weights for ' + model_name)


def predict(cnn_name, image):
    # preprocess input
    with graph.as_default():
        resized_image = np.ma.resize(image, IMAGE_SIZE)
        x = kerasImage.img_to_array(resized_image)
        x = np.expand_dims(x, axis=0)
        post_processed_input_images = np.vstack([x])

        # predict output
        output_probability = net_models[cnn_name].predict(post_processed_input_images)
        output_classes = output_probability.argmax(axis=-1)

    return output_classes[0], output_probability[0].tolist()


def load_model(model_name: str):
    with open('model_' + model_name + '_architecture.json', 'r') as f:
        net_models = model_from_json(f.read())
    net_models.load_weights('model-' + model_name + '-final.h5')
    return net_models


app = Flask(__name__)
with app.app_context():
    app.secret_key = 'supersecretkey'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.max_redirects = 60
    init_models()


@app.route('/net/<net_name>', methods=['GET', 'POST'])
def use_net_to_classify_image(net_name):
    r = request

    # if r.files.get("image"):
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)

    image = r.files["image"].read()
    image = Image.open(io.BytesIO(image))

    t_start = datetime.now()
    prediction, prob = predict(net_name, image)
    t_end = datetime.now()
    difference_in_seconds = get_difference_in_seconds(t_start, t_end)

    response = {"type": prediction.item(), "probabilities": prob, "type_probability": prob[prediction],
                "time": difference_in_seconds}
    return jsonify(response)
