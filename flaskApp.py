#!/usr/bin/env python
import base64
import io

from PIL import Image
import numpy as np
from model import *
import logging
from datetime import datetime
from flask import Flask, request, jsonify, flash, redirect
import tensorflow as tf

from tensorflow.python import keras

logger = logging.getLogger(__name__)
logging.basicConfig()
logger = logging.getLogger("flask_app")
logger.setLevel(logging.INFO)

DEFAULT_DATE_TIME_FORMAT = "%Y%m%d-%H%M%S.%s"
header = ["Start Time", "End Time", "Duration (s)"]
IMAGE_SIZE = (224, 224, 3)
net_models = dict()

global graph, model
graph = tf.compat.v1.get_default_graph()
tf.compat.v1.disable_eager_execution()


def init_models():
    models = ['MobileNet']
    # 'InceptionV3', 'MobileNetV2', 'ResNet50', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'InceptionV3', 'InceptionResNetV2',
    logger.info('Loading all models...')
    logger.debug(models)

    for model_name in models:
        if model_name.startswith("Mobile"):
            with keras.utils.CustomObjectScope({'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
                net_models[model_name] = load_model2(model_name)
        else:
            net_models[model_name] = load_model2(model_name)
        logger.info('Loaded model and weights for ' + model_name)
    logger.info('all models loaded!')


def predict(cnn_name, image):
    with graph.as_default():
        resized_image = np.ma.resize(image, IMAGE_SIZE)
        x = tf.keras.preprocessing.image.img_to_array(resized_image)
        x = np.expand_dims(x, axis=0)
        post_processed_input_images = np.vstack([x])
        #####
        # with tf.keras.utils.CustomObjectScope({'relu6': tf.keras.layers.ReLU(6.), 'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D}):
        #     net_models[cnn_name] = load_model(cnn_name)
        #####
        # predict output
        logger.warning(net_models[cnn_name])
        logger.warning(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        output_probability = net_models[cnn_name].predict(post_processed_input_images)
        output_classes = output_probability.argmax(axis=-1)
    return output_classes[0], output_probability[0].tolist()


def load_model(model_name: str):
    with open('model_' + model_name + '_architecture.json', 'r') as f:
        net_model = keras.models.model_from_json(f.read())
    net_model.load_weights('model-' + model_name + '-final.h5')
    return net_model


def load_model2(model_name: str):
    net_model = keras.models.load_model('model-' + model_name + '-final.h5old')
    return net_model


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
