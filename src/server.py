import os
import io
import requests
import flask
from flask import request
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import PIL.Image as Image

CURR_DIR = os.curdir

# Initialize the flask application
app = flask.Flask(__name__)

def load_baseline_model():
    """
    Loads baseline CNN model from the checkpoint
    """
    checkpoint_filepath = f"{CURR_DIR}/model/baseline-cnn-model.hdf5"
    global baseline_model
    baseline_model = load_model(checkpoint_filepath)
    baseline_model._make_predict_function()


def load_vgg_model():
    """
    Loads VGG-16 model from the checkpoint
    """
    checkpoint_filepath = f"{CURR_DIR}/model/vgg16-classifier-model.hdf5"
    global vgg_model
    vgg_model = load_model(checkpoint_filepath)
    vgg_model._make_predict_function()


def prepare_image(image, target):
    """
    Apply preprocessing to image

    :param image: image to be processed
    :param target: resize parameters
    :return: processed image
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = np.array(img_to_array(image))

    image = np.expand_dims(image, axis=0)
    processed_image = preprocess_input(image)

    return processed_image


def interpreted_prediction(prediction):
    """
    Interpret the prediction value as good or bad

    :param prediction: prediction from the model
    :return: probability of the prediction and prediction as a dictionary value
    """
    class_dict = {0: 'bad', 1: 'good'}
    rp = int(round(prediction[0][0]))
    return float(prediction[0][0]), rp, class_dict.get(rp)


@app.route("/predict", methods=["GET"])
def predict():
    """
    Predict using VGG-16 model

    :return: prediction results as json
    """
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}

    image_url = request.args.get("image_url")
    if image_url:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))

        # Preprocess the image and process it for classification
        image = prepare_image(image, target=(224, 224))

        # Classify the input image and then initialize the list of predictions to return to the client
        predictions = vgg_model.predict(image)
        data = {"prediction": interpreted_prediction(predictions), "success": True}

    return flask.jsonify(data)

@app.route("/baseline/predict", methods=["GET"])
def baseline_predict():
    """
    Predict using baseline CNN model

    :return:
    """
    # Initialize the data dictionary that will be returned from the view
    data = {"success": False}
    image_url = request.args.get("image_url")
    if image_url:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))

        # Preprocess the image and process it for classification
        image = prepare_image(image, target=(224, 224))

        # Classify the input image and then initialize the list of predictions to return to the client
        predictions = baseline_model.predict(image)
        data = {"prediction": interpreted_prediction(predictions), "success": True}

    return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_baseline_model()
    load_vgg_model()
    app.run()
