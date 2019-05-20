import os
import io
import flask
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import preprocess_input
import numpy as np
import PIL.Image as Image

CURR_DIR = os.curdir

# initialize the flask application
app = flask.Flask(__name__)


def load_vgg_model():
    """
    loads the model from the checkpoint
    """
    checkpoint_filepath = f"{CURR_DIR}/model/nail-classifier-model-categorical.hdf5"
    global model
    model = load_model(checkpoint_filepath)
    model._make_predict_function()


def prepare_image(image, target):
    """
    apply preprocessing to image
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
    # processed_image = processed_image / 255.

    return processed_image


def interpreted_prediction(prediction):
    """
    interpret the prediction value as good or bad
    :param prediction: prediction from the model
    :return: probability of the prediction and prediction as a dictionary value
    """
    class_dict = {0: 'bad', 1: 'good'}
    rp = int(round(prediction[0][0]))
    return float(prediction[0][0]), rp, class_dict.get(rp)


@app.route("/predict", methods=["POST"])
def predict():
    """

    :return:
    """
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and process it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            predictions = model.predict(image)
            data = {"prediction": interpreted_prediction(predictions), "success": True}

    return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_vgg_model()
    app.run()
