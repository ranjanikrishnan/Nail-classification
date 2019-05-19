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
    checkpoint_filepath = f"{CURR_DIR}/model/nail-classifier-model.hdf5"
    global model
    model = load_model(checkpoint_filepath)
    model._make_predict_function()


def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = np.array(img_to_array(image))

    image = np.expand_dims(image, axis=0)
    print('shape', image.shape)
    processed_image = preprocess_input(image)
    # image = image / 255.

    return processed_image


@app.route("/predict", methods=["POST"])
def predict():
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
            label_dict = {"0.0": "bad", "1.0": "good"}
            print(predictions[0][0])
            data = {"prediction": label_dict[str(predictions[0][0])], "success": True}

    return flask.jsonify(data)


# def test_predict():
#     test_image = f'{CURR_DIR}/data/nailgun/good/1522072665_good.jpeg'
#     # image = flask.request.files[test_image].read()
#     # image = Image.open(io.BytesIO(test_image))
#     image = load_img(test_image)
#     # preprocess the image and process it for classification
#     image = prepare_image(image, target=[224, 224])
#
#     # classify the input image and then initialize the list
#     # of predictions to return to the client
#     model = load_vgg_model()
#     predictions = model.predict(image)
#     return predictions
#
#
# print(f"result: {test_predict()}")

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_vgg_model()
    app.run()
