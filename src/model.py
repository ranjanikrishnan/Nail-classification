import os
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from keras.models import Sequential
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, RMSprop

CURR_DIR = os.curdir


def load_data():
    good_nails_path = f"{CURR_DIR}/data/nailgun/good"
    bad_nails_path = f"{CURR_DIR}/data/nailgun/bad"

    good_nails = map((lambda x: f'{good_nails_path}/{x}'), os.listdir(good_nails_path))
    bad_nails = map((lambda x: f'{bad_nails_path}/{x}'), os.listdir(bad_nails_path))
    filenames = list(good_nails) + list(bad_nails)
    all_nails = []
    for image in filenames:
        if ".jpeg" in image:
            all_nails.append(image)

    df = pd.DataFrame(data=all_nails, columns=['image'])

    df['label'] = df['image'].map(lambda image_name: 'good' if 'good' in image_name else 'bad')

    return df


def train_test_data(df):
    data = df["image"]
    label = df["label"]
    split = ShuffleSplit(n_splits=1, test_size=.50, random_state=0)
    for train_index, test_index in split.split(data, label):
        train_data, test_data = data[train_index], data[test_index]
        train_label, test_label = label[train_index], label[test_index]
    return train_data, test_data, train_label, test_label


def image_preprocess(train_df, test_df):
    checkpoint_filepath = f"{CURR_DIR}/model/nail-classifier-model-categorical.hdf5"
    image_size = [224, 224]
    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True,
                                        vertical_flip=True, zoom_range=0.2, shear_range=0.2 )

    train_generator = data_generator.flow_from_dataframe(
        train_df,
        x_col="image",
        y_col="label",
        batch_size=6,
        target_size=image_size,
        drop_duplicates=True,
        class_mode='categorical')

    test_generator = data_generator.flow_from_dataframe(
        test_df,
        x_col="image",
        y_col="label",
        batch_size=10,
        target_size=image_size,
        drop_duplicates=True,
        class_mode='categorical')

    # vgg_model = VGG16()
    vgg_model = Sequential()
    vgg_model.add(VGG16(weights="imagenet", include_top=False, pooling='avg'))
    vgg_model.add(Dense(units=2, activation='softmax'))
    vgg_model.layers[0].trainable = False

    # compile the model
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    vgg_model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Checkpoint the model
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Fit the model
    vgg_model.fit_generator(
        train_generator,
        steps_per_epoch=3,
        validation_data=test_generator,
        validation_steps=1,
        epochs=15,
        callbacks=callbacks_list)


if __name__ == "__main__":
    df_loaded = load_data()
    train_data, test_data, train_label, test_label = train_test_data(df_loaded)
    train_df = pd.concat([train_data, train_label], axis=1)
    test_df = pd.concat([test_data, test_label], axis=1)

    image_preprocess(train_df, test_df)
