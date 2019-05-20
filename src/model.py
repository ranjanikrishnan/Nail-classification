import os
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from keras.models import Sequential
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, RMSprop

CURR_DIR = os.curdir


def load_data():
    """
    Load image data from folder into a dataframe with image path and label

    :return: dataframe with image path and labels(good or bad)
    """
    good_nails_path = f'{CURR_DIR}/data/nailgun/good'
    bad_nails_path = f'{CURR_DIR}/data/nailgun/bad'

    good_nails = map((lambda x: f'{good_nails_path}/{x}'), os.listdir(good_nails_path))
    bad_nails = map((lambda x: f'{bad_nails_path}/{x}'), os.listdir(bad_nails_path))
    filenames = list(good_nails) + list(bad_nails)
    all_nails = []
    for image in filenames:
        if '.jpeg' in image:
            all_nails.append(image)

    df = pd.DataFrame(data=all_nails, columns=['image'])

    df['label'] = df['image'].map(lambda image_name: 'good' if 'good' in image_name else 'bad')

    return df


def train_test_split(df):
    """
    Split into training and testing data

    :param df: dataframe with image path and label
    :return: training data, test data
    """
    data = df['image']
    label = df['label']
    split = ShuffleSplit(n_splits=1, test_size=.25, random_state=0)
    for train_index, test_index in split.split(data, label):
        train_data, test_data = data[train_index], data[test_index]
        train_label, test_label = label[train_index], label[test_index]
    train_df = pd.concat([train_data, train_label], axis=1)
    test_df = pd.concat([test_data, test_label], axis=1)
    return train_df, test_df


def image_preprocess(train_df, test_df, class_mode):
    """
    Data augmentation applied

    :param train_df: training data
    :param test_df: testing data
    :param class_mode: takes values binary or categorical
    :return:
    """
    image_size = [224, 224]
    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True,
                                        vertical_flip=True, zoom_range=0.2, rotation_range=40,
                                        fill_mode='nearest')

    train_generator = data_generator.flow_from_dataframe(
        train_df,
        x_col='image',
        y_col='label',
        batch_size=6,
        target_size=image_size,
        drop_duplicates=True,
        class_mode=class_mode)

    test_generator = data_generator.flow_from_dataframe(
        test_df,
        x_col='image',
        y_col='label',
        batch_size=6,
        target_size=image_size,
        drop_duplicates=True,
        class_mode=class_mode)
    return train_generator, test_generator


def get_baseline_model():
    """
    Network architecture definition for baseline CNN model

    :return: compiled baseline CNN model
    """
    baseline_model = Sequential()
    baseline_model.add(Conv2D(32, (2, 2), input_shape=(224, 224, 3)))
    baseline_model.add(Activation('relu'))
    baseline_model.add(MaxPooling2D(pool_size=(2, 2)))

    baseline_model.add(Conv2D(32, (2, 2)))
    baseline_model.add(Activation('relu'))
    baseline_model.add(MaxPooling2D(pool_size=(2, 2)))

    baseline_model.add(Conv2D(64, (2, 2)))
    baseline_model.add(Activation('relu'))
    baseline_model.add(MaxPooling2D(pool_size=(2, 2)))

    baseline_model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    baseline_model.add(Dense(64))
    baseline_model.add(Activation('relu'))
    baseline_model.add(Dropout(0.5))
    baseline_model.add(Dense(1))
    baseline_model.add(Activation('sigmoid'))

    # Compile the model
    baseline_model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return baseline_model


def get_vgg16_model():
    """
    Network architecture definition for VGG-16 model

    :return: compiled VGG-16 model
    """
    vgg_model = Sequential()
    vgg_model.add(VGG16(weights='imagenet', include_top=False, pooling='avg'))
    vgg_model.add(Dense(units=2, activation='softmax'))
    vgg_model.layers[0].trainable = False

    # Compile the model
    vgg_model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return vgg_model


def train_baseline_model():
    """
    Train baseline CNN model
    """
    checkpoint_filepath = f'{CURR_DIR}/model/baseline-cnn-model.hdf5'

    df = load_data()
    train_df, test_df = train_test_split(df)
    train_generator, test_generator = image_preprocess(train_df, test_df, class_mode='binary')

    # Get baseline model
    baseline_model = get_baseline_model()

    # Configuration to checkpoint the model
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # Fit the baseline model
    baseline_model.fit_generator(
        train_generator,
        steps_per_epoch=3,
        validation_data=test_generator,
        validation_steps=1,
        epochs=12,
        callbacks=[checkpoint])


def train_vgg16_model():
    """
    Train VGG-16 model
    """
    checkpoint_filepath = f'{CURR_DIR}/model/vgg16-classifier-model.hdf5'

    df = load_data()
    train_df, test_df = train_test_split(df)
    train_generator, test_generator = image_preprocess(train_df, test_df, class_mode='categorical')

    # Get VGG-16 model
    vgg_model = get_vgg16_model()

    # Configuration to checkpoint the model
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Fit the VGG-16 model
    vgg_model.fit_generator(
        train_generator,
        steps_per_epoch=3,
        validation_data=test_generator,
        validation_steps=1,
        epochs=15,
        callbacks=callbacks_list)


if __name__ == '__main__':
    train_baseline_model()
    train_vgg16_model()
