# from tensorflow import keras
# keras backend for amd gpu --> IS ACTUALLY SLOWER THAN CPU ON 2015 iMac lol
# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# import keras
# from keras import backend as K

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, SpatialDropout2D, BatchNormalization
# from keras.datasets import mnist
# from keras.utils import to_categorical
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# CPU backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, SpatialDropout2D, BatchNormalization
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
np.random.seed(4666)  # for reproducibility


def load_and_featurize_data():

    # Read in data
    df = pd.read_csv('data/classes.csv')
    # Mask for Oak and Maple only
    df = df[(df['class'] == 'oak') | (df['class'] == 'maple')]
    # Train test split
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=4666)
    val_df, test_df = train_test_split(val_df, test_size = .50, random_state=4666)
    return train_df, val_df, test_df
    

def generators():
    ## Reduce overfit by shearing, zoom, flip
    train_datagen = ImageDataGenerator(rescale=1./255., shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255.)

    # Create generators for train, test, val to save memory
    train_generator=train_datagen.flow_from_dataframe(dataframe=train_df, directory="./data/", 
                        x_col="image_name", y_col="class", subset='training', batch_size=32, seed=4666, 
                        shuffle=True, class_mode="categorical", target_size=(150,150))
    
    val_generator=train_datagen.flow_from_dataframe(dataframe=val_df, directory="./data/", x_col="image_name", 
                        y_col="class", batch_size=32, seed=4666, shuffle=True, class_mode="categorical", 
                        target_size=(150,150))

    test_generator=test_datagen.flow_from_dataframe(dataframe=test_df, directory="./data/", x_col="image_name", 
                        y_col="class", batch_size=32, seed=4666, shuffle=False, class_mode="categorical", 
                        target_size=(150,150))
    
    return train_generator, val_generator, test_generator



def define_model(nb_filters, kernel_size, input_shape, pool_size):
    model = Sequential()  # model is a linear stack of layers (don't change)

    # note: the convolutional layers and dense layers require an activation function
    # see https://keras.io/activations/
    # and https://en.wikipedia.org/wiki/Activation_function
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
    # ADD MORE LAYERS
    # 1st layer
    ## VGG16
    # model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=kernel_size, padding='same', activation='relu'))  
    # model.add(Conv2D(filters=64, kernel_size=kernel_size, padding='same', activation='relu')) # 2nd conv. layer KEEP
    # model.add(MaxPooling2D(pool_size=pool_size, strides=strides))  # decreases size, helps prevent overfitting
    # model.add(Dropout(0.5))  # zeros out some fraction of inputs, helps prevent overfitting

    
    # model.add(Conv2D(filters=128, kernel_size=kernel_size, padding='same', activation='relu'))
    # model.add(Conv2D(filters=128, kernel_size=kernel_size, padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=pool_size, strides=strides))

    # model.add(Conv2D(filters=256, kernel_size=kernel_size, padding='same', activation='relu'))
    # model.add(Conv2D(filters=256, kernel_size=kernel_size, padding='same', activation='relu'))
    # model.add(Conv2D(filters=256, kernel_size=kernel_size, padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=pool_size, strides=strides))

    # model.add(Conv2D(filters=512, kernel_size=kernel_size, padding='same', activation='relu'))
    # model.add(Conv2D(filters=512, kernel_size=kernel_size, padding='same', activation='relu'))
    # model.add(Conv2D(filters=512, kernel_size=kernel_size, padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=pool_size, strides=strides))

    # model.add(Conv2D(filters=512, kernel_size=kernel_size, padding='same', activation='relu'))
    # model.add(Conv2D(filters=512, kernel_size=kernel_size, padding='same', activation='relu'))
    # model.add(Conv2D(filters=512, kernel_size=kernel_size, padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=pool_size, strides=strides))
    # model.add(Conv2D(nb_filters,
    #                  (kernel_size[0], kernel_size[1]),
    #                  padding='same'))  # 3rd layer
    # model.add(Activation('relu'))  # Activation specification necessary for Conv2D and Dense layers
    # model.add(Conv2D(nb_filters,
    #                  (kernel_size[0], kernel_size[1]),
    #                  padding='same'))  # 4th layer
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
    # model.add(Dropout(0.1))  # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Conv2D(input_shape=input_shape, filters=32, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size, strides=strides))
    model.add(SpatialDropout2D(0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size, strides=strides))
    model.add(SpatialDropout2D(0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size, strides=strides))
    model.add(SpatialDropout2D(0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size, strides=strides))
    model.add(SpatialDropout2D(0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size, strides=strides))
    model.add(SpatialDropout2D(0.1))
    model.add(BatchNormalization())


    # now start a typical neural network
    model.add(Flatten())  # necessary to flatten before going into conventional dense layer  KEEP
    print('Model flattened out to ', model.output_shape)
    model.add(Dense(120, activation='relu'))  # (only) 32 neurons in this layer, really?   KEEP
    model.add(Dropout(0.1))
    model.add(Dense(20, activation='relu'))  # (only) 32 neurons in this layer, really?   KEEP
    model.add(Dropout(0.1))
    model.add(Dense(20, activation='relu'))  # (only) 32 neurons in this layer, really?   KEEP
    model.add(Dropout(0.1))
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))  # zeros out some fraction of inputs, helps prevent overfitting
    model.add(Dense(nb_classes, activation='softmax')) 
    

    # many optimizers available, see https://keras.io/optimizers/#usage-of-optimizers
    # suggest you KEEP loss at 'categorical_crossentropy' for this multiclass problem,
    # and KEEP metrics at 'accuracy'
    # suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # important inputs to the model: don't changes the ones marked KEEP
    batch_size = 64  # number of training samples used at a time to update the weights
    nb_classes = 2    # number of output possibilities: [0 - 9] KEEP
    nb_epoch = 12       # number of passes through the entire train dataset before weights "final"
    img_rows, img_cols = 150, 150   # the size of the MNIST images KEEP
    input_shape = (img_rows, img_cols, 3)   # 1 channel image input (grayscale) KEEP
    nb_filters = 64    # number of convolutional filters to use
    pool_size = (2, 2)  # pooling decreases image size, reduces computation, adds translational invariance
    kernel_size = (3, 3)  # convolutional kernel size, slides over image to learn features
    strides = (1, 1)
    train_df, val_df, test_df = load_and_featurize_data()
    train_generator, val_generator, test_generator = generators()

    model = define_model(nb_filters, kernel_size, input_shape, pool_size)

    steps_per_epoch = int(train_df.shape[0] / batch_size)
    # during fit process watch train and test error simultaneously
    # model.summary()
    model.fit(train_generator, steps_per_epoch = steps_per_epoch, epochs = nb_epoch, verbose = 1, validation_data=val_generator)
    # model.fit_generator(train_generator, steps_per_epoch=2000 // batch_size,
    #     epochs=50,
    #     validation_data=val_generator,
    #     validation_steps=800 // batch_size)
    
    # score = model.evaluate(test_df, verbose=0)
    # # model.evaluate(test_df, verbose=1)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])  # this is the one we care about