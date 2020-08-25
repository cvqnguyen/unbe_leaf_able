from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
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
    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid',
                     input_shape=input_shape))  # first conv. layer  KEEP
    model.add(Activation('tanh'))  # Activation specification necessary for Conv2D and Dense layers
    
    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid'))  # 2nd conv. layer KEEP
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
    model.add(Dropout(0.25))  # zeros out some fraction of inputs, helps prevent overfitting
    

    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid'))  # 3rd layer
    model.add(Activation('tanh'))  # Activation specification necessary for Conv2D and Dense layers
    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid'))  # 4th layer
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
    model.add(Dropout(0.25))  # zeros out some fraction of inputs, helps prevent overfitting

    # now start a typical neural network
    model.add(Flatten())  # necessary to flatten before going into conventional dense layer  KEEP
    print('Model flattened out to ', model.output_shape)
    model.add(Dense(32))  # (only) 32 neurons in this layer, really?   KEEP
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # zeros out some fraction of inputs, helps prevent overfitting
    model.add(Dense(nb_classes))  # 10 final nodes (one for each class)  KEEP
    model.add(Activation('softmax'))  # softmax at end to pick between classes 0-9 KEEP

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
    batch_size = 50  # number of training samples used at a time to update the weights
    nb_classes = 2    # number of output possibilities: [0 - 9] KEEP
    nb_epoch = 10       # number of passes through the entire train dataset before weights "final"
    img_rows, img_cols = 150, 150   # the size of the MNIST images KEEP
    input_shape = (img_rows, img_cols, 2)   # 1 channel image input (grayscale) KEEP
    nb_filters = 20    # number of convolutional filters to use
    pool_size = (2, 2)  # pooling decreases image size, reduces computation, adds translational invariance
    kernel_size = (3, 3)  # convolutional kernel size, slides over image to learn features

    train_df, val_df, test_df = load_and_featurize_data()
    train_generator, val_generator, test_generator = generators()

    model = define_model(nb_filters, kernel_size, input_shape, pool_size)

    steps_per_epoch = int(train_df.shape[0] / batch_size)
    # during fit process watch train and test error simultaneously
    # model.summary()
    model.fit(train_generator, steps_per_epoch = steps_per_epoch, epochs = nb_epoch, verbose = 0, validation_data=val_generator)
    # model.fit_generator(train_generator, steps_per_epoch=2000 // batch_size,
    #     epochs=50,
    #     validation_data=val_generator,
    #     validation_steps=800 // batch_size)
    
    # score = model.evaluate(test_df, verbose=0)
    # # model.evaluate(test_df, verbose=1)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])  # this is the one we care about