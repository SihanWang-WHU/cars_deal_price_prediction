from keras.layers import Conv1D, Activation, MaxPool1D, Flatten, Dense
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, merge, Add
from tensorflow import keras

def NN_model(input_dim):
    init = keras.initializers.glorot_uniform(seed=1)
    model = keras.models.Sequential()
    model.add(Dense(units=300, input_dim=input_dim, kernel_initializer=init, activation='softplus'))
    #model.add(Dropout(0.2))
    model.add(Dense(units=300, kernel_initializer=init, activation='softplus'))
    #model.add(Dropout(0.2))
    model.add(Dense(units=64, kernel_initializer=init, activation='softplus'))
    model.add(Dense(units=32, kernel_initializer=init, activation='softplus'))
    model.add(Dense(units=8, kernel_initializer=init, activation='softplus'))
    model.add(Dense(units=1))
    return model


