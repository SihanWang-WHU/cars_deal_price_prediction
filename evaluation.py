from keras.callbacks import Callback, EarlyStopping
import numpy as np
from sklearn.metrics import mean_absolute_error

class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_train, y_train = self.data[0][0], self.data[0][1]
        y_pred3 = self.model.predict(X_train)
        y_pred = np.zeros((len(y_pred3),))
        y_true = np.zeros((len(y_pred3),))
        for i in range(len(y_pred3)):
            y_pred[i] = y_pred3[i]
        for i in range(len(y_pred3)):
            y_true[i] = y_train[i]
        trn_s = mean_absolute_error(y_true, y_pred)
        logs['trn_score'] = trn_s

        X_val, y_val = self.data[1][0], self.data[1][1]
        y_pred3 = self.model.predict(X_val)
        y_pred = np.zeros((len(y_pred3),))
        y_true = np.zeros((len(y_pred3),))
        for i in range(len(y_pred3)):
            y_pred[i] = y_pred3[i]
        for i in range(len(y_pred3)):
            y_true[i] = y_val[i]
        val_s = mean_absolute_error(y_true, y_pred)
        logs['val_score'] = val_s
        print('trn_score', trn_s, 'val_score', val_s)

        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)