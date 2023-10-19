import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import data_utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

ROOT = '/home/sameer/biods220/EHR-Sepsis-Prediction/'
target = "VANCOMYCIN" # 'SEPSIS' or 'MI' or 'VANCOMYCIN'
EPOCHS=10
BATCH_SIZE=16

def build_lstm_model(lstm_hidden_units=256):
    """
    Return a simple Keras model with a single LSTM layer, dropout later,
    and then dense prediction layer.

    Args:
    lstm_hidden_units (int): units in the LSTM layer

    Returns:
    model_lstm (tf.keras.Model) LSTM keras model with output dimension (None,1)
    """
    model_lstm = Sequential()
    model_lstm.add(LSTM(lstm_hidden_units, dropout=0.5))
    model_lstm.add(Dense(15, activation='sigmoid'))

    return model_lstm

def build_masked_lstm_model(num_timesteps, num_features, lstm_hidden_units=256):
    """
    Return a simple Keras model with a masking single LSTM layer, dropout later,
    and then dense prediction layer.

    Args:
    num_timesteps (int): num timesteps per input data object.
    num_features (int): num features per input data object.
    lstm_hidden_units (int): units in the LSTM layer

    Returns:
    model_lstm (tf.keras.Model) LSTM keras model with output dimension (None,1)
    """
    model_lstm = build_lstm_model(lstm_hidden_units)
    for layer in model_lstm.layers:
        layer.supports_masking = True
    
    model = Sequential() 
    model.add(tf.keras.layers.Masking(mask_value=0., input_shape=(num_timesteps, num_features)))
    model.add(model_lstm)
    return model

if __name__ == "__main__":

    (
        train_x,
        val_x,
        train_y,
        val_y,
        no_feature_cols,
        test_x,
        test_y,
        x_boolmat_test,
        y_boolmat_test,
        x_boolmat_val,
        y_boolmat_val,
        features,
    ) = data_utils.load_seq_dataset(ROOT, target)

    # convert all float64 to float32
    train_x = train_x.astype(np.float32)
    val_x = val_x.astype(np.float32)
    test_x = test_x.astype(np.float32)
    train_y = train_y.astype(np.float32)
    val_y = val_y.astype(np.float32)
    test_y = test_y.astype(np.float32)

    print("train shapes ", train_x.shape, train_y.shape)
    print("val shapes   ", val_x.shape, val_y.shape)
    print("test shapes  ", test_x.shape, test_y.shape)
    print("# features   ", len(features))
    
    num_timesteps, num_features = train_x.shape[-2:]
    lstm_hidden_units = 256
    model = build_masked_lstm_model(num_timesteps, num_features, lstm_hidden_units)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    model.summary()
    
    history = model.fit(train_x, train_y, validation_data=(val_x,val_y),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS, verbose=2,
                    shuffle=True)
    
    results = model.evaluate(test_x, test_y, batch_size=128)
    print("test loss, test acc:", results)

    print("Generate predictions for 3 samples")
    predictions = model.predict(test_x[:3])
    print("predictions shape:", predictions.shape)
    