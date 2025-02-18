import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Bidirectional,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def create_lstm_model(
    input_shape=(30, 1),
    lstm_units=[128, 64],
    dense_units=[32, 16],
    dropout_rates=[0.3, 0.2, 0.1],
    learning_rate=0.001,
    loss="huber",
):
    """
    Creates an improved LSTM model for stock returns prediction.

    Parameters:
    - input_shape: Tuple, (sequence_length, features). Default assumes 30 time steps with 5 features.
    - lstm_units: List of integers, number of units in each LSTM layer.
    - dense_units: List of integers, number of units in each Dense layer.
    - dropout_rates: List of floats, dropout rates for LSTM and Dense layers.
    - learning_rate: Float, learning rate for Adam optimizer.
    - loss: String, loss function to use ('huber', 'mse', or 'mae').

    Returns:
    - Compiled Keras model
    """
    model = Sequential()

    # First LSTM layer (Bidirectional)
    model.add(Input(shape=input_shape))
    model.add(
        Bidirectional(
            LSTM(
                lstm_units[0],
                return_sequences=True,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=dropout_rates[0],
                recurrent_dropout=dropout_rates[0],
                kernel_regularizer=l2(0.0005),
                recurrent_regularizer=l2(0.0005),
            )
        )
    )
    model.add(BatchNormalization())

    # Second LSTM layer
    model.add(
        LSTM(
            lstm_units[1],
            return_sequences=False,
            activation="tanh",
            recurrent_activation="sigmoid",
            dropout=dropout_rates[1],
            recurrent_dropout=dropout_rates[1],
            kernel_regularizer=l2(0.0005),
            recurrent_regularizer=l2(0.0005),
        )
    )
    model.add(BatchNormalization())

    # Dense layers
    for i, units in enumerate(dense_units):
        model.add(Dense(units, activation="relu", kernel_regularizer=l2(0.0005)))
        if i < len(dense_units) - 1:  # No dropout after last dense layer
            model.add(Dropout(dropout_rates[2]))
            model.add(BatchNormalization())

    # Output layer
    model.add(Dense(1))

    # Choose appropriate loss function
    if loss == "huber":
        loss_function = tf.keras.losses.Huber(delta=1.0)
    elif loss == "mae":
        loss_function = "mae"
    else:
        loss_function = "mse"

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function,
    )

    return model


# Define callbacks for training
def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001),
    ]

    # Sample training code (assuming X_train, y_train, X_val, y_val are defined)
    """
    callbacks = get_callbacks()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    """
