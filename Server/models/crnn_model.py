"""Keras implementation of CRNN."""
import keras.backend as K
from keras.models import Model
from keras.layers import (Input, Dense, Activation, Conv1D, Conv2D, MaxPool2D,
                          BatchNormalization, LSTM, GRU, Reshape, Lambda, 
                          Bidirectional, concatenate)
from keras.layers import LeakyReLU


def CRNN(input_shape, num_classes, leaky=None, fctc=None, drop=None, prediction_only=False, gru=False, cnn=False):
    """Define the CRNN architecture.
    
    Args:
        input_shape: Shape of the input image, typically (256, 32, 1).
        num_classes: Number of characters in alphabet, including the CTC blank character.
        alpha, gamma: Parameters for the focal CTC loss.
        prediction_only: If True, return only the prediction model.
        gru: If True, use GRU layers in the RNN part. Otherwise, use LSTM layers.
        cnn: If True, use 1D CNN layers instead of RNN layers.

    Returns:
        model_train: A model used for training. Only returned if prediction_only is False.
        model_pred: A model used for predictions.

    Reference:
        https://arxiv.org/abs/1507.05717

    Note:
        This implementation includes modifications like Leaky ReLU and a focal CTC loss function.
    """

    # Activation function
    if leaky:
        act = LeakyReLU(alpha=leaky)
    else:
        act = 'relu'

    # Convolutional layers
    x = image_input = Input(shape=input_shape, name='image_input')
    
    # Feature extraction layers
    x = Conv2D(64, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv1_1')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv2_1')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv3_2')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding='same', name='pool3')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv4_1')(x)
    x = BatchNormalization(name='batchnorm1')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv5_1')(x)
    x = BatchNormalization(name='batchnorm2')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding='valid', name='pool5')(x)
    x = Conv2D(512, (2, 2), strides=(1, 1), activation=act, padding='valid', name='conv6_1')(x)

    # RNN layers
    s = x.shape
    x = Reshape((s[1],s[3]), name='reshape_1')(x)

    # Using 1D CNN layers instead of RNN if specified
    if cnn:
        for i in range(6):
            x = BatchNormalization(name=f'batch_normalization_{i+1}')(x)
            x1 = Conv1D(128, 5, activation=act, padding='same', name=f'conv1d_{i*2+1}')(x)
            x2 = Conv1D(128, 5, dilation_rate=2, activation=act, padding='same', name=f'conv1d_{i*2+2}')(x)
            x = concatenate([x1, x2], name=f'concatenate_{i+1}')
    elif gru:
        x = Bidirectional(GRU(256, return_sequences=True, reset_after=False), name='gru_1')(x)
        x = Bidirectional(GRU(256, return_sequences=True, reset_after=False), name='gru_2')(x)
    else:
        if drop:
            dropout, re_dropout = drop
            x = Bidirectional(LSTM(256, return_sequences=True, dropout=dropout, recurrent_dropout=re_dropout, name='lstm_dropout_1'))(x)
            x = Bidirectional(LSTM(256, return_sequences=True, dropout=dropout, recurrent_dropout=re_dropout, name='lstm_dropout_2'))(x)
        else:
            x = Bidirectional(LSTM(256, return_sequences=True, name='lstm_1'))(x)
            x = Bidirectional(LSTM(256, return_sequences=True, name='lstm_2'))(x)

    # Dense layer for classification
    x = Dense(num_classes, name='dense1')(x)
    x = y_pred = Activation('softmax', name='softmax')(x)

    model_pred = Model(inputs=image_input, outputs=x)

    if prediction_only:
        return model_pred

    # Focal CTC loss
    def focal_ctc_lambda_func(args):
        labels, y_pred, input_length, label_length = args
        ctc_loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)
        p = K.exp(-ctc_loss)
        focal_ctc_loss = alpha * K.pow((1-p), gamma) * ctc_loss
        return focal_ctc_loss
    
    # CTC loss
    def ctc_lambda_func(args):
        labels, y_pred, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    labels = Input(shape=[s[1]], dtype='float32', name='label_input')
    input_length = Input(shape=[1], dtype='int64', name='input_length')
    label_length = Input(shape=[1], dtype='int64', name='label_length')
    
    if fctc:
        alpha, gamma = fctc
        focal_ctc_loss = Lambda(focal_ctc_lambda_func, output_shape=(1,), name='focal_ctc')([labels, y_pred, input_length, label_length])
        model_train = Model(inputs=[image_input, labels, input_length, label_length], outputs=focal_ctc_loss)
    else:
        ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([labels, y_pred, input_length, label_length])
        model_train = Model(inputs=[image_input, labels, input_length, label_length], outputs=ctc_loss)

    return model_train, model_pred