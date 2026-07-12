import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, Dropout, Bidirectional, LSTM, Dense
from keras.activations import gelu
from src.models.attention import TemporalAttention
from config import MAX_TIME_FRAMES, N_MFCC

def build_temporal_model(input_shape=(MAX_TIME_FRAMES, N_MFCC)) -> Model:
    """
    Builds a compact CNN-BiLSTM model with temporal attention.
    """
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = Conv1D(64, kernel_size=5, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation(gelu)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.15)(x)
    
    # Block 2
    x = Conv1D(96, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation(gelu)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.20)(x)
    
    # BiLSTM
    x = Bidirectional(LSTM(48, return_sequences=True))(x)
    x = Dropout(0.20)(x)
    
    # Temporal Attention
    x = TemporalAttention()(x)
    
    # Dense Classifier
    x = Dense(64)(x)
    x = tf.keras.layers.Activation(gelu)(x)
    x = Dropout(0.35)(x)
    
    x = Dense(16)(x)
    x = tf.keras.layers.Activation(gelu)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    model = build_temporal_model()
    model.summary()
