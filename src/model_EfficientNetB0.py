from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def create_model(input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = preprocess_input(inputs)

    base_model = EfficientNetB0(weights='imagenet',include_top=False,input_shape=input_shape)
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True 

    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)

    model.compile(optimizer=Adam(),loss='binary_crossentropy',metrics=['accuracy'])

    return model