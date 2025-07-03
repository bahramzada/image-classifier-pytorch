# image-classifier-pytorch

import tensorflow as tf
import tensorflow_hub as hub

resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
IMAGE_SHAPE = (224, 224)
NUM_CLASSES = 10  # Burada öz sinif sayını yaz

def create_functional_model(model_url, num_classes=10):
    inputs = tf.keras.Input(shape=IMAGE_SHAPE + (3,))
    x = hub.KerasLayer(model_url, trainable=False, name='Feature_extraction_layer')(inputs)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = create_functional_model(resnet_url, num_classes=NUM_CLASSES)
model.summary()
