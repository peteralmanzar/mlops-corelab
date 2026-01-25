from enum import Enum
from typing import Optional, List
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, LSTM, MaxPooling1D, MaxPooling2D

class LayerActivation(Enum):
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    SOFTMAX = 'softmax'
    LINEAR = 'linear'

class Optimizer(Enum):
    ADAM = 'adam'
    SGD = 'sgd'
    RMS_PROP = 'rmsprop'
    ADA_DELTA = 'adadelta'
    ADA_GRAD = 'adagrad'
    ADA_MAX = 'adamax'
    NADAM = 'nadam'
    FTRL = 'ftrl'

class LossFunction(Enum):
    MEAN_SQUARED_ERROR = 'mean_squared_error'
    MEAN_ABSOLUTE_ERROR = 'mean_absolute_error'
    MEAN_ABSOLUTE_PERCENTAGE_ERROR = 'mean_absolute_percentage_error'
    MEAN_SQUARED_LOGARITHMIC_ERROR = 'mean_squared_logarithmic_error'
    HINGE = 'hinge'
    KULLBACK_LEIBLER_DIVERGENCE = 'kullback_leibler_divergence'
    CATEGORICAL_CROSSENTROPY = 'categorical_crossentropy'
    SPARSE_CATEGORICAL_CROSSENTROPY = 'sparse_categorical_crossentropy'
    BINARY_CROSSENTROPY = 'binary_crossentropy'
    POISSON = 'poisson'
    COSINE_SIMILARITY = 'cosine_similarity'

class Metric(Enum):
    ACCURACY = 'accuracy'
    AUC = 'auc'
    MEAN_SQUARED_ERROR = 'mean_squared_error'
    MEAN_ABSOLUTE_ERROR = 'mean_absolute_error'
    MEAN_ABSOLUTE_PERCENTAGE_ERROR = 'mean_absolute_percentage_error'
    MEAN_SQUARED_LOGARITHMIC_ERROR = 'mean_squared_logarithmic_error'
    HINGE = 'hinge'
    KULLBACK_LEIBLER_DIVERGENCE = 'kullback_leibler_divergence'
    CATEGORICAL_CROSSENTROPY = 'categorical_crossentropy'
    SPARSE_CATEGORICAL_CROSSENTROPY = 'sparse_categorical_crossentropy'
    BINARY_CROSSENTROPY = 'binary_crossentropy'
    POISSON = 'poisson'
    COSINE_SIMILARITY = 'cosine_similarity'


def resolve_metrics(metrics: List[Metric]):
    """Map Metric enum members to keras-safe metric identifiers or instances."""
    metric_map = {
        Metric.ACCURACY: 'accuracy',
        Metric.AUC: 'auc',
        Metric.MEAN_SQUARED_ERROR: 'mean_squared_error',
        Metric.MEAN_ABSOLUTE_ERROR: 'mean_absolute_error',
        Metric.MEAN_ABSOLUTE_PERCENTAGE_ERROR: 'mean_absolute_percentage_error',
        Metric.MEAN_SQUARED_LOGARITHMIC_ERROR: 'mean_squared_logarithmic_error',
        Metric.HINGE: 'hinge',
        Metric.KULLBACK_LEIBLER_DIVERGENCE: 'kullback_leibler_divergence',
        Metric.CATEGORICAL_CROSSENTROPY: 'categorical_crossentropy',
        Metric.SPARSE_CATEGORICAL_CROSSENTROPY: 'sparse_categorical_crossentropy',
        Metric.BINARY_CROSSENTROPY: 'binary_crossentropy',
        Metric.POISSON: 'poisson',
        Metric.COSINE_SIMILARITY: 'cosine_similarity'
    }
    return [metric_map[m] for m in metrics]

def getModelTemplateNone() -> Optional[Model]:
    '''
    Returns a template for a None model.
    '''
    return None

def GetModelTemplateMLPRegression(numberOfFeatures: int, optimizer: Optimizer = Optimizer.ADAM) -> Model:
    '''
    Returns a template for a Multi-Layer Perceptron (MLP) model for regression tasks.
    '''
    model = Sequential([
        Dense(64, activation=LayerActivation.RELU.value, input_shape=(numberOfFeatures,)),
        Dense(64, activation=LayerActivation.RELU.value),
        Dense(1)
    ])

    model.compile(optimizer=optimizer.value, loss=LossFunction.MEAN_SQUARED_ERROR.value, metrics=resolve_metrics([Metric.MEAN_SQUARED_ERROR]))

    return model

def GetModelTemplateMLPBinaryClassification(numberOfFeatures: int, optimizer: Optimizer = Optimizer.ADAM) -> Model:
    '''
    Returns a template for a Multi-Layer Perceptron (MLP) model for binary classification tasks.
    '''
    model = Sequential([
        Dense(64, activation=LayerActivation.RELU.value, input_shape=(numberOfFeatures,)),
        Dense(64, activation=LayerActivation.RELU.value),
        Dense(1, activation=LayerActivation.SIGMOID.value)
    ])

    model.compile(optimizer=optimizer.value, loss=LossFunction.BINARY_CROSSENTROPY.value, metrics=resolve_metrics([Metric.ACCURACY]))

    return model

def GetModelTemplateMLPMultiClassification(numberOfFeatures: int, num_classes: int, optimizer: Optimizer = Optimizer.ADAM) -> Model:
    '''
    Returns a template for a Multi-Layer Perceptron (MLP) model for multiple category classification tasks.
    '''
    model = Sequential([
        Dense(64, activation=LayerActivation.RELU.value, input_shape=(numberOfFeatures,)),
        Dense(64, activation=LayerActivation.RELU.value),
        Dense(num_classes, activation=LayerActivation.SOFTMAX.value)
    ])

    model.compile(optimizer=optimizer.value, loss=LossFunction.CATEGORICAL_CROSSENTROPY.value, metrics=resolve_metrics([Metric.ACCURACY]))

    return model

def GetModelTemplateLSTM(numberOfSteps: int, numberOfFeatures: int, optimizer: Optimizer = Optimizer.ADAM) -> Model:
    '''
    Returns a template for a Long Short-Term Memory (LSTM) model. Ideal for time series forecasting.
    '''
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(numberOfSteps, numberOfFeatures)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(1)
    ])

    model.compile(optimizer=optimizer.value, loss=LossFunction.MEAN_SQUARED_ERROR.value, metrics=resolve_metrics([Metric.MEAN_ABSOLUTE_ERROR]))

    return model

def GetModelTemplate1DCNN(numberOfSteps: int, numberOfFeatures: int, optimizer: Optimizer = Optimizer.ADAM) -> Model:
    '''
    Returns a template for a 1D Convolutional Neural Network (CNN) model. Ideal for time series forecasting.
    '''
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation=LayerActivation.RELU.value, input_shape=(numberOfSteps, numberOfFeatures)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=2, activation=LayerActivation.RELU.value),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation=LayerActivation.RELU.value),
        Dense(1)
    ])

    model.compile(optimizer=optimizer.value, loss=LossFunction.MEAN_SQUARED_ERROR.value, metrics=resolve_metrics([Metric.MEAN_ABSOLUTE_ERROR]))

    return model

def GetModelTemplate2DCNN(image_height: int, image_width: int, num_classes: int, optimizer: Optimizer = Optimizer.ADAM) -> Model:
    '''
    Returns a template for a 2D Convolutional Neural Network (CNN) model. Ideal for image classification tasks.
    '''
    model = Sequential([
        Conv2D(32, (3, 3), activation=LayerActivation.RELU.value, input_shape=(image_height, image_width, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation=LayerActivation.RELU.value),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation=LayerActivation.RELU.value),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation=LayerActivation.RELU.value),
        Dropout(0.5),
        Dense(num_classes, activation=LayerActivation.SOFTMAX.value)
    ])

    model.compile(optimizer=optimizer.value, loss=LossFunction.CATEGORICAL_CROSSENTROPY.value, metrics=resolve_metrics([Metric.ACCURACY]))

    return model
