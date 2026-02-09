from enum import Enum
from typing import Optional, List, Dict, Any
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, LSTM, MaxPooling1D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl

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


def _create_optimizer_with_lr(optimizer: Optimizer, learning_rate: float):
    """Create Keras optimizer instance with specified learning rate."""
    optimizer_map = {
        Optimizer.ADAM: lambda lr: Adam(learning_rate=lr),
        Optimizer.SGD: lambda lr: SGD(learning_rate=lr),
        Optimizer.RMS_PROP: lambda lr: RMSprop(learning_rate=lr),
        Optimizer.ADA_DELTA: lambda lr: Adadelta(learning_rate=lr),
        Optimizer.ADA_GRAD: lambda lr: Adagrad(learning_rate=lr),
        Optimizer.ADA_MAX: lambda lr: Adamax(learning_rate=lr),
        Optimizer.NADAM: lambda lr: Nadam(learning_rate=lr),
        Optimizer.FTRL: lambda lr: Ftrl(learning_rate=lr),
    }
    return optimizer_map.get(optimizer, lambda lr: Adam(learning_rate=lr))(learning_rate)


def build_dynamic_mlp(
    num_features: int,
    task_type: str,
    num_classes: int = 1,
    architecture_params: Optional[Dict[str, Any]] = None,
    optimizer: Optimizer = Optimizer.ADAM,
    learning_rate: float = 0.001
) -> Model:
    """
    Build MLP model with dynamic architecture from hyperparameters.

    Args:
        num_features: Number of input features
        task_type: 'regression', 'binary_classification', or 'multi_classification'
        num_classes: Number of output classes
        architecture_params: Dict with keys:
            - num_hidden_layers: int
            - hidden_units: List[int] - units per layer
            - dropout_rate: float
            - activation: str
        optimizer: Optimizer enum
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras Model
    """
    # Default architecture if not provided
    if architecture_params is None:
        architecture_params = {
            'num_hidden_layers': 2,
            'hidden_units': [64, 64],
            'dropout_rate': 0.0,
            'activation': 'relu'
        }

    num_layers = architecture_params.get('num_hidden_layers', 2)
    hidden_units = architecture_params.get('hidden_units', [64] * num_layers)
    dropout_rate = architecture_params.get('dropout_rate', 0.0)
    activation = architecture_params.get('activation', 'relu')

    # Ensure hidden_units list matches num_layers
    if len(hidden_units) < num_layers:
        hidden_units = hidden_units + [hidden_units[-1]] * (num_layers - len(hidden_units))

    model = Sequential()

    # Input + first hidden layer
    model.add(Dense(
        hidden_units[0],
        activation=activation,
        input_shape=(num_features,)
    ))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    # Additional hidden layers
    for i in range(1, num_layers):
        model.add(Dense(hidden_units[i], activation=activation))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Output layer and compilation based on task type
    if task_type == 'regression':
        model.add(Dense(1))
        loss = LossFunction.MEAN_SQUARED_ERROR.value
        metrics = resolve_metrics([Metric.MEAN_SQUARED_ERROR])
    elif task_type == 'binary_classification':
        model.add(Dense(1, activation=LayerActivation.SIGMOID.value))
        loss = LossFunction.BINARY_CROSSENTROPY.value
        metrics = resolve_metrics([Metric.ACCURACY])
    else:  # multi_classification
        model.add(Dense(num_classes, activation=LayerActivation.SOFTMAX.value))
        loss = LossFunction.CATEGORICAL_CROSSENTROPY.value
        metrics = resolve_metrics([Metric.ACCURACY])

    # Create optimizer with learning rate
    optimizer_instance = _create_optimizer_with_lr(optimizer, learning_rate)

    model.compile(optimizer=optimizer_instance, loss=loss, metrics=metrics)

    return model


def build_dynamic_lstm(
    sequence_length: int,
    num_features: int,
    task_type: str,
    num_classes: int = 1,
    architecture_params: Optional[Dict[str, Any]] = None,
    optimizer: Optimizer = Optimizer.ADAM,
    learning_rate: float = 0.001
) -> Model:
    """
    Build LSTM model with dynamic architecture from hyperparameters.

    Args:
        sequence_length: Number of time steps per sequence
        num_features: Number of features per time step
        task_type: 'regression', 'binary_classification', or 'multi_classification'
        num_classes: Number of output classes
        architecture_params: Dict with keys:
            - num_hidden_layers: int
            - hidden_units: List[int] - units per LSTM layer
            - dropout_rate: float
        optimizer: Optimizer enum
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras Model
    """
    if architecture_params is None:
        architecture_params = {
            'num_hidden_layers': 3,
            'hidden_units': [64, 64, 32],
            'dropout_rate': 0.2,
        }

    num_layers = architecture_params.get('num_hidden_layers', 3)
    hidden_units = architecture_params.get('hidden_units', [64] * num_layers)
    dropout_rate = architecture_params.get('dropout_rate', 0.2)

    if len(hidden_units) < num_layers:
        hidden_units = hidden_units + [hidden_units[-1]] * (num_layers - len(hidden_units))

    model = Sequential()

    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        if i == 0:
            model.add(LSTM(hidden_units[i], return_sequences=return_sequences,
                           input_shape=(sequence_length, num_features)))
        else:
            model.add(LSTM(hidden_units[i], return_sequences=return_sequences))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Output layer and compilation based on task type
    if task_type == 'regression':
        model.add(Dense(1))
        loss = LossFunction.MEAN_SQUARED_ERROR.value
        metrics = resolve_metrics([Metric.MEAN_ABSOLUTE_ERROR])
    elif task_type == 'binary_classification':
        model.add(Dense(1, activation=LayerActivation.SIGMOID.value))
        loss = LossFunction.BINARY_CROSSENTROPY.value
        metrics = resolve_metrics([Metric.ACCURACY])
    else:  # multi_classification
        model.add(Dense(num_classes, activation=LayerActivation.SOFTMAX.value))
        loss = LossFunction.CATEGORICAL_CROSSENTROPY.value
        metrics = resolve_metrics([Metric.ACCURACY])

    optimizer_instance = _create_optimizer_with_lr(optimizer, learning_rate)
    model.compile(optimizer=optimizer_instance, loss=loss, metrics=metrics)

    return model


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

def GetModelTemplateLSTMBinaryClassification(numberOfSteps: int, numberOfFeatures: int, optimizer: Optimizer = Optimizer.ADAM) -> Model:
    '''
    Returns a template for an LSTM model for binary classification tasks on sequential data.
    '''
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(numberOfSteps, numberOfFeatures)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(1, activation=LayerActivation.SIGMOID.value)
    ])

    model.compile(optimizer=optimizer.value, loss=LossFunction.BINARY_CROSSENTROPY.value, metrics=resolve_metrics([Metric.ACCURACY]))

    return model

def GetModelTemplateLSTMMultiClassification(numberOfSteps: int, numberOfFeatures: int, num_classes: int, optimizer: Optimizer = Optimizer.ADAM) -> Model:
    '''
    Returns a template for an LSTM model for multi-class classification tasks on sequential data.
    '''
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(numberOfSteps, numberOfFeatures)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(num_classes, activation=LayerActivation.SOFTMAX.value)
    ])

    model.compile(optimizer=optimizer.value, loss=LossFunction.CATEGORICAL_CROSSENTROPY.value, metrics=resolve_metrics([Metric.ACCURACY]))

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
