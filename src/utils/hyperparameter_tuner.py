"""
Hyperparameter Tuning Module using Optuna.

Provides OptunaHyperparameterTuner class for automated hyperparameter
optimization of Keras models with MLflow integration.
"""

import optuna
from optuna.integration import TFKerasPruningCallback
import mlflow
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable, List

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adamax, Nadam
from tensorflow.keras.callbacks import EarlyStopping

from config_load import Config


class OptunaHyperparameterTuner:
    """
    Orchestrates Optuna-based hyperparameter tuning for Keras models.

    Attributes:
        config: Configuration object with HYPERPARAMETER_TUNING section
        task_type: 'regression', 'binary_classification', or 'multi_classification'
        num_features: Number of input features
        num_classes: Number of output classes (1 for regression)
        mlflow_logger: Optional MLFlowLogger instance for trial tracking
    """

    def __init__(
        self,
        config: Config,
        task_type: str,
        num_features: int,
        num_classes: int,
        mlflow_logger: Optional[Any] = None,
        invocation_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
        sequence_length: Optional[int] = None
    ):
        self.config = config
        self.task_type = task_type
        self.num_features = num_features
        self.num_classes = num_classes
        self.mlflow_logger = mlflow_logger
        self.invocation_id = invocation_id
        self.experiment_name = experiment_name
        self.sequence_length = sequence_length

        # Extract tuning config
        self.tuning_config = getattr(config, 'HYPERPARAMETER_TUNING', {})
        self.search_space = self.tuning_config.get('SEARCH_SPACE', {})

    def _get_optimizer(self, optimizer_name: str, learning_rate: float):
        """Map optimizer name to Keras optimizer with learning rate."""
        optimizers = {
            'adam': Adam(learning_rate=learning_rate),
            'sgd': SGD(learning_rate=learning_rate),
            'rmsprop': RMSprop(learning_rate=learning_rate),
            'adamax': Adamax(learning_rate=learning_rate),
            'nadam': Nadam(learning_rate=learning_rate),
        }
        return optimizers.get(optimizer_name.lower(), Adam(learning_rate=learning_rate))

    def _build_model_from_params(self, params: Dict[str, Any]) -> Sequential:
        """Build a Keras model from sampled hyperparameters."""
        model = Sequential()

        num_layers = params['num_hidden_layers']
        dropout_rate = params.get('dropout_rate', 0.0)
        activation = params.get('activation', 'relu')

        if self.sequence_length is not None:
            # LSTM model for 3D sequence data (samples, seq_len, features)
            for i in range(num_layers):
                units_key = f'hidden_units_{i}'
                units = params.get(units_key, params['hidden_units_0'])
                return_sequences = (i < num_layers - 1)
                if i == 0:
                    model.add(LSTM(units, return_sequences=return_sequences,
                                   input_shape=(self.sequence_length, self.num_features)))
                else:
                    model.add(LSTM(units, return_sequences=return_sequences))
                if dropout_rate > 0:
                    model.add(Dropout(dropout_rate))
        else:
            # MLP model for 2D tabular data
            model.add(Dense(
                params['hidden_units_0'],
                activation=activation,
                input_shape=(self.num_features,)
            ))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
            for i in range(1, num_layers):
                units_key = f'hidden_units_{i}'
                units = params.get(units_key, params['hidden_units_0'])
                model.add(Dense(units, activation=activation))
                if dropout_rate > 0:
                    model.add(Dropout(dropout_rate))

        # Output layer based on task type
        if self.task_type == 'regression':
            model.add(Dense(1))
            loss = 'mean_squared_error'
            metrics = ['mean_squared_error']
        elif self.task_type == 'binary_classification':
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:  # multi_classification
            model.add(Dense(self.num_classes, activation='softmax'))
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']

        optimizer = self._get_optimizer(params['optimizer'], params['learning_rate'])
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model

    def create_objective(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Callable:
        """
        Create Optuna objective function for the given data.

        Returns:
            Callable objective function for optuna.Study.optimize()
        """
        arch_space = self.search_space.get('ARCHITECTURE', {})
        train_space = self.search_space.get('TRAINING', {})
        tuning_epochs = self.tuning_config.get('TUNING_EPOCHS', 30)
        log_trials = self.tuning_config.get('MLFLOW_TRACKING', {}).get('LOG_TRIALS', True)

        def objective(trial: optuna.Trial) -> float:
            # Sample architecture hyperparameters
            num_layers_cfg = arch_space.get('NUM_HIDDEN_LAYERS', {'low': 1, 'high': 3})
            num_hidden_layers = trial.suggest_int(
                'num_hidden_layers',
                num_layers_cfg['low'],
                num_layers_cfg['high']
            )

            units_cfg = arch_space.get('HIDDEN_UNITS', {'low': 32, 'high': 256, 'step': 32})
            hidden_units = {}
            for i in range(num_hidden_layers):
                hidden_units[f'hidden_units_{i}'] = trial.suggest_int(
                    f'hidden_units_{i}',
                    units_cfg['low'],
                    units_cfg['high'],
                    step=units_cfg.get('step', 32)
                )

            dropout_cfg = arch_space.get('DROPOUT_RATE', {'low': 0.0, 'high': 0.5})
            dropout_rate = trial.suggest_float(
                'dropout_rate',
                dropout_cfg['low'],
                dropout_cfg['high'],
                step=dropout_cfg.get('step', 0.1)
            )

            activation_choices = arch_space.get('ACTIVATION', ['relu', 'tanh'])
            activation = trial.suggest_categorical('activation', activation_choices)

            # Sample training hyperparameters
            lr_cfg = train_space.get('LEARNING_RATE', {'low': 1e-5, 'high': 1e-2, 'log': True})
            learning_rate = trial.suggest_float(
                'learning_rate',
                lr_cfg['low'],
                lr_cfg['high'],
                log=lr_cfg.get('log', True)
            )

            batch_choices = train_space.get('BATCH_SIZE', [32, 64, 128])
            batch_size = trial.suggest_categorical('batch_size', batch_choices)

            optimizer_choices = train_space.get('OPTIMIZER', ['adam', 'rmsprop'])
            optimizer = trial.suggest_categorical('optimizer', optimizer_choices)

            # Build params dict
            params = {
                'num_hidden_layers': num_hidden_layers,
                **hidden_units,
                'dropout_rate': dropout_rate,
                'activation': activation,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'optimizer': optimizer,
            }

            # Build and train model
            model = self._build_model_from_params(params)

            # Callbacks for pruning and early stopping
            callbacks = [
                TFKerasPruningCallback(trial, 'val_loss'),
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=0
                )
            ]

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=tuning_epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )

            # Get best validation metric
            val_loss = min(history.history['val_loss'])

            # Log trial to MLflow if enabled
            if log_trials and self.mlflow_logger:
                self._log_trial_to_mlflow(trial, params, val_loss)

            return val_loss

        return objective

    def _log_trial_to_mlflow(self, trial: optuna.Trial, params: Dict, val_loss: float):
        """Log individual trial results to MLflow as nested run."""
        try:
            run_name = f"D3S2T{trial.number:03d}_HPO_Trial_{trial.number:03d}"

            # Build tags consistent with other pipeline runs
            tags = {
                "task_type": "hyperparameter_tuning_trial",
                "model_type": self.task_type,
                "trial_number": str(trial.number),
                "pipeline_step": "D3S2",
                "dag": "3",
                "dag_step": "2"
            }

            if self.invocation_id:
                tags["invocation_id"] = self.invocation_id
            if self.experiment_name:
                tags["experiment_name"] = self.experiment_name

            with mlflow.start_run(nested=True, run_name=run_name, tags=tags):
                # Log params (convert to strings for MLflow compatibility)
                mlflow_params = {k: str(v) if not isinstance(v, (int, float, str, bool)) else v
                                 for k, v in params.items()}
                mlflow.log_params(mlflow_params)
                mlflow.log_metric('val_loss', val_loss)
                mlflow.log_metric('trial_number', trial.number)
        except Exception as e:
            print(f"Warning: Failed to log trial {trial.number} to MLflow: {e}")

    def run_study(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        study_name: str = "hyperparameter_tuning"
    ) -> Tuple[Dict[str, Any], optuna.Study]:
        """
        Execute Optuna hyperparameter optimization study.

        Returns:
            Tuple of (best_params_dict, optuna.Study)
        """
        # Configure sampler
        sampler_type = self.tuning_config.get('SAMPLER', 'TPE')
        if sampler_type == 'TPE':
            sampler = optuna.samplers.TPESampler(seed=self.config.RANDOM_SEED if hasattr(self.config, 'RANDOM_SEED') else 42)
        elif sampler_type == 'RandomSampler':
            sampler = optuna.samplers.RandomSampler(seed=self.config.RANDOM_SEED if hasattr(self.config, 'RANDOM_SEED') else 42)
        elif sampler_type == 'CmaEsSampler':
            sampler = optuna.samplers.CmaEsSampler(seed=self.config.RANDOM_SEED if hasattr(self.config, 'RANDOM_SEED') else 42)
        else:
            sampler = optuna.samplers.TPESampler(seed=42)

        # Configure pruner
        pruner_config = self.tuning_config.get('PRUNER', {})
        pruner_type = pruner_config.get('TYPE', 'MedianPruner')
        if pruner_type == 'MedianPruner':
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=pruner_config.get('N_STARTUP_TRIALS', 5),
                n_warmup_steps=pruner_config.get('N_WARMUP_STEPS', 10),
                interval_steps=pruner_config.get('INTERVAL_STEPS', 1)
            )
        elif pruner_type == 'SuccessiveHalvingPruner':
            pruner = optuna.pruners.SuccessiveHalvingPruner()
        elif pruner_type == 'HyperbandPruner':
            pruner = optuna.pruners.HyperbandPruner()
        elif pruner_type == 'NopPruner':
            pruner = optuna.pruners.NopPruner()
        else:
            pruner = optuna.pruners.MedianPruner()

        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',  # minimize val_loss
            sampler=sampler,
            pruner=pruner
        )

        # Create objective
        objective = self.create_objective(X_train, y_train, X_val, y_val)

        # Run optimization
        n_trials = self.tuning_config.get('N_TRIALS', 50)
        timeout = self.tuning_config.get('TIMEOUT_SECONDS', 3600)

        print(f"Starting Optuna study with {n_trials} trials (timeout: {timeout}s)")
        print(f"  Sampler: {sampler_type}, Pruner: {pruner_type}")

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            catch=(Exception,)  # Continue on individual trial failures
        )

        print(f"\nStudy completed: {len(study.trials)} trials")
        print(f"  Best trial: {study.best_trial.number}")
        print(f"  Best val_loss: {study.best_value:.6f}")

        return study.best_params, study

    def get_best_params_for_training(self, best_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Optuna best_params to format expected by model training.

        Returns config dict with:
        - Architecture params for dynamic model building
        - Training params (epochs, batch_size, optimizer, learning_rate)
        """
        # Extract architecture params
        num_hidden_layers = best_params.get('num_hidden_layers', 2)
        hidden_units = [best_params.get(f'hidden_units_{i}', 64) for i in range(num_hidden_layers)]

        return {
            'architecture': {
                'num_hidden_layers': num_hidden_layers,
                'hidden_units': hidden_units,
                'dropout_rate': best_params.get('dropout_rate', 0.0),
                'activation': best_params.get('activation', 'relu'),
            },
            'training': {
                'learning_rate': best_params.get('learning_rate', 0.001),
                'batch_size': best_params.get('batch_size', 32),
                'optimizer': best_params.get('optimizer', 'adam'),
                'epochs': self.config.MODEL.get('EPOCHS', 200),  # Use full epochs for final training
                'early_stopping_patience': self.config.MODEL.get('EARLY_STOPPING_PATIENCE', 10),
                'validation_split': self.config.MODEL.get('VALIDATION_SPLIT', 0.2),
            }
        }

    def get_study_summary(self, study: optuna.Study) -> Dict[str, Any]:
        """
        Generate summary statistics from completed study.

        Returns dict with study metrics for MLflow logging.
        """
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        return {
            'n_trials_total': len(study.trials),
            'n_trials_completed': len(completed_trials),
            'n_trials_pruned': len(pruned_trials),
            'n_trials_failed': len(failed_trials),
            'best_val_loss': study.best_value if study.best_trial else None,
            'best_trial_number': study.best_trial.number if study.best_trial else None,
        }


__all__ = ['OptunaHyperparameterTuner']
