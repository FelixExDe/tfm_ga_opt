# Filename: classifier_model.py

import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

class ClassifierModel:
    def __init__(self, model_type: str, model_params: dict = None,
                 model_dir: str = './models', model_version: str = '', random_state: int = 42, filename: str = None,
                 verbose: int = 1):
        """
        Initializes the ClassifierModel.

        Args:
            model_type (str): Type of the model to use ('logistic', 'knn', etc.).
            model_params (dict, optional): Parameters for the scikit-learn model.
            model_dir (str, optional): Directory to save/load models.
            model_version (str, optional): A version name for the model, used for subdirectories.
            random_state (int, optional): Random state for reproducibility.
            filename (str, optional): If provided, load a model from this file instead of initializing a new one.
            verbose (int, optional): Verbosity level (0 for silent, 1 for major actions).
        """

        if model_type is None or model_type.lower() not in ['logistic', 'knn', 'random_forest']:
            raise ValueError("model_type must be one of: 'logistic', 'knn', 'random_forest'.")

        self.model_type = model_type.lower()
        self.model_version = model_version
        self.model_dir = model_dir
        self.verbose = verbose

        if filename is not None:
            if not os.path.exists(os.path.join(model_dir, model_type, model_version, filename)):
                raise FileNotFoundError(f"Model file {filename} does not exist in {model_dir}/{model_type}/{model_version}.")
            self._load_model(filename=filename)
        else:
            self.model_params_input = model_params if model_params is not None else {}
            self.random_state = random_state
            self.model = None
            self.model_version = model_version
            self.model_filename = None
            self._initialize_model()

        if self.verbose > 0:
            os.makedirs(self.model_dir, exist_ok=True)

    def _initialize_model(self):
        current_params = self.model_params_input.copy()
        default_base_params = {
            'logistic': {'solver': 'lbfgs',
                         'max_iter': 1000,
                         'C': 1.0,
                         'random_state': self.random_state,
                         'n_jobs': None},

            'knn': {'n_neighbors': 5,
                    'n_jobs': 4},

            'random_forest': {'n_estimators': 100,
                              'random_state': self.random_state,
                              'n_jobs': None}
        }
        if self.model_type not in default_base_params:
            raise ValueError(f"Unsupported model_type: {self.model_type}.")
        model_defaults = default_base_params[self.model_type].copy()
        final_params = {**model_defaults, **current_params}

        if self.model_type == 'logistic': self.model = LogisticRegression(**final_params)
        elif self.model_type == 'knn': self.model = KNeighborsClassifier(**final_params)
        elif self.model_type == 'random_forest': self.model = RandomForestClassifier(**final_params)

        self.model_actual_params = self.model.get_params()

        if self.verbose > 0:
            print(f"Initialized {self.model_type} model with params: {self.model_actual_params}")

    def _generate_model_filename(self):
        base_name = f"{self.model_type}_model"
        if 'random_state' in self.model_actual_params:
            base_name += f"_{self.model_actual_params['random_state']}"
        else:
            base_name += f"_{self.random_state}"
        return f"{base_name}.joblib"


    def train(self, X_train_transformed, y_train):
        if self.model is None: raise RuntimeError("Model not initialized.")
        if X_train_transformed is None or y_train is None: raise ValueError("Training data cannot be None.")

        if self.verbose > 0:
            print(f"\nTraining {self.model_type} model...")
        try:
            self.model.fit(X_train_transformed, y_train)

            if self.verbose > 0:
                print("Model training completed successfully.")
            self.model_filename_suggestion = self._generate_model_filename()
        except Exception as e:
            if self.verbose > 0:
                print(f"Error during model training: {e}")
            raise

    def predict(self, X_transformed):
        if self.model is None: raise RuntimeError("Model not trained or loaded.")
        if X_transformed is None: raise ValueError("Input data cannot be None.")
        if self.verbose > 0:
            print(f"Making predictions with {self.model_type} model...")
        return self.model.predict(X_transformed)

    def predict_proba(self, X_transformed):
        if self.model is None: raise RuntimeError("Model not trained or loaded.")
        if not hasattr(self.model, "predict_proba"):
            warnings.warn(f"{self.model_type} model does not support predict_proba. Returning None.")
            return None
        if X_transformed is None: raise ValueError("Input data cannot be None.")
        if self.verbose > 0:
            print(f"Predicting probabilities with {self.model_type} model...")
        return self.model.predict_proba(X_transformed)

    def evaluate(self, y_true, y_pred, title_suffix=""):
        if y_pred is None:
            if self.verbose > 0: print("Predictions are None, cannot evaluate.")
            return None
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        if self.verbose > 0:
            print(f"\n--- Evaluation Results for {self.model_type} {title_suffix} ---")
            print(f"Accuracy: {report['accuracy']:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, zero_division=0))
            print("--------------------------------------")
        return report

    def save_model(self, filename: str = None):
        if self.model is None: raise RuntimeError("No model to save.")
        filename_to_use = filename or getattr(self, 'model_filename_suggestion', self._generate_model_filename())
        self.model_filename = filename_to_use
        model_path = os.path.join(self.model_dir, self.model_type, self.model_version, filename_to_use)
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if self.verbose > 0:
            print(f"\nSaving trained {self.model_type} model to: {model_path}")
        try:
            joblib.dump(self.model, model_path)
            if self.verbose > 0: print("Model saved successfully.")
        except Exception as e:
            if self.verbose > 0: print(f"Error saving model: {e}")
            raise

    def _load_model(self, filename: str = None):
        filename_to_use = filename or self._generate_model_filename()
        model_path = os.path.join(self.model_dir, self.model_type, self.model_version, filename_to_use)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if self.verbose > 0:
            print(f"\nLoading model from: {model_path}")
        try:
            loaded_model = joblib.load(model_path)
            self.model = loaded_model
            self.model_filename = filename_to_use
            filename_random_state = int(filename_to_use.split('_')[-1].split('.')[0])
            self.random_state = getattr(self.model, 'random_state', filename_random_state)
            if hasattr(self.model, 'get_params'):
                self.model_actual_params = self.model.get_params()
            if self.verbose > 0:
                print(f"Model loaded successfully: {type(self.model)}")
        except Exception as e:
            if self.verbose > 0: print(f"Error loading model: {e}")
            raise

    def _create_confusion_matrix_plot(self, y_true, y_pred, title_suffix=""):
        if y_pred is None:
            if self.verbose > 0: print("Predictions are None, cannot create confusion matrix plot.")
            return None
        if hasattr(self.model, 'classes_'): labels = self.model.classes_
        else: labels = np.unique(np.concatenate((y_true, y_pred))); labels.sort()
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix for {self.model_type} {title_suffix}')
        ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
        plt.tight_layout()
        return fig

    def show_confusion_matrix(self, y_true, y_pred, title_suffix=""):
        fig = self._create_confusion_matrix_plot(y_true, y_pred, title_suffix=title_suffix)
        if fig:
            try:
                plt.show()
            except Exception as e:
                print(f"Error displaying plot: {e}. Ensure you have a GUI backend if not saving to file.")
            finally:
                plt.close(fig)

    def save_confusion_matrix_plot(self, y_true, y_pred, data_version_info: str, title_suffix: str = ""):
        fig = self._create_confusion_matrix_plot(y_true, y_pred, title_suffix=title_suffix)
        if fig:
            safe_model_type = self.model_type.replace(' ', '_')
            safe_data_version = data_version_info.replace(' ', '_')
            plot_filename = f"confusion_matrix_{safe_model_type}_{safe_data_version}.png"
            plots_dir = os.path.join(self.model_dir, "evaluation_plots")
            os.makedirs(plots_dir, exist_ok=True)
            plot_filepath = os.path.join(plots_dir, plot_filename)
            try:
                fig.savefig(plot_filepath)
                if self.verbose > 0:
                    print(f"Confusion matrix plot saved to: {plot_filepath}")
            except Exception as e:
                if self.verbose > 0: print(f"Error saving confusion matrix plot: {e}")
            finally:
                plt.close(fig)

    def get_random_state(self):
        return self.random_state