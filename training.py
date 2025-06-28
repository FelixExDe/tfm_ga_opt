# Filename: run_all_training.py

import os
import numpy as np
from classifier_model import ClassifierModel
import random
from data_load import load_pump_data

# --- Configuration ---
# Specify the data version from data_cleaning.py to use.
# An empty string '' will use the latest version as per load_pump_data logic.
DATA_VERSIONS_TO_TRAIN = ['7']

# Define model types supported by ClassifierModel
MODEL_TYPES_TO_TRAIN = [
    'logistic',
    'knn',
    'random_forest'
]

# Common settings
MODEL_STORAGE_DIR = './models'
MODEL_VERSION = 'q25'
RANDOM_STATES = [
    42, 17, 74, 5, 84, 19, 30, 26, 78, 21, 90, 68, 22, 75, 32,
    56, 33, 8, 60, 67, 14, 7, 20, 58, 79, 92, 95, 98, 66, 1, 72
]
N_SEED = 1
VALIDATION_SIZE = 0.2
RESAMPLE = True
FILE_ROOT_PATH = './'
VERBOSE = False
PARAMS = {
            'logistic': {'solver': 'lbfgs',
                         'max_iter': 1000,
                         'C': 1.0,
                         'n_jobs': 1},

            'knn': {'n_neighbors': 5,
                    'n_jobs': 4},

            'random_forest': {'n_estimators': 100,
                              'n_jobs': 4}
        }

# --- Main Training Loop ---
def main():
    print("Starting batch model training process...")
    os.makedirs(MODEL_STORAGE_DIR, exist_ok=True)

    for data_ver in DATA_VERSIONS_TO_TRAIN:
        for i in range(N_SEED):
            random_state = RANDOM_STATES[i]
            random.seed(random_state)
            np.random.seed(random_state)
            data_version_name_for_print = data_ver if data_ver else "LATEST"

            print(f"\n{'='*60}")
            print(f"Processing Data Version: '{data_version_name_for_print}'")
            print(f"{'='*60}")

            # load_pump_data returns already transformed data and handles preprocessor saving/loading.
            try:
                print(f"Loading and preprocessing data (Version: {data_version_name_for_print})...")
                # df_TEST_X_transformed is also returned by load_pump_data but not used in this training script.
                X_train_transformed, y_train, X_val_transformed, y_val, _ = \
                    load_pump_data(
                        data_version=data_ver,
                        random_state=random_state,
                        val_size=VALIDATION_SIZE,
                        file_root=FILE_ROOT_PATH,
                        resample=RESAMPLE,
                        verbose=VERBOSE
                    )
                print("Data loading and preprocessing complete.")
            except Exception as e:
                print(f"ERROR during data loading/preprocessing for Version: {data_version_name_for_print}. Skipping this combination.")
                print(f"Details: {e}")
                continue

            for model_type in MODEL_TYPES_TO_TRAIN:
                print(f"\n--- Training Model Type: {model_type.upper()} ---")

                try:
                    classifier = ClassifierModel(
                        model_type=model_type,
                        model_version=MODEL_VERSION,
                        model_dir=MODEL_STORAGE_DIR,
                        random_state=random_state,
                        model_params=PARAMS.get(model_type, {}),
                        verbose=VERBOSE
                    )

                    start_time = np.datetime64('now')
                    classifier.train(X_train_transformed, y_train)
                    end_time = np.datetime64('now')
                    print(f"Training completed in {end_time - start_time} seconds.")

                    classifier.save_model()

                except Exception as e:
                    print(f"ERROR during training/evaluation/saving of {model_type.upper()} for Version: {data_version_name_for_print}. Skipping this model.")
                    print(f"Details: {e}")

    print(f"\n{'='*60}")
    print("Batch model training process finished.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()