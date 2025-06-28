import os
import time
import argparse
import matplotlib
import numpy as np
import random
from classifier_model import ClassifierModel
from data_load import load_pump_data


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier model.")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=['logistic', 'knn', 'random_forest'],
                        help="Type of the model to evaluate.")
    parser.add_argument("--model_version", type=str, required=True,
                        help="Model version to load")
    parser.add_argument("--data_version", type=str, default="",
                        help="Data version to load. Empty for latest.")
    parser.add_argument("--resample", type=str, default="False",
                        choices=['True', 'False'],
                        help="Whether to resample the data.")
    parser.add_argument("--model_dir", type=str, default="./models",
                        help="Directory where models are saved.")
    parser.add_argument("--plot_action", type=str, default="none",
                        choices=['show', 'save', 'none'],
                        help="Action for confusion matrix: 'show', 'save' to PNG, or 'none'.")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation set size.")
    parser.add_argument("--file_root", type=str, default="./", help="Root directory for data loading.")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.plot_action == "save":
        try:
            # Use a non-interactive backend suitable for saving files without a GUI.
            matplotlib.use('Agg')
            print("Using 'Agg' backend for Matplotlib (for saving plots).")
        except Exception as e:
            print(f"Warning: Could not set Matplotlib backend to 'Agg'. Error: {e}")
    import matplotlib.pyplot as plt


    model_type_str = args.model_type
    model_version_str = args.model_version
    data_ver_str = args.data_version
    model_storage_dir = os.path.join(args.model_dir, model_type_str, model_version_str)
    stats_filename = os.path.join(model_storage_dir, f"{model_type_str}_average_stats.csv")

    if os.path.exists(stats_filename):
        print(f"Stats file already exists: {stats_filename}")
        print("Please delete it before running the evaluation again.")
        return

    data_version_name_for_components = data_ver_str if data_ver_str else "latest"
    # Suffix for the plot title.
    plot_display_title_suffix = f"(DataVer: {data_version_name_for_components.upper()})"

    print(f"\n{'='*60}")
    print(f"Evaluating Model Type: {model_type_str.upper()}")
    print(f"Data Version for components: '{data_version_name_for_components}'")
    print(f"Plot Action: {args.plot_action}")
    print(f"{'='*60}")

    print("Loading models from storage directory:", model_storage_dir)
    models = os.listdir(model_storage_dir)
    models = [model for model in models if model.startswith(f"{model_type_str}_model")]
    seeds = []
    reports = []

    for model_file in models:
        try:
            classifier = ClassifierModel(
                model_type=model_type_str,
                model_version=model_version_str,
                model_dir=args.model_dir,
                filename=model_file,
                verbose=0,
            )
        except FileNotFoundError:
            generated_filename = classifier._generate_model_filename()
            print(f"ERROR: Model file '{generated_filename}' not found in '{model_storage_dir}'."); return
        except Exception as e:
            print(f"Error loading model: {e}"); return

        try:
            print("Loading and preprocessing validation data...")
            _, _, X_val_transformed, y_val, _ = \
                load_pump_data(
                    data_version=data_ver_str,
                    random_state=classifier.get_random_state(), val_size=args.val_size,
                    file_root=args.file_root
                )
            if X_val_transformed is None or y_val is None:
                print("Error: Validation data could not be loaded or is None."); return
            print("Validation data loaded and preprocessed successfully.")
            random.seed(classifier.get_random_state())
            np.random.seed(classifier.get_random_state())
        except Exception as e:
            print(f"ERROR during data loading/preprocessing. Details: {e}"); return

        try:
            predict_time = time.time()
            y_pred_val = classifier.predict(X_val_transformed)
            predict_duration = time.time() - predict_time
            print(f"Prediction on validation set took {predict_duration:.2f} seconds.")
            if y_pred_val is None: print("Prediction on validation set returned None."); return
        except Exception as e:
            print(f"Error during prediction: {e}"); return

        print("\n--- Detailed Evaluation ---")
        report = classifier.evaluate(y_val, y_pred_val, title_suffix=plot_display_title_suffix)
        reports.append(report)
        seeds.append(classifier.random_state)

        if args.plot_action == "show":
            print("Displaying confusion matrix...")
            classifier.show_confusion_matrix(y_val, y_pred_val, title_suffix=plot_display_title_suffix)
        elif args.plot_action == "save":
            print("Saving confusion matrix plot...")
            classifier.save_confusion_matrix_plot(
                y_val, y_pred_val,
                data_version_info=data_version_name_for_components,
                title_suffix=plot_display_title_suffix
            )
        elif args.plot_action == "none":
            print("Skipping confusion matrix plot.")

    print("\n--- Summary of Evaluation Reports ---")
    summary_keys = {'accuracy', 'macro avg', 'weighted avg'}
    class_labels = sorted([key for key in report.keys() if key not in summary_keys])

    # The following block summarizes the metrics from the LAST evaluated model, not an average.
    if len(class_labels) != 3:
        print(f"  Warning: Expected 3 classes but found {len(class_labels)}: {class_labels}. Skipping per-class metrics.")
        report_data = {
            'Model ID': classifier.random_state,
            'Accuracy': report.get('accuracy', 0),
            'Macro Precision': report.get('macro avg', {}).get('precision', 0),
            'Macro Recall': report.get('macro avg', {}).get('recall', 0),
            'Macro F1-score': report.get('macro avg', {}).get('f1-score', 0)
        }
    else:
        report_data = {
            'Model ID': classifier.random_state,
            'Accuracy': report['accuracy'],
            'Macro Precision': report['macro avg']['precision'],
            'Macro Recall': report['macro avg']['recall'],
            'Macro F1-score': report['macro avg']['f1-score'],
            # Per-class precision
            f'Precision_{class_labels[0]}': report[class_labels[0]]['precision'],
            f'Precision_{class_labels[1]}': report[class_labels[1]]['precision'],
            f'Precision_{class_labels[2]}': report[class_labels[2]]['precision'],
            # Per-class recall
            f'Recall_{class_labels[0]}': report[class_labels[0]]['recall'],
            f'Recall_{class_labels[1]}': report[class_labels[1]]['recall'],
            f'Recall_{class_labels[2]}': report[class_labels[2]]['recall'],
        }

    print(f"Accuracy (last model): {report_data['Accuracy']:.4f}")
    print(f"Macro Precision (last model): {report_data['Macro Precision']:.4f}")
    print(f"Macro Recall (last model): {report_data['Macro Recall']:.4f}")
    print(f"Macro F1-score (last model): {report_data['Macro F1-score']:.4f}")

    # Save stats to a file.
    with open(stats_filename, 'w') as f:
        f.write("Model ID,Accuracy,Precision,Recall,F1 Score\n")
        for seed, r in zip(seeds, reports):
            f.write(f"{seed},{r['accuracy']:.4f},{r['macro avg']['precision']:.4f},{r['macro avg']['recall']:.4f},{r['macro avg']['f1-score']:.4f}\n")
    print(f"Evaluation stats saved to '{stats_filename}'.")

    print(f"\n{'='*60}")
    print("Model evaluation finished.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()