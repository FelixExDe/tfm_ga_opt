import os
import argparse
import time
import random
import numpy as np
import pandas as pd
import json
from datetime import datetime

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from pymoo.core.callback import Callback

from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.console import Console
from rich.text import Text

from data_load import load_pump_data
from classifier_model import ClassifierModel
from sklearn.metrics import precision_score, recall_score

# --- HYPERPARAMETER SPACE DEFINITIONS ---
HPARAM_SPACES = {
    'knn': {
        'params_ordered': ['n_neighbors', 'weights', 'p', 'algorithm', 'leaf_size'],
        'config': {
            'n_neighbors': {'type': 'integer', 'range': [1, 50]},
            'weights': {'type': 'categorical', 'choices': ['uniform', 'distance']},
            'p': {'type': 'integer', 'range': [1, 5]},
            'algorithm': {'type': 'categorical', 'choices': ['auto', 'ball_tree', 'kd_tree', 'brute']},
            'leaf_size': {'type': 'integer', 'range': [10, 50]},
        }
    },
    'logistic': {
        'params_ordered': ['C', 'solver', 'max_iter'],
        'fixed_params': {'penalty': 'l2'},
        'config': {
            'C': {'type': 'float', 'range': [0.001, 100.0], 'log_scale': True},
            'solver': {'type': 'categorical', 'choices': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']},
            'max_iter': {'type': 'integer', 'range': [100, 500]},
        }
    },
    'random_forest': {
        'params_ordered': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'criterion'],
        'config': {
            'n_estimators': {'type': 'integer', 'range': [50, 300]},
            'max_depth': {'type': 'integer', 'range': [5, 50]},
            'min_samples_split': {'type': 'integer', 'range': [2, 20]},
            'min_samples_leaf': {'type': 'integer', 'range': [1, 20]},
            'criterion': {'type': 'categorical', 'choices': ['gini', 'entropy', 'log_loss']},
        }
    }
}


# --- CUSTOM CALLBACK FOR RICH PROGRESS BAR AND HISTORY ---
class RichProgressPymooCallback(Callback):
    def __init__(self, progress_bar_task_id, rich_progress_object, problem_instance, show_pymoo_verbose=False):
        super().__init__()
        self.task_id = progress_bar_task_id
        self.progress = rich_progress_object
        self.problem = problem_instance
        self.show_pymoo_verbose = show_pymoo_verbose
        self.history_data = []
        self.last_gen_processed_by_bar = 0


    def notify(self, algorithm, **kwargs):
        completed_gens = algorithm.n_gen
        if completed_gens > self.last_gen_processed_by_bar:
            self.progress.update(self.task_id, completed=completed_gens)
            self.last_gen_processed_by_bar = completed_gens

        if not self.show_pymoo_verbose:
            metrics_str = ""
            if algorithm.opt is not None and len(algorithm.opt) > 0:
                best_f1 = -np.min(algorithm.opt.get("F")[:, 0])
                best_f2 = -np.min(algorithm.opt.get("F")[:, 1])
                metrics_str = f" Best(P):{best_f1:.3f}, Best(R):{best_f2:.3f}, NDS:{len(algorithm.opt)}"
            else:
                metrics_str = " Optimizing..."
            self.progress.update(self.task_id, metrics=metrics_str)

        current_pareto_X = algorithm.opt.get("X")
        current_pareto_F = algorithm.opt.get("F")

        if current_pareto_X is not None and current_pareto_F is not None and len(current_pareto_X) > 0:
            generation_solutions = []
            for i in range(len(current_pareto_X)):
                chromosome = current_pareto_X[i]
                objectives_pymoo = current_pareto_F[i]

                decoded_params = self.problem._decode_chromosome(chromosome)

                solution_data = {
                    "hyperparameters": decoded_params,
                    "objectives": {
                        "macro_precision": -objectives_pymoo[0], # Negate back to original value
                        "macro_recall": -objectives_pymoo[1]
                    }
                }
                generation_solutions.append(solution_data)

            self.history_data.append({
                "generation_number": algorithm.n_gen,
                "num_evaluations_so_far": algorithm.evaluator.n_eval,
                "pareto_front_solutions": generation_solutions
            })

# --- PYMOO PROBLEM DEFINITION ---
class HyperparameterOptimizationProblem(ElementwiseProblem):
    def __init__(self, model_type, hparam_details, X_train_t, y_train_s, X_val_t, y_val_s, random_state_model, verbose_level=0):
        self.model_type = model_type
        self.hparam_params_ordered = hparam_details['params_ordered']
        self.hparam_config = hparam_details['config']
        self.fixed_params = hparam_details.get('fixed_params', {})
        self.verbose_level = verbose_level

        self.X_train_transformed = X_train_t
        self.y_train = y_train_s
        self.X_val_transformed = X_val_t
        self.y_val = y_val_s
        self.random_state_model = random_state_model

        n_var = len(self.hparam_params_ordered)
        self.xl_internal = []
        self.xu_internal = []

        for param_name in self.hparam_params_ordered:
            conf = self.hparam_config[param_name]
            if conf['type'] == 'categorical':
                self.xl_internal.append(0)
                self.xu_internal.append(len(conf['choices']) - 1)
            elif conf['type'] == 'integer':
                self.xl_internal.append(conf['range'][0])
                self.xu_internal.append(conf['range'][1])
            elif conf['type'] == 'float':
                if conf.get('log_scale', False):
                    self.xl_internal.append(np.log10(conf['range'][0]))
                    self.xu_internal.append(np.log10(conf['range'][1]))
                else:
                    self.xl_internal.append(conf['range'][0])
                    self.xu_internal.append(conf['range'][1])

        super().__init__(n_var=n_var, n_obj=2, xl=np.array(self.xl_internal), xu=np.array(self.xu_internal))

    def _decode_chromosome(self, x_pymoo):
        params_decoded = {}
        for i, param_name in enumerate(self.hparam_params_ordered):
            conf = self.hparam_config[param_name]
            val_from_pymoo = x_pymoo[i]

            if conf['type'] == 'categorical':
                choice_index = int(round(val_from_pymoo))
                choice_index = max(int(self.xl_internal[i]), min(choice_index, int(self.xu_internal[i])))
                if param_name == 'gamma' and isinstance(conf['choices'][choice_index], (float, int)):
                    params_decoded[param_name] = float(conf['choices'][choice_index])
                else:
                    params_decoded[param_name] = conf['choices'][choice_index]

            elif conf['type'] == 'integer':
                int_val = int(round(val_from_pymoo))
                params_decoded[param_name] = max(int(self.xl_internal[i]), min(int_val, int(self.xu_internal[i])))
            elif conf['type'] == 'float':
                clipped_val = max(self.xl_internal[i], min(val_from_pymoo, self.xu_internal[i]))
                if conf.get('log_scale', False):
                    params_decoded[param_name] = 10**clipped_val
                else:
                    params_decoded[param_name] = float(clipped_val)

        if self.model_type == 'mlp' and 'hidden_layer_sizes_neurons' in params_decoded:
             params_decoded['hidden_layer_sizes'] = (params_decoded['hidden_layer_sizes_neurons'],)
             del params_decoded['hidden_layer_sizes_neurons']

        params_decoded.update(self.fixed_params)

        if self.model_type == 'logistic':
            if params_decoded.get('penalty') != 'elasticnet' and 'l1_ratio' in params_decoded:
                 del params_decoded['l1_ratio']
        return params_decoded

    def _evaluate(self, x_pymoo, out, *args, **kwargs):
        decoded_params_for_eval = {}
        try:
            model_params = self._decode_chromosome(x_pymoo)
            decoded_params_for_eval = model_params

            classifier_verbose_level = 0 if self.verbose_level < 2 else (self.verbose_level -1)

            if self.verbose_level >= 2:
                 print(f"  Evaluating GA individual with params: {model_params}")

            classifier = ClassifierModel(
                model_type=self.model_type,
                model_params=model_params,
                random_state=self.random_state_model,
                verbose=classifier_verbose_level
            )

            classifier.train(self.X_train_transformed, self.y_train)
            print("---Starting evaluation on validation set---")
            y_pred_val = classifier.predict(self.X_val_transformed)
            print("---Finished evaluation on validation set---")

            macro_precision = precision_score(self.y_val, y_pred_val, average='macro', zero_division=0)
            macro_recall = recall_score(self.y_val, y_pred_val, average='macro', zero_division=0)

            out["F"] = [-macro_precision, -macro_recall]

        except Exception as e:
            error_message = str(e)
            if self.verbose_level > 0:
                print(f"  Error evaluating GA individual (decoded: {decoded_params_for_eval}): {error_message[:300]}{'...' if len(error_message) > 300 else ''}. Assigning poor fitness.")
            out["F"] = [1e6, 1e6]


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning using Pymoo (NSGA-II).")
    parser.add_argument("--model_type", type=str, required=True, choices=list(HPARAM_SPACES.keys()),
                        help="Type of classifier model to tune.")
    parser.add_argument("--data_version", type=str, required=True, help="Data version string for load_pump_data.")
    parser.add_argument("--resample", type=str, default="False", help="Whether to resample the data.")
    parser.add_argument("--pop_size", type=int, default=40, help="Population size for GA.")
    parser.add_argument("--n_gen", type=int, default=50, help="Number of generations for GA.")
    parser.add_argument("--results_dir", type=str, default="./ga_results", help="Directory to save Pareto front CSV and history JSON.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation set size for data loading.")
    parser.add_argument("--file_root", type=str, default="./", help="Root directory for data loading.")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                        help="Verbosity level: 0=Rich progress bar only, 1=Rich progress + Pymoo summary, 2=Rich progress + Pymoo summary + individual eval details. Default: 1")
    args = parser.parse_args()


    pymoo_verbose_output = (args.verbose >= 1)
    script_print_verbose = (args.verbose >=1)
    console = Console()
    resample_data = True if args.resample.lower() == 'true' else False
    if resample_data not in [True, False]:
        console.print("[bold red]Error: --resample must be 'True' or 'False'.[/bold red]")
        return

    if script_print_verbose:
        console.rule("[bold cyan]Pymoo Hyperparameter Optimization Setup[/bold cyan]")
        console.print(f"Model Type: [bold magenta]{args.model_type}[/bold magenta]")
        console.print(f"Data Version: [bold magenta]{args.data_version}[/bold magenta]")
        console.print(f"Resample Data: [cyan]{resample_data}[/cyan]")
        console.print(f"Population Size: [cyan]{args.pop_size}[/cyan], Generations: [cyan]{args.n_gen}[/cyan]")
        console.print(f"Random State: [cyan]{args.random_state}[/cyan]")
        console.print(f"Results will be saved in: [green]{args.results_dir}[/green]")
        console.print(f"Script Verbosity: {args.verbose} (Pymoo internal verbose: {pymoo_verbose_output})")


    random.seed(args.random_state)
    np.random.seed(args.random_state)

    if script_print_verbose: console.print("\n[yellow]Loading and preprocessing data...[/yellow]")
    try:
        X_train_t, y_train_s, X_val_t, y_val_s, _ = load_pump_data(
            data_version=args.data_version, resample=resample_data,
            random_state=args.random_state, val_size=args.val_size, file_root=args.file_root
        )
        if script_print_verbose: console.print("[green]Data loaded and preprocessed successfully.[/green]")
    except Exception as e:
        console.print(f"[bold red]ERROR: Could not load data. Exiting. Details: {e}[/bold red]"); return

    if args.model_type not in HPARAM_SPACES:
        console.print(f"[bold red]Error: Hyperparameter space for model_type '{args.model_type}' is not defined.[/bold red]")
        return


    selected_hparam_details = HPARAM_SPACES[args.model_type]
    problem = HyperparameterOptimizationProblem(
        model_type=args.model_type, hparam_details=selected_hparam_details,
        X_train_t=X_train_t, y_train_s=y_train_s, X_val_t=X_val_t, y_val_s=y_val_s,
        random_state_model=args.random_state,
        verbose_level=args.verbose
    )

    algorithm = NSGA2(pop_size=args.pop_size)
    termination = get_termination("n_gen", args.n_gen)

    if script_print_verbose:
        console.print(f"\n[yellow]Running NSGA-II optimization for {args.n_gen} generations with pop_size {args.pop_size}...[/yellow]")

    start_time = time.time()
    progress_columns = [
        TextColumn("[progress.description]{task.description}"), BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TextColumn("•"),
        TimeElapsedColumn(), TextColumn("•"), TimeRemainingColumn(),
        TextColumn("{task.fields[metrics]}"),
    ]

    with Progress(*progress_columns, console=console, transient=(not pymoo_verbose_output)) as progress:
        main_task = progress.add_task("Optimizing", total=args.n_gen, metrics="")

        rich_callback = RichProgressPymooCallback(
            progress_bar_task_id=main_task,
            rich_progress_object=progress,
            problem_instance=problem,
            show_pymoo_verbose=pymoo_verbose_output
        )

        res = minimize(
            problem, algorithm, termination,
            seed=args.random_state, save_history=False,
            verbose=pymoo_verbose_output,
            callback=rich_callback
        )
    end_time = time.time()

    if script_print_verbose:
        console.print(f"Optimization finished in [cyan]{(end_time - start_time)/60:.2f}[/cyan] minutes.")

    os.makedirs(args.results_dir, exist_ok=True)
    history_filename = f"optimization_history_{args.model_type}_{args.data_version}"
    if resample_data == True:
        history_filename += "_resampled"
    history_filename += f"_{args.pop_size}I_{args.n_gen}G_{args.random_state}.json"
    history_filepath = os.path.join(args.results_dir, history_filename)

    full_history_output = {
        "metadata": {
            "model_type": args.model_type,
            "data_version": args.data_version,
            "resample": resample_data,
            "pop_size": args.pop_size,
            "n_gen_total": args.n_gen,
            "random_state": args.random_state,
            "optimization_start_time_iso": datetime.utcnow().isoformat() + "Z",
            "objectives_optimized": ["maximize_macro_precision", "maximize_macro_recall"]
        },
        "generations_pareto_fronts": rich_callback.history_data
    }
    try:
        with open(history_filepath, 'w') as f:
            json.dump(full_history_output, f, indent=4)
        if script_print_verbose:
            console.print(f"Per-generation Pareto front history saved to: [green]{history_filepath}[/green]")
        elif args.verbose == 0:
            console.print(f"History saved to: [green]{history_filepath}[/green]")
    except Exception as e:
        console.print(f"[bold red]Error saving history JSON: {e}[/bold red]")


if __name__ == "__main__":
    main()