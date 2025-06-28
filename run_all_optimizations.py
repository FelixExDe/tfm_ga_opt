import os
import sys
import subprocess
import time
import datetime
import concurrent.futures

# --- CONFIGURATION ---
MODEL_TYPES = [
               'knn',
               'logistic',
               'random_forest'
               ]
DATA_VERSIONS = ["7", "8"]
RESAMPLE_OPTIONS = [False, True]
RANDOM_STATES = [42, 17, 74, 5, 84, 19, 30, 26, 78, 21, 90, 68, 22, 75, 32,
                 56, 33, 8, 60, 67, 14, 7, 20, 58, 79, 92, 95, 98, 66, 1, 72
                ]
DEFAULT_POP_SIZE = 20
DEFAULT_N_GEN = 50
DEFAULT_RESULTS_DIR_BASE = "./ga_results"
DEFAULT_VAL_SIZE = 0.2
DEFAULT_FILE_ROOT = "./"
DEFAULT_VERBOSE_LEVEL_GENETIC_OPT = "2"

# --- SCRIPT SETTINGS for the orchestrator ---
MAX_CONCURRENT_JOBS = os.cpu_count() or 4 # 4 for Oracle Cloud, adjust as needed
OPTIMIZATION_SCRIPT_NAME = "genetic_optimization.py"
PYTHON_EXECUTABLE = sys.executable
MASTER_LOG_DIR = os.path.join(DEFAULT_RESULTS_DIR_BASE, "master_run_logs")

class OptimizationJob:
    def __init__(self, model_type, data_version, resample, pop_size, n_gen, results_dir_base, random_state, val_size, file_root, verbose_genetic_opt):
        self.model_type = model_type
        self.data_version = data_version
        self.resample = resample
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.results_dir_base = results_dir_base
        self.random_state = random_state
        self.val_size = val_size
        self.file_root = file_root
        self.verbose_genetic_opt = verbose_genetic_opt
        self.id = f"{self.model_type}_{self.data_version}"
        if self.resample:
            self.id += "_resampled"
        self.id += f"_{self.pop_size}I_{self.n_gen}G"
        self.status = "PENDING"
        self.process = None
        self.start_time = None
        self.end_time = None
        self.duration_str = "-"
        self.log_file_path = os.path.join(MASTER_LOG_DIR, f"{self.id}_{self.random_state}.log")
        self.expected_output_json = os.path.join(
            self.results_dir_base,
            f"optimization_history_{self.id}_{self.random_state}.json"
        )

    def get_command(self):
        cmd = [
            PYTHON_EXECUTABLE, OPTIMIZATION_SCRIPT_NAME,
            "--model_type", self.model_type, "--data_version", self.data_version,
            "--resample", str(self.resample), "--pop_size", str(self.pop_size),
            "--n_gen", str(self.n_gen), "--results_dir", self.results_dir_base,
            "--random_state", str(self.random_state), "--val_size", str(self.val_size),
            "--file_root", self.file_root, "--verbose", self.verbose_genetic_opt
        ]
        return cmd

    def check_if_already_completed(self):
        return os.path.exists(self.expected_output_json)

def create_job_list():
    jobs = []
    for random_state in RANDOM_STATES:
        for model in MODEL_TYPES:
            for version in DATA_VERSIONS:
                for resample in RESAMPLE_OPTIONS:
                    job = OptimizationJob(
                        model_type=model, data_version=version, resample=resample,
                        pop_size=DEFAULT_POP_SIZE, n_gen=DEFAULT_N_GEN,
                        results_dir_base=DEFAULT_RESULTS_DIR_BASE, random_state=random_state,
                        val_size=DEFAULT_VAL_SIZE, file_root=DEFAULT_FILE_ROOT,
                        verbose_genetic_opt=DEFAULT_VERBOSE_LEVEL_GENETIC_OPT
                    )
                    jobs.append(job)
    return jobs

def run_single_job(job: OptimizationJob) -> OptimizationJob:
    """
    Executes a single optimization job using subprocess.run to prevent I/O buffer deadlocks,
    updates its status, and returns the job object.
    """
    command_to_run = job.get_command()
    job.status = "RUNNING"
    job.start_time = time.time()

    print(f"[STARTING] Job: {job.id} -> Log: {job.log_file_path}")

    proc_env = os.environ.copy()

    # Limit the number of threads for scientific libraries to prevent over-subscription.
    # This forces each optimization process to run on a single core,
    # leaving the high-level parallelism to ProcessPoolExecutor.
    proc_env["OMP_NUM_THREADS"] = "1"
    proc_env["MKL_NUM_THREADS"] = "1"
    proc_env["OPENBLAS_NUM_THREADS"] = "1"

    try:
        with open(job.log_file_path, 'w') as log_f:
            process_result = subprocess.run(
                command_to_run,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=50 * 3600,
                check=False,
                env=proc_env
            )

        job.end_time = time.time()
        duration = job.end_time - job.start_time
        job.duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))

        # Append duration and exit code to the log
        with open(job.log_file_path, 'a') as log_f:
            log_f.write(f"\nJob Duration: {job.duration_str}\n")
            log_f.write(f"Exit Code: {process_result.returncode}\n")

        if process_result.returncode == 0 and job.check_if_already_completed():
            job.status = "COMPLETED"
        elif process_result.returncode == 0 and not job.check_if_already_completed():
            job.status = "FAILED (Output Missing)"
        else:
            job.status = f"FAILED (Exit Code: {process_result.returncode})"

    except subprocess.TimeoutExpired:
        job.status = "FAILED (Timeout)"
        job.end_time = time.time()
        duration = job.end_time - job.start_time
        job.duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
        with open(job.log_file_path, 'a') as log_f:
            log_f.write(f"\n--- JOB TIMED OUT AFTER {job.duration_str} ---\n")

    except Exception as e:
        job.status = f"FAILED (Launch Error: {e})"
        job.end_time = time.time()
        if job.start_time:
             duration = job.end_time - job.start_time
             job.duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
        else:
            job.duration_str = "Error"

    print(f"[{job.status}] Job: {job.id} | Duration: {job.duration_str}")
    return job

def run_batch_optimizations_parallel():
    print(f"--- Master Optimization Runner (Parallel Mode) ---")
    print(f"Max concurrent jobs: {MAX_CONCURRENT_JOBS}")
    os.makedirs(MASTER_LOG_DIR, exist_ok=True)
    os.makedirs(DEFAULT_RESULTS_DIR_BASE, exist_ok=True)

    # Generate the full list of jobs and check for pre-completed ones
    all_jobs_to_run = []
    skipped_jobs = []
    initial_job_list = create_job_list()

    for job in initial_job_list:
        if job.check_if_already_completed():
            job.status = "COMPLETED (Skipped - Output Exists)"
            job.duration_str = "N/A (Skipped)"
            skipped_jobs.append(job)
        else:
            all_jobs_to_run.append(job)

    total_jobs = len(initial_job_list)
    print(f"Generated {total_jobs} total jobs.")
    print(f"{len(skipped_jobs)} jobs already completed and will be skipped.")
    print(f"Submitting {len(all_jobs_to_run)} new jobs to the worker pool...")

    completed_jobs = []

    # Use a ProcessPoolExecutor to manage a pool of worker processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        # Submit all jobs to the pool. `submit` returns a Future object.
        future_to_job = {executor.submit(run_single_job, job): job for job in all_jobs_to_run}

        # Use `as_completed` to process jobs as they finish, in any order.
        for future in concurrent.futures.as_completed(future_to_job):
            try:
                # The result() of the future is the updated job object returned by run_single_job
                completed_job = future.result()
                completed_jobs.append(completed_job)
            except Exception as exc:
                # Handle potential exceptions from the worker function itself
                original_job = future_to_job[future]
                original_job.status = f"FAILED (Executor Error: {exc})"
                print(f"Job {original_job.id} generated an exception: {exc}")
                completed_jobs.append(original_job)

    # --- Final Summary ---
    print("\n--- Batch Optimization Summary ---")
    final_job_list = sorted(skipped_jobs + completed_jobs, key=lambda j: j.id)

    completed_count = 0
    failed_count = 0
    skipped_count = len(skipped_jobs)

    for job in final_job_list:
        print(f"Job ID: {job.id:<50} | Status: {job.status:<30} | Duration: {job.duration_str}")
        if "COMPLETED" in job.status:
            completed_count += 1
        elif "FAILED" in job.status:
            failed_count += 1

    print(f"\nTotal Jobs: {total_jobs}")
    print(f"Completed (including skipped): {completed_count}")
    print(f"Failed: {failed_count}")
    print(f"Successfully Processed in this run: {completed_count - skipped_count}")


if __name__ == "__main__":
    run_batch_optimizations_parallel()