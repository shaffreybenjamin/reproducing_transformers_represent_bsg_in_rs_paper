"""Multi-GPU job scheduler for the per-process training scripts.

Faithful port of epsilon-transformers' `launcher_cuda_parallel.py` (quantum-public)
scheduling model: discover the GPUs on this one (multi-GPU) machine, keep a queue of
jobs, and run ONE job per free GPU, recycling each GPU to the next queued job as jobs
finish. The control loop here mirrors theirs exactly:

    num_gpus = torch.cuda.device_count()
    available_gpus = list(range(num_gpus))
    while queue or running:
        while queue and available_gpus: pop a gpu + a job, Popen it pinned to that gpu
        reap finished jobs -> return their gpu to the pool
        time.sleep(5)

The only difference from their launcher: they pin a run to its GPU via the run config
(`device='cuda:gpu_id'`) plus a `--gpu_id` flag; we pin via `CUDA_VISIBLE_DEVICES=gpu_id`
(so each training script's `device="cuda"` maps to that one physical GPU). jax is pinned
to the CPU (`JAX_PLATFORMS=cpu`) since it is only used for the one-time MSP enumeration;
torch owns the GPU.

Run from the repo root (i.e. /workspace/repo on the pod):
    python reproduction/train/launch_parallel.py <epochs> [proc ...]
    python reproduction/train/launch_parallel.py 20000              # all 7 processes
    python reproduction/train/launch_parallel.py 20000 mess3 arch   # a chosen subset
"""

import os
import subprocess
import sys
import time

import torch

ALL_PROCESSES = ["mess3", "wing", "fern", "arch", "strata", "zero_one_random", "rrxor"]
HERE = os.path.dirname(os.path.abspath(__file__))            # reproduction/train
REPRODUCTION_DIR = os.path.dirname(HERE)                     # reproduction
LOG_DIR = os.path.join(REPRODUCTION_DIR, "train_logs")


def main() -> None:
    epochs = sys.argv[1] if len(sys.argv) > 1 else "20000"
    procs = sys.argv[2:] if len(sys.argv) > 2 else list(ALL_PROCESSES)
    unknown = [p for p in procs if p not in ALL_PROCESSES]
    if unknown:
        print(f"ERROR: unknown process(es) {unknown}; valid: {ALL_PROCESSES}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(LOG_DIR, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("ERROR: no CUDA GPUs visible to the scheduler", file=sys.stderr)
        sys.exit(1)
    # SLOTS_PER_GPU lets several processes share one physical GPU. These tiny models leave a
    # fast GPU mostly idle, so packing a few per GPU runs each near full speed (measured ~no
    # slowdown for 2 on a 4090) and finishes far sooner/cheaper than one-at-a-time. A "slot" is
    # just a GPU id; multiple slots map to the same physical GPU.
    slots_per_gpu = max(1, int(os.environ.get("SLOTS_PER_GPU", "1")))
    available_slots = [g for g in range(num_gpus) for _ in range(slots_per_gpu)]

    # Cap CPU threads per job so concurrent processes don't oversubscribe the host CPU.
    cpu_count = os.cpu_count() or 8
    peak_concurrent = max(1, min(len(available_slots), len(procs)))
    threads_per_job = max(1, cpu_count // peak_concurrent)
    print(f"GPUs: {num_gpus}  slots/gpu: {slots_per_gpu}  total slots: {len(available_slots)}  "
          f"jobs: {procs}  epochs: {epochs}  cpu_threads/job: {threads_per_job} (of {cpu_count})", flush=True)

    experiment_queue = list(procs)
    running_processes = []
    failures = []

    while experiment_queue or running_processes:
        while experiment_queue and available_slots:
            gpu_id = available_slots.pop(0)
            proc = experiment_queue.pop(0)

            script = os.path.join("reproduction", "train", f"train_{proc}.py")
            log_path = os.path.join(LOG_DIR, f"{proc}.log")
            log = open(log_path, "w")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)   # pin this run to one physical GPU
            env["JAX_PLATFORMS"] = "cpu"                # jax only does the one-time MSP enumeration
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[var] = str(threads_per_job)         # avoid CPU oversubscription across jobs

            p = subprocess.Popen(
                [sys.executable, "-u", script, str(epochs)],
                stdout=log, stderr=subprocess.STDOUT, env=env,
            )
            running_processes.append({"proc": proc, "gpu_id": gpu_id, "process": p, "log": log})
            print(f"Started {proc} (pid {p.pid}) on GPU {gpu_id} -> {log_path}", flush=True)

        for info in running_processes[:]:
            retcode = info["process"].poll()
            if retcode is not None:
                running_processes.remove(info)
                info["log"].close()
                available_slots.append(info["gpu_id"])
                if retcode == 0:
                    print(f"{info['proc']} on GPU {info['gpu_id']} finished successfully.", flush=True)
                else:
                    print(f"{info['proc']} on GPU {info['gpu_id']} FAILED (exit {retcode}).", flush=True)
                    failures.append(info["proc"])

        time.sleep(5)

    if failures:
        print(f">> DONE with failures: {failures}", flush=True)
        sys.exit(1)
    print(">> ALL JOBS DONE (success)", flush=True)


if __name__ == "__main__":
    main()
