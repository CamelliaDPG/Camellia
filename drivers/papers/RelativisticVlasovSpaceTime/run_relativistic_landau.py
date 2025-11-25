#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
import platform

def detect_cpu_model():
    system = platform.system()

    # macOS: use system_profiler to get the full "Chip:" line
    if system == "Darwin":
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPHardwareDataType"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=60
            )
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("Chip:"):
                    sysinfo.append(("CPU model", line.split(":",1)[1].strip()))
                elif line.startswith("Total Number of Cores:"):
                    sysinfo.append(("Total Number of Cores", line.split(":",1)[1].strip()))
                elif line.startswith("Memory:"):
                    sysinfo.append(("Memory", line.split(":",1)[1].strip()))
            if sysinfo:
                return sysinfo
        except Exception:
            pass

    # Windows: use wmic to get the CPU Name
    if system == "Windows":
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5
            )
            # wmic prints:
            #   Name
            #   Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
            # so skip the header
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            if len(lines) >= 2:
                return lines[1]
        except Exception:
            pass

    # Linux: use lscpu to build a detailed string
    if system == "Linux":
        try:
            out = subprocess.check_output(
                ["lscpu"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5
            )
            info = {}
            for line in out.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    info[key.strip()] = val.strip()

            parts = []
            if "Model name" in info:
                parts.append(info["Model name"])
            if "CPU(s)" in info:
                parts.append(f"{info['CPU(s)']} CPUs")
            if "Thread(s) per core" in info:
                parts.append(f"{info['Thread(s) per core']} threads/core")
            if "Architecture" in info:
                parts.append(f"Arch: {info['Architecture']}")

            if parts:
                return " | ".join(parts)
        except Exception:
            pass

    # Fallback #1: parse /proc/cpuinfo for "model name"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass

    # Fallback #2: platform.processor()
    proc = platform.processor()
    if proc:
        return proc.strip()

    return "unknown"

def main():
    parser = argparse.ArgumentParser(
        description="Run the Relativistic Landau Space–Time driver with MPI, "
                    "capture console output and timing."
    )
    parser.add_argument(
        "--ranks",
        type=int,
        required=True,
        help="Number of MPI ranks to launch"
    )
    parser.add_argument(
        "--ur",
        type=int,
        required=True,
        help="Number of uniform refinements"
    )
    parser.add_argument(
        "--p",
        type=int,
        required=True,
        help="Polynomial order"
    )
    parser.add_argument(
        "--theta",
        type=float,
        required=True,
        help="Fraction of speed of light at u=1.3 (floating point)"
    )
    parser.add_argument(
        "--tWidth",
        type=float,
        required=True,
        help="Width of each time slab"
    )
    parser.add_argument(
        "--numTimeSlabs",
        type=float,
        required=True,
        help="Number of time slabs"
    )
    args = parser.parse_args()

    # Format theta so that the directory and logfile names match exactly args.theta
    theta_str = str(args.theta)

    # Build the output directory under ~/relativistic-landau/{theta}c/p{p}/ur{ur}/
    base_dir = os.path.expanduser("~")
    output_dir = os.path.join(
        base_dir,
        "relativistic-landau",
        f"{theta_str}c",
        f"p{args.p}",
        f"ur{args.ur}",
        f"tWidth{args.tWidth}",
        f"numTimeSlabs{args.numTimeSlabs}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Name of the logfile
    log_fname = f"console-relativistic-landau-graph-norm-{theta_str}c-p{args.p}-ur{args.ur}.log"
    log_path = os.path.join(output_dir, log_fname)

    # Construct the mpirun command
    mpirun_cmd = [
        "mpirun",
        "-np", str(args.ranks),
        "./drivers/papers/RelativisticVlasovSpaceTime/LandauRelativisticSpaceTime",
        "--useStandardAssembly",
        f"--theta={args.theta}",
        f"--p={args.p}",
        f"--numUniformRefinements={args.ur}",
        f"--tWidth={args.tWidth}",
        f"--numTimeSlabs={args.numTimeSlabs}",
        f"--outputDir={output_dir}"
    ]

    # Prepare environment
    env = os.environ.copy()
    env["OMP_PROC_BIND"]   = "spread"
    env["OMP_PLACES"]      = "threads"
    env["OMP_NUM_THREADS"] = "1"

    # Open the log file, write the mpirun invocation, run the job, and time it
    with open(log_path, "w") as log:
        # 1) First line: the exact mpirun invocation
        invocation = " ".join(mpirun_cmd)
        log.write(invocation + "\n")
        log.flush()

        # 2) Second line: detected CPU model
        cpu_model = detect_cpu_model()
        log.write(f"CPU model: {cpu_model}\n")
        log.flush()

        # 2) run and time
        start = time.time()
        proc = subprocess.Popen(
            mpirun_cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env
        )
        ret = proc.wait()
        end = time.time()

        # 3) last line: total time to completion
        elapsed = end - start
        log.write(f"Total time to completion: {elapsed:.3f} seconds\n")

    if ret != 0:
        sys.stderr.write(f"ERROR: mpirun exited with code {ret}\n")
        sys.exit(ret)

if __name__ == "__main__":
    main()
    
# Generated 10/17/25 by ChatGPT o4-mini-high.
# Prompt:
# I'd like a python script that accepts as inputs the number of mpi ranks (argument name: ranks), an integer number of uniform refinements (argument name: ur), an integer polynomial order (argument name: p), and a floating point speed of light (argument name: c), and then does the following:
# 
# 1. creates a new output directory (using mkdir -p) underneath ~/relativistic-landau/c{c-arg}/p{p-arg}/ur{ur-arg}/.
# 2. Sets environment variables OMP_PROC_BIND=spread OMP_PLACES=threads OMP_NUM_THREADS=1
# 2. Invokes the following:
# 
# mpirun -np {ranks-arg} ./drivers/papers/RelativisticVlasovSpaceTime/LandauRelativisticSpaceTime --useStandardAssembly --p={p-arg} --numUniformRefinements={ur-arg} --outputDir={output-directory}
# 
# It should capture the console output to a file beneath the output directory named console-relativistic-landau-graph-norm-c{c-arg}-p{p-arg}-ur{ur-arg}.log.   The first line of the console output should include the mpirun invocation.  The last line should report the total time to completion.
#
# Revision Prompts:
# - Please add something that will determine what processor the CPU is running, and record that to the console log.
#
# - For an M2 Ultra, that gives me just "arm" as the CPU model.  Could you revise to give me more specifics?  I'm particularly interested in having this distinguish between an M2 Ultra and an M3 Ultra.
# 
# - Please revise the CPU model detection to record everything reported back in the "Chip" line (except for the "Chip:" prefix).  If it can be made to work cross-platform on Intel processors (etc.) with a similar level of detail, that would be even better.
