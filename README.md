This repository contains the code used to reproduce the results presented in the article “Choose, Then Choose Again: Preserving Plurality of Objectives in Social-Ecological System Management.”

It includes three dynamical systems:
- lake phosphorus dynamics
- predator-prey dynamics
- savanna tree-grass dynamics

The core numerical routines are implemented in C++ (Pybind11 extension `viability`), and experiments/plots are driven by Python scripts.

## Project Layout

- `ViabilityKernelCPU.cpp`: viability kernel C++ core and module entry point
- `CaptureBasinCPU.cpp`: capture basin C++ core
- `ROptionSetsComputer.cpp`: reversible option sets C++ core
- `models/`: dynamics models exposed to Python
- `helpers/`: plotting, utilities, trajectory helpers
- `lake/run_lake.py`: lake experiment runner
- `savana/run_savana.py`: savanna experiment runner
- `prey_predateur/run_pp.py`: predator-prey experiment runner
- `prey_predateur/plot_ratio.py`: parameter sweep and ratio plot
- `complie_scripts/`: build command templates for macOS/Linux/Windows

## Requirements

- Python 3.10+ (same version used to build the extension)
- C++17 compiler
- OpenMP runtime/compiler support
- Python packages:
  - `numpy`
  - `matplotlib`
  - `pybind11`

Install Python dependencies:

```bash
python -m pip install numpy matplotlib pybind11
```

## Build the C++ Extension

Run build commands from the repository root (`choose-then-choose-again-main`).

### Linux

```bash
bash complie_scripts/linux.txt
```

### macOS (Conda + libomp)

```bash
bash complie_scripts/mac.txt
```

### Windows (x64 Native Tools Command Prompt for Visual Studio)

Open the **x64 Native Tools Command Prompt for Visual Studio**, navigate to the repository root, then run:

```bat
call complie_scripts\win.bat
```
This must be run from a Visual Studio developer command prompt so that the MSVC compiler (`cl`) is available.


After a successful build, you should see a file like:

- `viability.cpython-<python-version>-<platform>.so` (Linux/macOS), or
- `viability.cp<version>-win_amd64.pyd` (Windows).

Quick check:

```bash
python -c "import viability; print('viability import OK')"
```

## Run Experiments

Run from repository root.

### Lake

```bash
python -m lake.run_lake
```

Default outputs: `lake/d1.png`, `lake/d2.png`, `lake/O.png`, `lake/s.png`.

### Savanna

```bash
python -m savana.run_savana
```

Default outputs: `savana/10.png`, `savana/20.png`, `savana/100.PNG`.

### Predator-Prey

```bash
python -m prey_predateur.run_pp
```

Default output: `prey_predateur/10.png`, `prey_predateur/11.png`, `prey_predateur/15.png`, `prey_predateur/reapp.png`.

### Predator-Prey Ratio Sweep

```bash
python -m prey_predateur.plot_ratio
```

Default output: `prey_predateur/u.png`.

## Reproducibility Notes

- The experiments reported here were run on an **AMD Ryzen Threadripper PRO 7985WX 64-Cores** machine. On this system, `N_CORES` was set to **126**. This parameter should be adapted to the actual number of available logical CPU cores on the target machine.
- Default grid sizes are calibrated for a machine with **512 GB of RAM**. With this configuration (`GRID_POINTS` up to 3000), memory usage can reach **approximately 180 GB of RAM per run**. These parameters should therefore be adjusted according to the available memory on the target machine.
- Core experiment parameters are defined near the top of each `run_*.py` script.
