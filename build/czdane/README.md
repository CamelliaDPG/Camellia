# Build Instructions for czdane

This set of instructions aims to build Camellia on CZ Dane using minimal
dependencies:

The following instructions assume you are inside the root Camellia directory.
Otherwise, please adjust the paths to the scripts accordingly.

To compile, launch an interactive job on the debug partition:
```sh
salloc --account=<account> --partition=pdebug --nodes=1 --ntasks=14 --time=01:00:00
```

Start with a clean environment:
```sh
module --force purge
module load StdEnv
module unload intel-classic/2021.6.0-magic mvapich2/2.3.7 jobutils/1.0 texlive/20220321
```
The last commands basically removes everything other than `StdEnv` itself.

Load the necessary modules:
```sh
module load intel/2023.2.1
module load mkl/2022.1.0
module load mvapich2/2.3.7
```

Download the dependencies:
```sh
./build/czdane/hdf5-do-download
./build/czdane/trilinos-do-download
```

Build and install the dependencies, then build camellia
```sh
# hdf5
./build/czdane/hdf5-do-configure
./build/czdane/hdf5-do-build
./build/czdane/hdf5-do-install
# trilinos
./build/czdane/trilinos-do-configure
./build/czdane/trilinos-do-build
./build/czdane/trilinos-do-install
# camellia
./build/czdane/camellia-do-configure
./build/czdane/camellia-do-build
```
