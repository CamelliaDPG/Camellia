# Build Instructions for czdane

This set of instructions aims to build Camellia on CZ Dane using minimal
dependencies.

The following assume you are inside the root Camellia directory. Otherwise,
please adjust the paths to the scripts accordingly.

To compile, launch an interactive job on the debug partition:
```sh
salloc --account=<acct> --partition=pdebug --nodes=1 --ntasks=32 --time=01:00:00
```

Load the system modules (appropriate compiler and MPI):
```sh
source ./build/czdane/modules.sh
```

Download the dependencies:
```sh
./build/czdane/download.sh
```

Finally, build and install the dependencies and build Camellia itself:
```sh
./build/czdane/build.sh
```

To start afresh, remove all build and install files (but not the downloaded source files) with:
```sh
./build/czdane/clean.sh
```
and re-run the build script.
