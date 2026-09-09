# Build Instructions for El Capitan

This set of instructions aims to build Camellia on CZ Tuolumne using minimal
dependencies:

The following instructions assume you are inside the root Camellia directory. 
Otherwise, please adjust the paths to the scripts
accordingly.

Download the dependencies:
```sh
./build/elcap/hdf5-do-download
./build/elcap/trilinos-do-download
# TODO ./build/elcap/boba-do-download
```

Install the dependencies, then build camellia:
```sh
# hdf5
./build/elcap/hdf5-do-configure
./build/elcap/hdf5-do-build
./build/elcap/hdf5-do-install
# trilinos
./build/elcap/trilinos-do-configure
./build/elcap/trilinos-do-build
./build/elcap/trilinos-do-install
# boba
./build/elcap/boba-do-build
# camellia
./build/elcap/camellia-do-configure  # fails here
./build/elcap/camellia-do-build
```
