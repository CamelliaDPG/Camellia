# Build Instructions for MacOS

This set of instructions aims to build Camellia on MacOS using minimal
dependencies:

-   XCode command line tools: Install by running
    ```sh
    xcode-select --install
    ```

-   CMake: Install from the [official website](https://cmake.org) or via
    [Homebrew](https://brew.sh).

The following instructions assume you are inside the root directory of the
cloned repository. Otherwise, please adjust the paths to the scripts
accordingly.

First, build and install the dependencies:

```sh
./build/macos/openmpi-do-download
./build/macos/openmpi-do-configure
./build/macos/openmpi-do-build
./build/macos/openmpi-do-install

./build/macos/hdf5-do-download
./build/macos/hdf5-do-configure
./build/macos/hdf5-do-build
./build/macos/hdf5-do-install

./build/macos/trilinos-do-download
./build/macos/trilinos-do-configure
./build/macos/trilinos-do-build
./build/macos/trilinos-do-install
```

Next, build and install boba:

```sh
./build/macos/boba-do-build
```

Finally, build Camellia with BOBA:

```sh
./build/macos/camellia-do-configure
./build/macos/camellia-do-build
```

To test the library was created correctly, build the space-time
mixed-formulation Poisson driver with BOBA support: go to the
`./build/macos/camellia-build/drivers/pulse/SpaceTimeMixedPoissonBoba`
directory, and run `make`. The compilation will emit a lot of warnings, but
should finish without errors.
