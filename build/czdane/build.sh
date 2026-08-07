#!/bin/bash

set -eE

PREFIX=$(dirname $(realpath $0))
SRCDIR=$(dirname $(dirname $PREFIX))
CURDIR=$(pwd)

GCC_ROOT=/usr/tce/packages/gcc/gcc-13.3.1
MPI_ROOT=/usr/tce/packages/openmpi/openmpi-4.1.2-gcc-13.3.1

OPENBLAS_ROOT=$PREFIX/openblas
EIGEN_ROOT=$PREFIX/eigen
HDF5_ROOT=$PREFIX/hdf5
BOBA_ROOT=$PREFIX/BoBa
TRILINOS_ROOT=$PREFIX/trilinos
CAMELLIA_ROOT=$PREFIX/camellia

on_error()
{
    local status=$?
    local line=$1

    echo
    echo "build/czdane/build.sh failed at line $line with exit code $status."
    echo "Last command: $BASH_COMMAND"
    local latest_log
    latest_log=$(find "$PREFIX" -maxdepth 2 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
    if [ -n "$latest_log" ]; then
        echo "Newest log: $latest_log"
        echo "Last 40 log lines:"
        tail -40 "$latest_log"
    else
        echo "No stage log was found under $PREFIX."
    fi
    exit $status
}

fail()
{
    echo "error: $*" >&2
    exit 1
}

require_dir()
{
    local path=$1
    local message=$2

    if [ ! -d "$path" ]; then
        fail "$message"
    fi
}

require_file()
{
    local path=$1
    local message=$2

    if [ ! -f "$path" ]; then
        fail "$message"
    fi
}

require_executable()
{
    local path=$1

    if [ ! -x "$path" ]; then
        fail "required executable not found: $path"
    fi
}

trap 'on_error $LINENO' ERR

require_executable "$GCC_ROOT/bin/gcc"
require_executable "$GCC_ROOT/bin/g++"
require_executable "$GCC_ROOT/bin/gfortran"
require_executable "$MPI_ROOT/bin/mpicc"
require_executable "$MPI_ROOT/bin/mpicxx"

require_dir "$OPENBLAS_ROOT/src" "OpenBLAS source is missing at $OPENBLAS_ROOT/src. Run ./build/czdane/download.sh first."
require_file "$OPENBLAS_ROOT/src/CMakeLists.txt" "OpenBLAS source at $OPENBLAS_ROOT/src is incomplete. Run ./build/czdane/download.sh first."
require_dir "$TRILINOS_ROOT/src" "Trilinos source is missing at $TRILINOS_ROOT/src. Run ./build/czdane/download.sh first."
require_file "$TRILINOS_ROOT/src/CMakeLists.txt" "Trilinos source at $TRILINOS_ROOT/src is incomplete. Run ./build/czdane/download.sh first."
require_dir "$SRCDIR/tpl/BoBa" "BoBa source is missing at $SRCDIR/tpl/BoBa. Run ./build/czdane/download.sh first."
require_file "$SRCDIR/tpl/BoBa/CMakeLists.txt" "BoBa source at $SRCDIR/tpl/BoBa is incomplete. Run ./build/czdane/download.sh first."
require_dir "$SRCDIR/tpl/BoBa/tpl/blt" "BoBa BLT source is missing at $SRCDIR/tpl/BoBa/tpl/blt. Run ./build/czdane/download.sh first."
require_dir "$SRCDIR/tpl/BoBa/tpl/eigen" "BoBa Eigen source is missing at $SRCDIR/tpl/BoBa/tpl/eigen. Run ./build/czdane/download.sh first."
require_dir "$SRCDIR/tpl/BoBa/tpl/hdf5" "BoBa HDF5 source is missing at $SRCDIR/tpl/BoBa/tpl/hdf5. Run ./build/czdane/download.sh first."

echo "-- building openblas"
mkdir -p $OPENBLAS_ROOT
cmake \
    -D BUILD_SHARED_LIBS=ON \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER=$GCC_ROOT/bin/gcc \
    -D CMAKE_Fortran_COMPILER=$GCC_ROOT/bin/gfortran \
    -D CMAKE_INSTALL_PREFIX=$OPENBLAS_ROOT \
    -D CMAKE_INSTALL_LIBDIR=$OPENBLAS_ROOT/lib \
    -B $OPENBLAS_ROOT/build \
    -S $OPENBLAS_ROOT/src > $OPENBLAS_ROOT/config.log 2>&1
cmake --build $OPENBLAS_ROOT/build --target all -j32 > $OPENBLAS_ROOT/build.log 2>&1
cmake --build $OPENBLAS_ROOT/build --target install > $OPENBLAS_ROOT/install.log 2>&1

echo "-- building hdf5"
mkdir -p $HDF5_ROOT
cmake \
    -D BUILD_SHARED_LIBS=OFF \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER=$MPI_ROOT/bin/mpicc \
    -D CMAKE_INSTALL_PREFIX=$HDF5_ROOT \
    -D CMAKE_INSTALL_LIBDIR=$HDF5_ROOT/lib \
    -D BUILD_TESTING=OFF \
    -D HDF5_BUILD_EXAMPLES=OFF \
    -D HDF5_BUILD_DOC=OFF \
    -D HDF5_ENABLE_Z_LIB_SUPPORT=OFF \
    -D HDF5_ENABLE_SZIP_SUPPORT=OFF \
    -D HDF5_ENABLE_PARALLEL=ON \
    -B $HDF5_ROOT/build \
    -S $SRCDIR/tpl/BoBa/tpl/hdf5 > $HDF5_ROOT/config.log 2>&1
cmake --build $HDF5_ROOT/build --target all -j32 > $HDF5_ROOT/build.log 2>&1
cmake --build $HDF5_ROOT/build --target install > $HDF5_ROOT/install.log 2>&1

echo "-- building trilinos"
mkdir -p $TRILINOS_ROOT
cmake \
    -D BUILD_SHARED_LIBS=OFF \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER=$MPI_ROOT/bin/mpicc \
    -D CMAKE_CXX_COMPILER=$MPI_ROOT/bin/mpicxx \
    -D CMAKE_INSTALL_PREFIX=$TRILINOS_ROOT \
    -D CMAKE_INSTALL_LIBDIR=$TRILINOS_ROOT/lib \
    -D TPL_ENABLE_BLAS=ON \
    -D BLAS_LIBRARY_DIRS=$OPENBLAS_ROOT/lib \
    -D BLAS_LIBRARY_NAMES=openblas \
    -D TPL_ENABLE_LAPACK=ON \
    -D LAPACK_LIBRARY_DIRS=$OPENBLAS_ROOT/lib \
    -D LAPACK_LIBRARY_NAMES=openblas \
    -D TPL_ENABLE_MPI=ON \
    -D MPI_BASE_DIR=$MPI_ROOT \
    -D TPL_ENABLE_HDF5=ON \
    -D HDF5_ROOT=$HDF5_ROOT \
    -D Trilinos_ENABLE_ALL_PACKAGES=OFF \
    -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF \
    -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION=ON \
    -D Trilinos_ENABLE_TESTS=OFF \
    -D Trilinos_ENABLE_EXAMPLES=OFF \
    -D Trilinos_ENABLE_Fortran=OFF \
    -D Trilinos_ENABLE_OpenMP=ON \
    -D Trilinos_ENABLE_Amesos=ON \
    -D Trilinos_ENABLE_Amesos2=ON \
    -D Trilinos_ENABLE_Belos=ON \
    -D Trilinos_ENABLE_Epetra=ON \
    -D Trilinos_ENABLE_EpetraExt=ON \
    -D EpetraExt_ENABLE_HDF5=ON \
    -D Trilinos_ENABLE_Ifpack=ON \
    -D Trilinos_ENABLE_Intrepid=ON \
    -D Trilinos_ENABLE_Intrepid2=ON \
    -D Intrepid2_ENABLE_DEBUG_INF_CHECK=OFF \
    -D Trilinos_ENABLE_Kokkos=ON \
    -D Trilinos_ENABLE_ML=ON \
    -D Trilinos_ENABLE_Shards=ON \
    -D Trilinos_ENABLE_Teuchos=ON \
    -D Trilinos_ENABLE_Tpetra=ON \
    -D Trilinos_ENABLE_Zoltan=ON \
    -D Trilinos_SHOW_DEPRECATED_WARNINGS=OFF \
    -B $TRILINOS_ROOT/build \
    -S $TRILINOS_ROOT/src > $TRILINOS_ROOT/config.log 2>&1
cmake --build $TRILINOS_ROOT/build --target all -j32 > $TRILINOS_ROOT/build.log 2>&1
cmake --build $TRILINOS_ROOT/build --target install > $TRILINOS_ROOT/install.log 2>&1

echo "-- building eigen"
mkdir -p $EIGEN_ROOT
cmake \
    -D BUILD_SHARED_LIBS=OFF \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER=$GCC_ROOT/bin/gcc \
    -D CMAKE_CXX_COMPILER=$GCC_ROOT/bin/g++ \
    -D CMAKE_INSTALL_PREFIX=$EIGEN_ROOT \
    -D CMAKE_INSTALL_LIBDIR=$EIGEN_ROOT/lib \
    -D CMAKE_CXX_STANDARD=20 \
    -D EIGEN_BUILD_TESTING=OFF \
    -D EIGEN_BUILD_DEMOS=OFF \
    -D EIGEN_BUILD_DOC=OFF \
    -D EIGEN_BUILD_BLAS=OFF \
    -D EIGEN_BUILD_LAPACK=OFF \
    -B $EIGEN_ROOT/build \
    -S $SRCDIR/tpl/BoBa/tpl/eigen > $EIGEN_ROOT/config.log 2>&1
cmake --build $EIGEN_ROOT/build --target all -j32 > $EIGEN_ROOT/build.log 2>&1
cmake --build $EIGEN_ROOT/build --target install > $EIGEN_ROOT/install.log 2>&1

echo "-- building boba"
mkdir -p $BOBA_ROOT
cmake \
    -D BUILD_SHARED_LIBS=OFF \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER=$GCC_ROOT/bin/gcc \
    -D CMAKE_CXX_COMPILER=$GCC_ROOT/bin/g++ \
    -D CMAKE_INSTALL_PREFIX=$BOBA_ROOT \
    -D CMAKE_INSTALL_LIBDIR=$BOBA_ROOT/lib \
    -D CMAKE_CXX_STANDARD=20 \
    -D BOBA_ENABLE_TESTS=OFF \
    -D BOBA_ENABLE_EXAMPLES=OFF \
    -D BOBA_ENABLE_TUTORIALS=OFF \
    -D BOBA_CHECKPOINTS=OFF \
    -D BOBA_DEBUG=OFF \
    -D ENABLE_OPENMP=OFF \
    -D BLT_SOURCE_DIR=$SRCDIR/tpl/BoBa/tpl/blt \
    -D BLT_CXX_STD=c++20 \
    -D BOBA_ENABLE_EIGEN=ON \
    -D EIGEN_DIR=$EIGEN_ROOT/share/eigen3/cmake \
    -D BOBA_ENABLE_HDF5=ON \
    -D HDF5_DIR=$HDF5_ROOT/cmake \
    -D BOBA_ENABLE_FMT=OFF \
    -D BOBA_ENABLE_CALIPER=OFF \
    -D BOBA_ENABLE_RAJA=OFF \
    -D BOBA_ENABLE_UMPIRE=OFF \
    -D BOBA_ENABLE_METAL=OFF \
    -D BOBA_ENABLE_CAMP=OFF \
    -D BOBA_ENABLE_MPI=OFF \
    -B $BOBA_ROOT/build \
    -S $SRCDIR/tpl/BoBa > $BOBA_ROOT/config.log 2>&1
cmake --build $BOBA_ROOT/build --target all -j32 > $BOBA_ROOT/build.log 2>&1
cmake --build $BOBA_ROOT/build --target install > $BOBA_ROOT/install.log 2>&1

echo "-- building camellia"
mkdir -p $CAMELLIA_ROOT
cmake \
    -D BUILD_SHARED_LIBS=OFF \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER=$MPI_ROOT/bin/mpicc \
    -D CMAKE_CXX_COMPILER=$MPI_ROOT/bin/mpicxx \
    -D CMAKE_INSTALL_PREFIX=$CAMELLIA_ROOT \
    -D CMAKE_INSTALL_LIBDIR=$CAMELLIA_ROOT/lib \
    -D CMAKE_CXX_EXTENSIONS=OFF \
    -D CMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -D TRILINOS_PATH=$TRILINOS_ROOT \
    -D BOBA_ROOT=$BOBA_ROOT \
    -D BOBA_DIR=$BOBA_ROOT/share/boba/cmake \
    -D BUILD_EXPERIMENTAL_DRIVERS=ON \
    -D BUILD_PRECONDITIONING_DRIVERS=ON \
    -B $CAMELLIA_ROOT/build \
    -S $SRCDIR > $CAMELLIA_ROOT/config.log 2>&1
cmake --build $CAMELLIA_ROOT/build --target all -j32 > $CAMELLIA_ROOT/build.log 2>&1
