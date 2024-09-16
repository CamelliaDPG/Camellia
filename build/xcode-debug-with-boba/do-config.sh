TRILINOS_DIR=$HOME/trilinos-builds/fork-debug-with-fortran
PULSE_DIR=$HOME/pulse
CAMELLIA_SOURCE_DIR=$PULSE_DIR/src
BOBA_ROOT=$PULSE_DIR/tpl/boba/install/BOBA_cpu/share/boba/cmake

export MPI_DIR="/Users/nvrober/install/open-mpi-4.1.5"

cmake \
  -G "Xcode" \
  -D TRILINOS_PATH:FILEPATH="$TRILINOS_DIR" \
  -D BOBA_ROOT:FILEPATH="$BOBA_ROOT" \
  -D CAMELLIA_SOURCE_DIR:FILEPATH="$CAMELLIA_SOURCE_DIR" \
  -D CMAKE_EXE_LINKER_FLAGS="-L${MPI_DIR}/lib -lmpi" \
  -D CMAKE_SHARED_LINKER_FLAGS="-L${MPI_DIR}/lib -lmpi" \
  -D CMAKE_CXX_FLAGS_DEBUG="-g -O0 -I${MPI_DIR}/include" \
  -DBUILD_CONVECTION_DRIVERS:BOOL=ON \
  -DBUILD_EXPERIMENTAL_DRIVERS:BOOL=ON \
  -DBUILD_PRECONDITIONING_DRIVERS:BOOL=ON \
  -DBUILD_BRENDAN_DRIVERS:BOOL=ON \
  -DCMAKE_INSTALL_PREFIX:PATH="/Users/nroberts/lib/Camellia" \
  ../..

#  -D mlc_ROOT="/usr/local/lib" \
