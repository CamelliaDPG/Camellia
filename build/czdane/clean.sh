#!/bin/bash

PREFIX=$(dirname $(realpath $0))

OPENBLAS_ROOT=$PREFIX/openblas
EIGEN_ROOT=$PREFIX/eigen
HDF5_ROOT=$PREFIX/hdf5
BOBA_ROOT=$PREFIX/boba
TRILINOS_ROOT=$PREFIX/trilinos
CAMELLIA_ROOT=$PREFIX/camellia

rm -rf $OPENBLAS_ROOT/build
rm -rf $OPENBLAS_ROOT/include
rm -rf $OPENBLAS_ROOT/lib
rm -rf $OPENBLAS_ROOT/*.log

rm -rf $EIGEN_ROOT

rm -rf $HDF5_ROOT

rm -rf $BOBA_ROOT

rm -rf $TRILINOS_ROOT/build
rm -rf $TRILINOS_ROOT/bin
rm -rf $TRILINOS_ROOT/include
rm -rf $TRILINOS_ROOT/lib
rm -rf $TRILINOS_ROOT/*.log

rm -rf $CAMELLIA_ROOT
