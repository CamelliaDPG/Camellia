#!/bin/bash

set -eE

TRILINOS_COMMIT=c4e1b93401939184e18b2f930312a8a0e8d482d6

PREFIX=$(dirname $(realpath $0))
SRCDIR=$(dirname $(dirname $PREFIX))

CURDIR=$(pwd)

on_error()
{
    local status=$?
    local line=$1

    echo
    echo "build/czdane/download.sh failed at line $line with exit code $status."
    echo "Last command: $BASH_COMMAND"
    exit $status
}

download_if_missing()
{
    local output=$1
    local url=$2

    if [ ! -f "$output" ]; then
        curl -fLo "$output" "$url"
    fi
}

extract_to_src_if_missing()
{
    local archive=$1
    local extracted_dir=$2

    if [ -d src ]; then
        return
    fi

    if [ ! -d "$extracted_dir" ]; then
        tar -xf "$archive"
    fi

    mv "$extracted_dir" src
}

trap 'on_error $LINENO' ERR

mkdir -p $PREFIX/openblas
cd $PREFIX/openblas
download_if_missing openblas.tar.gz https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.28.tar.gz
extract_to_src_if_missing openblas.tar.gz OpenBLAS-0.3.28

cd $SRCDIR
git submodule update --init --depth=1 tpl/BoBa

cd $SRCDIR/tpl/BoBa
git submodule update --init --depth=1 tpl/blt
git submodule update --init --depth=1 tpl/eigen
git submodule update --init --depth=1 tpl/hdf5

mkdir -p $PREFIX/trilinos
cd $PREFIX/trilinos
download_if_missing trilinos.tar.gz https://github.com/trilinos/Trilinos/archive/$TRILINOS_COMMIT.tar.gz
extract_to_src_if_missing trilinos.tar.gz Trilinos-$TRILINOS_COMMIT

cd $CURDIR