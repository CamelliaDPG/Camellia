Camellia: A Software Toolbox for Discountinuous Petrov-Galerkin (DPG) Methods
-------------------------------------------------------------------
by Nathan V. Roberts

******** PREREQUISITES ********
Trilinos is required for all builds of Camellia.  A couple of sample do-configure scripts for Trilinos can be found in distribution directory, under build/trilinos-do-configure-samples.  These include the packages within Trilinos that Camellia requires.

Building Trilinos (specifically Epetra) with HDF5 is not absolutely required, but allows some useful visualization and other output (which can be read in ParaView, e.g.).

For an MPI build, Camellia also requires some version of the MPI libraries.  Open MPI is what we use most of the time.  Additionally, Camellia supports MUMPS and SuperLU_Dist if both Camellia and Trilinos are built with these libraries.  MUMPS also requires SCALAPACK to be installed.

Instructions for building several of these libraries follow.

CMake install:
On a Mac, our experience is that due to Apple’s requirements for code signatures it is simpler to install CMake from source than to use the prebuilt binary.

SWIG install (required to build Trilinos with support for PyTrilinos):
1. Download source from http://www.swig.org/download.html.
2. Configure:
	./configure --prefix=/Users/nroberts/local/swig-3.0.2
3. Make:
	make -j6
4. Install:
	make install

OpenMPI install:
1. Download source from http://www.open-mpi.org/software/ompi/.
2. cd into source dir.
3. Configure (editing the prefix line according to where you'd like it installed):
	./configure --prefix=$HOME/lib/openmpi-1.8.3 CC=cc CXX=c++ FC=gfortran
4. Build:
	make -j6
5. Install:
	make install
6. Add the bin folder to your PATH, e.g. by adding to your .bashrc:
	export PATH=${PATH}:${HOME}/lib/openmpi-1.8.3/bin

HDF5 install (parallel build, not suitable for serial builds of Trilinos):
1. Download source for hdf5-1.8.X from http://www.hdfgroup.org/HDF5/release/obtainsrc.html#conf.
2. Untar.
3. Configure:
   CC=mpicc ./configure --prefix=/Users/nroberts/local/hdf5
4. Make and install:
   make -j6
   make install
5. In the Trilinos do-configure, you'll want to include lines like the following:
   -D TPL_ENABLE_HDF5:STRING=ON \
   -D HDF5_LIBRARY_DIRS:FILEPATH=/Users/nroberts/local/hdf5/lib \
   -D HDF5_LIBRARY_NAMES:STRING="hdf5" \
   -D TPL_HDF5_INCLUDE_DIRS:FILEPATH=/Users/nroberts/local/hdf5/include \
   -D EpetraExt_USING_HDF5:BOOL=ON \

HDF5 install (serial build, not suitable for parallel builds of Trilinos):
1. Download source for hdf5-1.8.X from http://www.hdfgroup.org/HDF5/release/obtainsrc.html#conf.
2. Untar.
3. Configure:
   CC=clang ./configure --prefix=/Users/nroberts/local/hdf5-serial
4. Make and install:
   make -j6
   make install
5. In the Trilinos do-configure, you'll want to include lines like the following:
   -D TPL_ENABLE_HDF5:STRING=ON \
   -D HDF5_LIBRARY_DIRS:FILEPATH=/Users/nroberts/local/hdf5-serial/lib \
   -D HDF5_LIBRARY_NAMES:STRING="hdf5" \
   -D TPL_HDF5_INCLUDE_DIRS:FILEPATH=/Users/nroberts/local/hdf5-serial/include \
   -D EpetraExt_USING_HDF5:BOOL=ON \

Scalapack install:
1. Download source from http://www.netlib.org/scalapack/
2. cd into source dir.
3. Configure:
	ccmake .
	(specify ~/lib/openmpi-1.8.3 as the MPI_BASE_DIR, and ~ as the CMAKE_INSTALL_PREFIX; the other values should be autofilled on configure.)
4. Build
	cmake .
	make -j6
5. Install
	make install

MUMPS install:
1. Download source from http://graal.ens-lyon.fr/MUMPS/
2. Copy <MUMPS dir>/Make.inc/Makefile.INTEL.PAR to <MUMPS dir>/Makefile.inc
3. Edit the following lines in Makefile.inc (some of these are specific to building on a Mac):
	FC = gfortran
	FL = gfortran
	SCALAP  = <home dir>/lib/libscalapack.a
	INCPAR = -I<home dir>/lib/openmpi-1.8.3/include
	LIBPAR = $(SCALAP)  -L<home dir>/lib/openmpi-1.8.3/lib -lmpi -lmpi_f77
	LIBBLAS = -framework vecLib # BLAS and LAPACK libraries for Mac
	OPTF    = -O3 -Dintel_ -DALLOW_NON_INIT 
	OPTL    = -O3
	OPTC    = -O3
4. make -j6
5. Copy the built libraries from <MUMPS dir>/lib to $HOME/lib/mumps-4.10.0.
6. Copy the include directory to $HOME/lib/mumps-4.10.0/include.

******** BUILDING CAMELLIA **********

Once that's done, you're ready to start on the Camellia build.

Instructions for a serial debug build:
1. Clone from repo.
	git clone https://github.com/CamelliaDPG/Camellia.git
2. Go to the serial-debug build directory:
	cd build/serial-debug
3. Edit do-configure-serial-debug in the following manner:
       - set the TRILINOS_PATH to your serial-debug Trilinos installation
       - set ZLIB_LIB to the path to the zlib library (for HDF5 support)
       - set CMAKE_INSTALL_PREFIX:PATH to your preferred install location for Camellia
4. Run the do-configure script:
	./do-configure-cli-serial-debug
5. make
6. make test
7. make install

As of this writing (4/9/15), all tests in DPGTests and runTests should pass, with the exception of two of the runTests tests, one of which tests new space-time features still under development; the other appears to be a bug in mesh saving/loading disk in the case of minimum-rule meshes with vertex-conforming traces.  Note that make test won't give you granular information about which tests are failing, and it also won't run DPGTests for you; DPGTests is our old collection of tests--new tests are being added to either runTests or runSlowTests, both of which make test *does* run.  Note also that make test will only run tests in serial; for full testing of an MPI build, one will want to use mpirun on unit_tests/runTests, slow_tests/runSlowTests, and drivers/DPGTests/DPGTests.  We generally run tests on 1 or 4 MPI nodes.
