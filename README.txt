Camellia: A Software Toolbox for Discountinuous Petrov-Galerkin (DPG) Methods
-------------------------------------------------------------------
by Nathan V. Roberts

******** PREREQUISITES ********
Trilinos is required for all builds of Camellia.  A couple of sample do-configure scripts for Trilinos can be found in distribution directory, under build/trilinos-do-configure-samples.  These include the packages within Trilinos that Camellia requires.

For a serial debug build, Camellia requires Boost.  VTK is not absolutely required, but very helpful to allow fuller visualization output capabilities.

For an MPI build, Camellia also requires some version of the MPI libraries.  Open MPI is what we use most of the time.  Additionally, Camellia supports MUMPS if both Camellia and Trilinos are built with the MUMPS libraries.  MUMPS also requires SCALAPACK to be installed.

Instructions for building several of these libraries follow.

Boost install:
1. Download source from http://sourceforge.net/projects/boost/files/boost/.
2. cd into source dir.
3. Configure:
	./bootstrap.sh --prefix=/usr/local
4. Build:
	./b2
5. Install
	./b2 install

HDF5 install (parallel build, not suitable for serial builds of Trilinos):
1. Download source for hdf5-1.8.X from http://www.hdfgroup.org/HDF5/release/obtainsrc.html#conf.
2. Untar.
3. Configure:
   CC=mpicc ./configure --prefix=/Users/nroberts/lib/hdf5
4. Make and install:
   make -j6
   make install
5. In the Trilinos do-configure, you'll want to include lines like the following:
   -D TPL_ENABLE_HDF5:STRING=ON \
   -D HDF5_LIBRARY_DIRS:FILEPATH=/Users/nroberts/lib/hdf5/lib \
   -D HDF5_LIBRARY_NAMES:STRING="hdf5" \
   -D TPL_HDF5_INCLUDE_DIRS:FILEPATH=/Users/nroberts/lib/hdf5/include \
   -D EpetraExt_USING_HDF5:BOOL=ON \

HDF5 install (serial build, not suitable for parallel builds of Trilinos):
1. Download source for hdf5-1.8.X from http://www.hdfgroup.org/HDF5/release/obtainsrc.html#conf.
2. Untar.
3. Configure:
   CC=clang ./configure --prefix=/Users/nroberts/lib/hdf5-serial
4. Make and install:
   make -j6
   make install
5. In the Trilinos do-configure, you'll want to include lines like the following:
   -D TPL_ENABLE_HDF5:STRING=ON \
   -D HDF5_LIBRARY_DIRS:FILEPATH=/Users/nroberts/lib/hdf5-serial/lib \
   -D HDF5_LIBRARY_NAMES:STRING="hdf5" \
   -D TPL_HDF5_INCLUDE_DIRS:FILEPATH=/Users/nroberts/lib/hdf5-serial/include \
   -D EpetraExt_USING_HDF5:BOOL=ON \

VTK install:
1. Download source from http://www.vtk.org/files/release/5.10/vtk-5.10.1.tar.gz.
2. Untar:
	tar -xvf vtk-5.10.1.tar.gz
	cd VTK5.10.1
3. Run ccmake:
	ccmake .
4. Within ccmake, type 'c' to configure.
5. Edit any configuration options.  "CMAKE_INSTALL_PREFIX" is probably the most interesting/relevant.
6. Type 'c' again.
7. Once it's done configuring, type 'g' to generate and exit.
8. Make and install:
	make
	make install

OpenMPI install:
1. Download source from http://www.open-mpi.org/software/ompi/.
2. cd into source dir.
3. Configure (editing the prefix line according to where you'd like it installed):
	./configure --prefix=$HOME/lib/openmpi-1.6.5
4. Build:
	make -j6
5. Install:
	make install
6. Add the bin folder to your PATH, e.g. by adding to your .bashrc:
	export PATH=${PATH}:${HOME}/lib/openmpi-1.6.5/bin

Scalapack install:
1. Download source from http://www.netlib.org/scalapack/
2. cd into source dir.
3. Configure:
	ccmake .
	(specify ~/lib/openmpi-1.6.5 as the MPI_BASE_DIR, and ~ as the CMAKE_INSTALL_PREFIX; the other values should be autofilled on configure.)
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
	INCPAR = -I<home dir>/lib/openmpi-1.6.5/include
	LIBPAR = $(SCALAP)  -L<home dir>/lib/openmpi-1.6.5/lib -lmpi -lmpi_f77
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
	cd build/cmake/cli-debug
3. Edit do-configure-cli-serial-debug in the following manner:
       - set the TRILINOS_PATH to your serial-debug Trilinos installation
       - set the VTK_DIR to match wherever you installed VTK
       - set the ZLIB_LIB to the path to the zlib library (for HDF5 support)
4. Run the do-configure script:
	./do-configure-cli-serial-debug
5. Try building DPGTests:
	make DPGTests
6. Assuming it builds, try running it:
	./DPGTests

All tests in DPGTests should pass, with one exception: ScratchPadTests::testLTResidualSimple().  This has been failing for quite a while.  I leave it there until I understand it well enough to fix either the test or, if the test is good, the bug it reveals.