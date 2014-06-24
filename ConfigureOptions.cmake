# Configure file for Camellia

set(TRILINOS_PATH /workspace/truman/trilinos)
# set(XDMF_PATH /workspace/truman/xdmf)
set(HDF5_PATH /workspace/truman/hdf5)
#set(MUMPS_PATH /opt/apps/ossw/libraries/mumps)
#set(TRILINOS_PATH $ENV{TRILINOS_DIR})
#set(MUMPS_PATH /workspace/jchan/MUMPS_4.9.2/)
#set(SCALAPACK_PATH /workspace/jchan/lib/scalapack)

set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_BUILD_TYPE DEBUG)
set(BUILD_SHARED_LIBS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
