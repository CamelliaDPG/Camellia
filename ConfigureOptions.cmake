# Configure file for Camellia

#set(TRILINOS_PATH /opt/apps/ossw/libraries/trilinos/trilinos-10.12.2/sl6/gcc-4.6/openmpi-1.6/mkl-gf-10.2.6.038)
#set(TRILINOS_PATH /opt/apps/ossw/libraries/trilinos/test)
set(TRILINOS_PATH /workspace/truman/trilinos)
#set(MUMPS_PATH /opt/apps/ossw/libraries/mumps)
#set(TRILINOS_PATH $ENV{TRILINOS_DIR})
#set(MUMPS_PATH /workspace/jchan/MUMPS_4.9.2/)
#set(SCALAPACK_PATH /workspace/jchan/lib/scalapack)

set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_BUILD_TYPE DEBUG)
set(BUILD_SHARED_LIBS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
