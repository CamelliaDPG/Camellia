cmake_minimum_required(VERSION 2.8)

project(Camellia)

# Set Trilinos Directory
set(TRILINOS_PATH /workspace/jchan/trilinos_builds/mpi_release/)
set(MUMPS_PATH /workspace/jchan/MUMPS_4.9.2/)
# Set Camellia Source Directory
set(CAMELLIA_SOURCE_DIR src)
set(DRIVER_DIR drivers)

find_package(Trilinos PATHS ${TRILINOS_PATH}/lib/cmake/Trilinos ${TRILINOS_PATH})

# If FIND_PACKAGE successfully found your Trilinos install, it will
# set the Boolean flag Trilinos_FOUND.  The following IF statement
# fails with a FATAL_ERROR if Trilinos was not found.  If it _was_
# found, it prints out the values of some Trilinos configuration
# details.  You may find them useful for building your application
# that uses Trilinos.
IF(Trilinos_FOUND)
   MESSAGE("\nFound Trilinos!  Here are the details: ")
   MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
   MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
   MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
   MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
   MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
   MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
   MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
   MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
   MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
   MESSAGE("   Trilinos_CXX_COMPILER = ${Trilinos_CXX_COMPILER}")
   MESSAGE("   Trilinos_C_COMPILER = ${Trilinos_C_COMPILER}")
   MESSAGE("   Trilinos_Fortran_COMPILER = ${Trilinos_Fortran_COMPILER}")
   MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
   MESSAGE("   Trilinos_C_COMPILER_FLAGS = ${Trilinos_C_COMPILER_FLAGS}")
   MESSAGE("   Trilinos_Fortran_COMPILER_FLAGS =
     ${Trilinos_Fortran_COMPILER_FLAGS}")
   MESSAGE("   Trilinos_LINKER = ${Trilinos_LINKER}")
   MESSAGE("   Trilinos_EXTRA_LD_FLAGS = ${Trilinos_EXTRA_LD_FLAGS}")
   MESSAGE("   Trilinos_AR = ${Trilinos_AR}")
   MESSAGE("End of Trilinos details\n")
ELSE()
  MESSAGE(FATAL_ERROR "Could not find Trilinos!")
ENDIF()

# If you haven't already set the C++ compiler, use the same compiler
# that was used to build Trilinos on your machine.  
IF(NOT CMAKE_CXX_COMPILER)
  MESSAGE("Setting CXX compiler.")
  SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
ENDIF()

# If you haven't already set the C compiler, use the same compiler
# that was used to build Trilinos on your machine.  
IF(NOT CMAKE_C_COMPILER)
  MESSAGE("Setting C compiler.")
  SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
ENDIF()

# If you haven't already set the Fortran compiler, use the same compiler
# that was used to build Trilinos on your machine.  
IF(NOT CMAKE_Fortran_COMPILER)
  MESSAGE("Setting Fortran compiler.")
  SET(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER})
ENDIF()

MESSAGE("CAMELLIA_SOURCE_DIR = ${CAMELLIA_SOURCE_DIR} ")
# Find all library source files
FILE(GLOB_RECURSE LIB_SOURCES "${CAMELLIA_SOURCE_DIR}/*.cpp")
set(HEADERS 
  ${CAMELLIA_SOURCE_DIR}/include
)

include_directories(
  ${Trilinos_TPL_INCLUDE_DIRS}
  ${Trilinos_INCLUDE_DIRS}
  ${CAMELLIA_SOURCE_DIR}/include
  ${MUMPS_PATH}/include
  ${MUMPS_PATH}/PORD/include
)

link_directories(
  ${Trilinos_TPL_LIBRARY}
  ${Trilinos_DIR}
  ${MUMPS_PATH}/lib
  ${MUMPS_PATH}/PORD/lib
)

add_library(Camellia STATIC ${LIB_SOURCES} ${HEADERS})
target_link_libraries(Camellia
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
)

#link_libraries(
#  ${Trilinos_LIBRARIES} 
#  ${Trilinos_TPL_LIBRARIES}
#  Camellia
#)

# Build Drivers
# Conservative Confusion
set(CONSERVATIVE_CONFUSION_SOURCES
  ${DRIVER_DIR}/ConservativeConfusion/ConfusionDriver.cpp
)
message("Driver sources = ${CONSERVATIVE_CONFUSION_SOURCES}")
add_executable(ConservativeConfusion ${CONSERVATIVE_CONFUSION_SOURCES})
target_link_libraries(ConservativeConfusion
  #  ${Trilinos_TPL_LIBRARIES}
  #  ${Trilinos_LIBRARIES} 
  Camellia
)
