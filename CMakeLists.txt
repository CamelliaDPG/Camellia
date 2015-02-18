cmake_minimum_required(VERSION 2.6)

# Set TARGET_SUPPORTS_SHARED_LIBS -- otherwise problems building shared libs on Macs (at least)
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

option(INCLUDE_DRIVERS_IN_ALL "Include drivers in make all (set to ON for IDE project generation)" OFF)

IF(INCLUDE_DRIVERS_IN_ALL)
  SET(EXCLUDE_DRIVERS_FROM_ALL "")
ELSE()
  SET(EXCLUDE_DRIVERS_FROM_ALL "EXCLUDE_FROM_ALL")
ENDIF(INCLUDE_DRIVERS_IN_ALL)

#IF(APPLE)
#   SET(CMAKE_SHARED_MODULE_CREATE_C_FLAGS
#       "${CMAKE_SHARED_MODULE_CREATE_C_FLAGS} -flat_namespace -undefined suppress")
#ENDIF(APPLE)

# Set Camellia directories
set(CAMELLIA_SOURCE_DIR src)
set(DRIVER_DIR drivers)

if (ZLIB_LIB)
  link_libraries(${ZLIB_LIB})
endif()

if (OMP_INCLUDE)
  include_directories(${OMP_INCLUDE})
endif()

if(MPI_INCLUDE)
  include_directories(${MPI_INCLUDE})
endif()
if(MPI_LINK_CC)
  link_libraries(${MPI_LINK_CC})
endif()
if (MPI_LINK_FORTRAN)
  link_libraries(${MPI_LINK_FORTRAN})
endif()

if(MPI_DIR)
  MESSAGE("MPI_DIR = ${MPI_DIR}")
  include_directories(${MPI_DIR}/include)
  link_directories(${MPI_DIR})
  link_libraries(mpi)
  link_libraries(mpi_mpifh)
  link_libraries(-L/usr/local/lib/gfortran/lib -lgfortran)
#  link_libraries(mpi_f77)
endif()

if(SCALAPACK_LIB)
  link_libraries(${SCALAPACK_LIB})
endif()

if(ADDITIONAL_LIBRARY_DIRS)
  MESSAGE("ADDITIONAL_LIBRARY_DIRS: ${ADDITIONAL_LIBRARY_DIRS}")
  MESSAGE("ADDITIONAL_LIBRARIES: ${ADDITIONAL_LIBRARIES}")
  link_directories(${ADDITIONAL_LIBRARY_DIRS})
  set(Trilinos_TPL_LIBRARIES ${Trilinos_TPL_LIBRARIES}
${ADDITIONAL_LIBRARIES})
#  link_libraries(${ADDITIONAL_LIBRARIES})
endif()

MESSAGE(" Looking for Trilinos in ${TRILINOS_PATH}/lib/cmake/Trilinos
  ${TRILINOS_PATH} $ENV{TRILINOS_DIR}")

# Find Trilinos
find_package(Trilinos PATHS ${TRILINOS_PATH}/lib/cmake/Trilinos
  ${TRILINOS_PATH} $ENV{TRILINOS_DIR})

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

# Testing something:
#SET(CMAKE_CXX_COMPILER "echo")

#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

IF(NOT CMAKE_CXX_FLAGS)
  # -fPIC allows shared libraries to be built against the Camellia library
  SET(CMAKE_CXX_FLAGS ${Trilinos_CXX_COMPILER_FLAGS})
ENDIF() 

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
#MESSAGE("CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

# If you haven't already set the C compiler, use the same compiler
# that was used to build Trilinos on your machine.  
IF(NOT CMAKE_C_COMPILER)
  MESSAGE("Setting C compiler.")
  SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
ENDIF()

IF (NOT CMAKE_C_FLAGS)
  SET(CMAKE_C_FLAGS ${Trilinos_C_COMPILER_FLAGS})
ENDIF()

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
#MESSAGE("CMAKE_C_FLAGS = ${CMAKE_C_FLAGS}")

# If you haven't already set the Fortran compiler, use the same compiler
# that was used to build Trilinos on your machine.  
IF(NOT CMAKE_Fortran_COMPILER)
  MESSAGE("Setting Fortran compiler.")
  SET(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER})
ENDIF()

IF(NOT BUILD_SHARED_LIBS)
  MESSAGE("Setting BUILD_SHARED_LIBS to ${Trilinos_BUILD_SHARED_LIBS}")
  SET(BUILD_SHARED_LIBS ${Trilinos_BUILD_SHARED_LIBS})
ENDIF()

project(Camellia)

# Find all library source files
FILE(GLOB_RECURSE LIB_SOURCES "${CAMELLIA_SOURCE_DIR}/*.cpp" "${CAMELLIA_SOURCE_DIR}/include/*.h")
set(HEADERS 
  ${CAMELLIA_SOURCE_DIR}/include
)

include_directories(
  ${Trilinos_TPL_INCLUDE_DIRS}
  ${Trilinos_INCLUDE_DIRS}
  ${CAMELLIA_SOURCE_DIR}/include
  TPL/choice
)

link_directories(
  ${Trilinos_LIBRARY_DIRS}
  ${Trilinos_TPL_LIBRARY_DIRS}
  #${MUMPS_PATH}/lib
)

if (MUMPS_DIR)
  include_directories(${MUMPS_DIR}/include)
  link_directories(${MUMPS_DIR})
endif()

link_libraries(
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  ${ADDITIONAL_LIBRARIES}
)

add_library(Camellia ${LIB_SOURCES} ${HEADERS})

link_libraries(
  Camellia
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  ${ADDITIONAL_LIBRARIES}
  # Xdmd
  # hdf5_cpp
  #  dmumps
  #  mumps_common
  #  pord
)
# target_link_libraries(Camellia Xdmf)

FILE(GLOB ALL_HEADER_FILES "${HEADERS}/*.h" "${HEADERS}/*.hpp")
install (TARGETS Camellia DESTINATION lib)
install (FILES ${ALL_HEADER_FILES} DESTINATION include/Camellia)

# Enable testing (see unit_tests directory)
include(CTest)
enable_testing()

# Build Drivers
add_subdirectory(drivers ${EXCLUDE_DRIVERS_FROM_ALL})
add_subdirectory(examples)
add_subdirectory(unit_tests)
add_subdirectory(slow_tests)
