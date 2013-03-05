#
# CMake example that uses FIND_PACKAGE(Trilinos ...) to build your C++
# application with Trilinos.  You should know a little bit about CMake
# before reading this example; in particular, you should know how to
# add C++ source files and header files to your project.
#

# You need CMake version >= 2.8 to use this example.
cmake_minimum_required(VERSION 2.8)

# Your "do-configure" script that invokes CMake should set
# TRILINOS_PATH to the path to your Trilinos install.
# You do _not_ need to edit this line.
FIND_PACKAGE(Trilinos PATHS ${TRILINOS_PATH}/lib/cmake/Trilinos ${TRILINOS_PATH})

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
if(VTK_LIBRARIES)
  link_libraries(${VTK_LIBRARIES})
else()
  link_libraries(vtkHybrid)
endif()

if(MPI_DIR)
  MESSAGE("MPI_DIR = ${MPI_DIR}")
  include_directories(${MPI_DIR}/include)
#  link_libraries(-L/Users/nroberts/lib/openmpi-1.4.3/lib -lmpi -lopen-rte -lopen-pal -lutil)
  link_directories(${MPI_DIR})
  link_libraries(mpi)
  link_libraries(gfortran)
  link_libraries(mpi_f77)
#  link_libraries(${MPI_DIR}/lib/mpi)
#  link_libraries(${MPI_DIR}/lib/mpi_f77)
  if(SCALAPACK_LIB)
    link_libraries(${SCALAPACK_LIB})
  endif()
endif()

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
# that was used to build Trilinos on your machine.  You do _not_ need
# to edit this line.
#IF(NOT CMAKE_CXX_COMPILER)
  MESSAGE("Setting CXX compiler.")
  SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
#ENDIF()

# If you haven't already set the C compiler, use the same compiler
# that was used to build Trilinos on your machine.  You do _not_ 
# need to edit this line.
#IF(NOT CMAKE_C_COMPILER)
  MESSAGE("Setting C compiler.")
  SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
#ENDIF()

# If you haven't already set the Fortran compiler, use the same compiler
# that was used to build Trilinos on your machine.  You do _not_ 
# need to edit this line.
IF(NOT CMAKE_Fortran_COMPILER)
  MESSAGE("Setting Fortran compiler.")
  SET(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER})
ENDIF()

#
# EDIT HERE: Replace TrilinosExample with your project's name.
#
IF( ${CAMELLIA_BUILD_TYPE} MATCHES "debug" ) 
    MESSAGE("Building Camellia-debug project.")
    project(Camellia-debug)
    set(CAMELLIA_LIBRARY_NAME "Camellia-debug")
ELSE( )
    MESSAGE("Building Camellia project.")
    project(Camellia)
    set(CAMELLIA_LIBRARY_NAME "Camellia")
ENDIF( ${CAMELLIA_BUILD_TYPE} MATCHES "debug" )
# 
# EDIT HERE: 
#
# Source file(s) in your project.  Here, we include one source file, 
# main.cpp, which contains the main() function.  You may also want to 
# set HEADERS to the list of header files in your project, if there 
# are any.
#
MESSAGE("CAMELLIA_SOURCE_DIR = ${CAMELLIA_SOURCE_DIR} ")
FILE(GLOB_RECURSE SOURCES "${CAMELLIA_SOURCE_DIR}/*.cpp")

set(DRIVERS_DIR "${CAMELLIA_SOURCE_DIR}/../drivers" )
set(DEMO_DRIVERS_DIR "${DRIVERS_DIR}/DemoDrivers" )

set(NAVIER_STOKES_DIR "${DRIVERS_DIR}/NavierStokes" )

set(INCOMPRESSIBLE_NS_DIR "${DRIVERS_DIR}/IncompressibleNS" )

set(SCRATCHPAD_DRIVER_DIR "${DRIVERS_DIR}/ScratchPad" )

set(STOKES_DIR "${DRIVERS_DIR}/Stokes" )

set(TEST_DIR "${DRIVERS_DIR}/DPGTests" )
FILE(GLOB TEST_SOURCES "${TEST_DIR}/*.cpp")

set( NEW_BURGERS_DRIVER
  "${SCRATCHPAD_DRIVER_DIR}/NewBurgersDriver.cpp"
)

set( NEW_CONFUSION_DRIVER
  "${SCRATCHPAD_DRIVER_DIR}/NewConfusionDriver.cpp"
)

set( STOKES_BACKWARD_FACING_STEP_DRIVER
  "${STOKES_DIR}/StokesBackwardFacingStepDriver.cpp"
  )
  
set( NAVIER_STOKES_CAVITY_FLOW_DRIVER
  "${INCOMPRESSIBLE_NS_DIR}/NavierStokesCavityFlowDriver.cpp"
  )

set( NAVIER_STOKES_CAVITY_FLOW_CONTINUATION_FIXED_MESH
  "${INCOMPRESSIBLE_NS_DIR}/NavierStokesCavityFlowContinuationFixedMesh.cpp"
  )

set( NAVIER_STOKES_CAVITY_FLOW_CONTINUATION_ADAPTIVE
  "${INCOMPRESSIBLE_NS_DIR}/NavierStokesCavityFlowContinuationAdaptive.cpp"
  )

set( LAPLACE_CURVILINEAR_DRIVER
  "${DEMO_DRIVERS_DIR}/LaplaceCurvilinear.cpp"
  )

set( NAVIER_STOKES_HEMKER_DRIVER
  "${INCOMPRESSIBLE_NS_DIR}/NavierStokesHemkerDriver.cpp"
  )
  
set( STOKES_CAVITY_FLOW_DRIVER
  "${STOKES_DIR}/StokesCavityFlowDriver.cpp"
  )
  
set( STOKES_STUDY
  "${STOKES_DIR}/StokesStudy.cpp"
  )
  
set( INCOMP_NS_STUDY
  "${INCOMPRESSIBLE_NS_DIR}/NavierStokesStudy.cpp"
  )

set( STOKES_SOURCES
  "${STOKES_DIR}/StokesManufacturedSolution.cpp"
  "${STOKES_DIR}/StokesBilinearForm.cpp"
  "${STOKES_DIR}/StokesMathBilinearForm.cpp"
  "${STOKES_DIR}/StokesVVPBilinearForm.cpp"
  "${STOKES_DIR}/LidDrivenFlowRefinementStrategy.cpp"
)

set( POISSON_SOURCES
  "${DRIVERS_DIR}/Poisson/PoissonBilinearForm.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonExactSolution.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonExactSolutionLinear.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonExactSolutionCubic.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonExactSolutionQuadratic.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonExactSolutionQuartic.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonRHSLinear.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonRHSQuadratic.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonRHSCubic.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonRHSQuartic.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonBCLinear.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonBCQuadratic.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonBCCubic.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonBCQuartic.cpp"
)

set( CONFUSION_SOURCES
  "${DRIVERS_DIR}/Confusion/ConfusionManufacturedSolution.cpp"
  "${DRIVERS_DIR}/Confusion/ConfusionProblemLegacy.cpp"
  "${DRIVERS_DIR}/Confusion/ConfusionBilinearForm.cpp"
)

set(HEADERS 
  "${CAMELLIA_SOURCE_DIR}/include"
  )

set( MULTI_ORDER_STUDY_SOURCES "${DRIVERS_DIR}/MultiOrderStudy/MultiOrderStudy.cpp" )

# 
# EDIT HERE:
#
# Add to the header include path the current directory, Trilinos'
# include directory, and the include directories of the third-party
# libraries (TPLs) with which Trilinos was built.
#
include_directories("${CAMELLIA_SOURCE_DIR}/include")
include_directories("${DRIVERS_DIR}/Poisson") # for certain tests in DPGTests
include_directories("${DRIVERS_DIR}/Confusion") # for certain tests in DPGTests
include_directories("${DRIVERS_DIR}/Stokes") # for certain tests in DPGTests
include_directories(${Trilinos_TPL_INCLUDE_DIRS})
include_directories(${Trilinos_INCLUDE_DIRS})
if (MUMPS_DIR)
  include_directories(${MUMPS_DIR}/include)
endif()

# Add to the library path Trilinos' library path, and the library
# paths of the third-party libraries (TPLs) with which Trilinos was
# built.  
link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${Trilinos_TPL_LIBRARY_DIRS})

# 
# EDIT HERE: Change trilinosExample to the name of the executable
# file that you want to build.  This line tells CMake which source
# file(s) to use to build the executable.
#
MESSAGE( "Building project for ${CAMELLIA_LIBRARY_NAME} library." )
add_library(${CAMELLIA_LIBRARY_NAME} ${SOURCES} ${HEADERS})
add_executable(LaplaceCurvilinear ${LAPLACE_CURVILINEAR_DRIVER})
add_executable(StokesStudy ${STOKES_SOURCES} ${MULTI_ORDER_STUDY_SOURCES} ${STOKES_STUDY})
add_executable(NavierStokesStudy ${INCOMP_NS_STUDY})
add_executable(StokesBackwardFacingStepDriver ${STOKES_SOURCES} ${STOKES_BACKWARD_FACING_STEP_DRIVER})
add_executable(NavierStokesCavityFlowDriver ${NAVIER_STOKES_CAVITY_FLOW_DRIVER})
add_executable(NavierStokesCavityFlowContinuationAdaptive ${NAVIER_STOKES_CAVITY_FLOW_CONTINUATION_ADAPTIVE})
add_executable(NavierStokesCavityFlowContinuationFixedMesh ${NAVIER_STOKES_CAVITY_FLOW_CONTINUATION_FIXED_MESH})
add_executable(NavierStokesHemkerDriver ${NAVIER_STOKES_HEMKER_DRIVER})
add_executable(StokesCavityFlowDriver ${STOKES_SOURCES} ${STOKES_CAVITY_FLOW_DRIVER})
add_executable(NewConfusionDriver ${NEW_CONFUSION_DRIVER})
add_executable(NewBurgersDriver ${NEW_BURGERS_DRIVER})
add_executable(DPGTests ${TEST_SOURCES} ${POISSON_SOURCES} ${CONFUSION_SOURCES})

#
# EDIT HERE: Change trilinosExample to the name of the executable
# file, as above.  This line tells the linker to link the executable 
# with the Trilinos libraries, and with the third-party libraries 
# (TPLs) with which Trilinos was built.
#

target_link_libraries(NewBurgersDriver
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(NewConfusionDriver
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(DPGTests
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(NavierStokesCavityFlowDriver
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(NavierStokesCavityFlowContinuationAdaptive
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(NavierStokesCavityFlowContinuationFixedMesh
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(LaplaceCurvilinear
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(NavierStokesHemkerDriver
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(StokesCavityFlowDriver
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(StokesBackwardFacingStepDriver
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(StokesStudy
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(NavierStokesStudy
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )

add_subdirectory(${CAMELLIA_SOURCE_DIR}/../tests ${PROJECT_BINARY_DIR}/tests)
