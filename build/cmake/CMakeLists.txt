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

SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "/usr/local/")

find_package(VTK)
if (VTK_FOUND)
  include(${VTK_USE_FILE})
  if(VTK_LIBRARIES)
    link_libraries(${VTK_LIBRARIES})
  else()
    link_libraries(vtkHybrid)
  endif()
endif()

if (ZLIB_LIB)
  set(Trilinos_TPL_LIBRARIES ${Trilinos_TPL_LIBRARIES} ${ZLIB_LIB})
#  link_libraries(${ZLIB_LIB})
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
#  link_libraries(-L/Users/nroberts/lib/openmpi-1.4.3/lib -lmpi -lopen-rte -lopen-pal -lutil)
  link_directories(${MPI_DIR})
  link_libraries(mpi)
  link_libraries(-L/usr/local/lib/gfortran/lib -lgfortran)
  link_libraries(mpi_f77)
#  link_libraries(${MPI_DIR}/lib/mpi)
#  link_libraries(${MPI_DIR}/lib/mpi_f77)
  if(SCALAPACK_LIB)
    link_libraries(${SCALAPACK_LIB})
  endif()
else()
  if(SCALAPACK_LIB)
    link_libraries(${SCALAPACK_LIB})
  endif()
endif()

if(ADDITIONAL_LIBRARY_DIRS) 
  MESSAGE("ADDITIONAL_LIBRARY_DIRS: ${ADDITIONAL_LIBRARY_DIRS}") 
  MESSAGE("ADDITIONAL_LIBRARIES: ${ADDITIONAL_LIBRARIES}")
  link_directories(${ADDITIONAL_LIBRARY_DIRS})
  set(Trilinos_TPL_LIBRARIES ${Trilinos_TPL_LIBRARIES}
${ADDITIONAL_LIBRARIES})
#  link_libraries(${ADDITIONAL_LIBRARIES})
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
  MESSAGE("Setting CXX compiler to ${Trilinos_CXX_COMPILER}.")
  SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
#ENDIF()

# If you haven't already set the C compiler, use the same compiler
# that was used to build Trilinos on your machine.  You do _not_ 
# need to edit this line.
#IF(NOT CMAKE_C_COMPILER)
  MESSAGE("Setting C compiler to ${Trilinos_C_COMPILER}.")
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

set( CONVECTING_CONE_DRIVER_SPACETIME
  "${DRIVERS_DIR}/Convection/SpaceTimePrototypeConvectingConeDriver.cpp"
)

set( CONVECTING_CONE_DRIVER
  "${DRIVERS_DIR}/Convection/ConvectingConeDriver.cpp"
)

set( SCRATCH_DRIVER
  "${DRIVERS_DIR}/ScratchDriver/ScratchDriver.cpp"
)

set( VESTA_CRASH_DRIVER 
  "${DRIVERS_DIR}/VestaCrashDriver/VestaCrashDriver.cpp"
)

set( NEW_BURGERS_DRIVER
  "${SCRATCHPAD_DRIVER_DIR}/NewBurgersDriver.cpp"
)

set( CONDITIONING_EXPERIMENT_DRIVER
  "${DRIVERS_DIR}/ConditioningExperiment/ConditioningExperimentDriver.cpp"
)

set( KRYLOV_SCHUR_CONDITIONING_EXPERIMENT_DRIVER
  "${DRIVERS_DIR}/ConditioningExperiment/BlockKrylovSchurEpetraExFile.cpp"
)

set( LOBPCG_CONDITIONING_EXPERIMENT_DRIVER
  "${DRIVERS_DIR}/ConditioningExperiment/LOBPCGEpetraExFile.cpp"
)

set( LSQR_CONDITIONING_EXPERIMENT_DRIVER
  "${DRIVERS_DIR}/ConditioningExperiment/LSQREpetraExFile.cpp"
)

set( SCALING_EXPERIMENT
  "${DRIVERS_DIR}/ScalingExperiment/ScalingExperiment.cpp"
)

set( SCALING_EXPERIMENT_ML
  "${DRIVERS_DIR}/ScalingExperiment/ScalingExperimentML.cpp"
)

set( CONFUSION_ITERATION_EXPERIMENT
  "${SCRATCHPAD_DRIVER_DIR}/ConfusionIterationExperiment.cpp"
)

set( NEW_CONFUSION_DRIVER
  "${SCRATCHPAD_DRIVER_DIR}/NewConfusionDriver.cpp"
)

set( STOKES_CONSERVATION_EXPERIMENT
  "${STOKES_DIR}/StokesConservationExperiment.cpp"
  )

set( STOKES_BACKWARD_FACING_STEP_DRIVER
  "${STOKES_DIR}/StokesBackwardFacingStepDriver.cpp"
  )

set( NAVIER_STOKES_BACKWARD_FACING_STEP_DRIVER
  "${INCOMPRESSIBLE_NS_DIR}/NavierStokesBackwardFacingStepDriver.cpp"
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

set( LAPLACE_3D_SINGLE_ELEMENT_DRIVER
  "${DEMO_DRIVERS_DIR}/Laplace3DSingleElement.cpp"
  )

set( NAVIER_STOKES_HEMKER_DRIVER
  "${INCOMPRESSIBLE_NS_DIR}/NavierStokesHemkerDriver.cpp"
  )
  
set( STOKES_CAVITY_FLOW_DRIVER
  "${STOKES_DIR}/StokesCavityFlowDriver.cpp"
  )
  
set( STOKES_CAVITY_FLOW_DRIVER_3D
  "${STOKES_DIR}/StokesCavityFlowDriver3D.cpp"
  )
  
set( STOKES_CAVITY_FLOW_DRIVER_FOR_PAPER
  "${STOKES_DIR}/StokesCavityFlowDriverForPaper.cpp"
  )
  
set( STOKES_STUDY
  "${STOKES_DIR}/StokesStudy.cpp"
  )

set( STOKES_STUDY_FOR_LESZEK
  "${STOKES_DIR}/StokesStudyForLeszek.cpp"
  )
  
set( INCOMP_NS_STUDY
  "${INCOMPRESSIBLE_NS_DIR}/NavierStokesStudy.cpp"
  )

set( OSEEN_STUDY
  "${INCOMPRESSIBLE_NS_DIR}/OseenStudy.cpp"
  )

set( TRUMAN_FLAT_PLATE_INS
  "${DRIVERS_DIR}/Truman/LocalConservation/FlatPlateINS.cpp"
)

#set( uBLAS_CHOLESKY_TEST
#  "${DRIVERS_DIR}/uBLAS/cholesky_test.cpp"
#)

set( STOKES_SOURCES
  "${STOKES_DIR}/StokesBilinearForm.cpp"
  "${STOKES_DIR}/StokesMathBilinearForm.cpp"
  "${STOKES_DIR}/StokesVVPBilinearForm.cpp"
  "${STOKES_DIR}/LidDrivenFlowRefinementStrategy.cpp"
  "${STOKES_DIR}/BackwardFacingStepRefinementStrategy.cpp"
)

set( POISSON_SOURCES
  "${DRIVERS_DIR}/Poisson/PoissonBilinearForm.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonExactSolution.cpp"
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
  link_directories(${MUMPS_DIR})
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
add_executable(ConditioningExperimentDriver ${CONDITIONING_EXPERIMENT_DRIVER})
add_executable(ConvectingConeDriver ${CONVECTING_CONE_DRIVER})
add_executable(SpaceTimePrototypeConvectingConeDriver ${CONVECTING_CONE_DRIVER_SPACETIME})
add_executable(KrylovSchurConditioningDriver ${KRYLOV_SCHUR_CONDITIONING_EXPERIMENT_DRIVER})
add_executable(LOBPCGConditioningDriver ${LOBPCG_CONDITIONING_EXPERIMENT_DRIVER})
add_executable(ScalingExperiment ${CONFUSION_SOURCES} ${SCALING_EXPERIMENT})
add_executable(ScalingExperimentML ${CONFUSION_SOURCES} ${SCALING_EXPERIMENT_ML})
add_executable(LSQRConditioningDriver ${LSQR_CONDITIONING_EXPERIMENT_DRIVER})
add_executable(LaplaceCurvilinear ${LAPLACE_CURVILINEAR_DRIVER})
add_executable(Laplace3DSingleElement ${LAPLACE_3D_SINGLE_ELEMENT_DRIVER})
add_executable(StokesStudy ${STOKES_SOURCES} ${MULTI_ORDER_STUDY_SOURCES} ${STOKES_STUDY})
add_executable(StokesStudyForLeszek ${STOKES_SOURCES} ${STOKES_STUDY_FOR_LESZEK})
add_executable(NavierStokesStudy ${INCOMP_NS_STUDY})
add_executable(OseenStudy ${OSEEN_STUDY})
add_executable(StokesBackwardFacingStepDriver ${STOKES_SOURCES} ${STOKES_BACKWARD_FACING_STEP_DRIVER})
add_executable(NavierStokesBackwardFacingStepDriver ${STOKES_SOURCES} ${NAVIER_STOKES_BACKWARD_FACING_STEP_DRIVER})
add_executable(StokesConservationExperiment ${STOKES_SOURCES} ${STOKES_CONSERVATION_EXPERIMENT})
add_executable(NavierStokesCavityFlowDriver ${STOKES_SOURCES} ${NAVIER_STOKES_CAVITY_FLOW_DRIVER})
add_executable(NavierStokesCavityFlowContinuationAdaptive ${NAVIER_STOKES_CAVITY_FLOW_CONTINUATION_ADAPTIVE})
add_executable(NavierStokesCavityFlowContinuationFixedMesh ${NAVIER_STOKES_CAVITY_FLOW_CONTINUATION_FIXED_MESH})
add_executable(NavierStokesHemkerDriver ${NAVIER_STOKES_HEMKER_DRIVER})
add_executable(StokesCavityFlowDriver ${STOKES_SOURCES} ${STOKES_CAVITY_FLOW_DRIVER})
add_executable(StokesCavityFlowDriver3D ${STOKES_CAVITY_FLOW_DRIVER_3D})
add_executable(StokesCavityFlowDriverForPaper ${STOKES_CAVITY_FLOW_DRIVER_FOR_PAPER})
add_executable(ConfusionIterationExperiment ${CONFUSION_ITERATION_EXPERIMENT})
add_executable(NewConfusionDriver ${NEW_CONFUSION_DRIVER})
add_executable(NewBurgersDriver ${NEW_BURGERS_DRIVER})
add_executable(ScratchDriver ${SCRATCH_DRIVER})
add_executable(VestaCrashDriver ${VESTA_CRASH_DRIVER})
add_executable(DPGTests ${TEST_SOURCES} ${POISSON_SOURCES} ${CONFUSION_SOURCES})
add_executable(FlatPlateINS ${TRUMAN_FLAT_PLATE_INS})
#add_executable(ublas_cholesky ${uBLAS_CHOLESKY_TEST})

#
# EDIT HERE: Change trilinosExample to the name of the executable
# file, as above.  This line tells the linker to link the executable 
# with the Trilinos libraries, and with the third-party libraries 
# (TPLs) with which Trilinos was built.
#
target_link_libraries(${CAMELLIA_LIBRARY_NAME}
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  )

target_link_libraries(ConditioningExperimentDriver
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(ConvectingConeDriver
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(SpaceTimePrototypeConvectingConeDriver
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(KrylovSchurConditioningDriver
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(LOBPCGConditioningDriver
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(LSQRConditioningDriver
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(ScalingExperiment
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(ScalingExperimentML
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(ScratchDriver
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(VestaCrashDriver
  )
target_link_libraries(NewBurgersDriver
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(ConfusionIterationExperiment
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
target_link_libraries(Laplace3DSingleElement
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
target_link_libraries(StokesCavityFlowDriver3D
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(StokesCavityFlowDriverForPaper
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(StokesBackwardFacingStepDriver
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(NavierStokesBackwardFacingStepDriver
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(StokesConservationExperiment
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(StokesStudy
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(StokesStudyForLeszek
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(NavierStokesStudy
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(FlatPlateINS
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )
target_link_libraries(OseenStudy
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  ${CAMELLIA_LIBRARY_NAME}
  )

add_subdirectory(${CAMELLIA_SOURCE_DIR}/../tests ${PROJECT_BINARY_DIR}/tests)
