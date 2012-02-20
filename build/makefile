#
# This makefile compiles a program that has dependencies
# on Trilinos packages.
#

# 
# Trilinos directory

ifndef BUILD_TYPE
BUILD_TYPE = serial-debug
endif

TRILINOS_DIR = /Users/nroberts/Documents/Sandia/trilinos-10.8.3-Source/build/$(BUILD_TYPE)
#TRILINOS_DIR = /workspace/jchan/trilinos-10.8.3-Source/trilinos_builds/$(BUILD_TYPE)
SRCS_DIR = ../../src
DRIVERS_DIR = ../../Drivers
MPI_LIBRARIES = -L/Users/nroberts/lib/openmpi-1.4.3/lib -lmpi -lmpi_f77 -lgfortran /Users/nroberts/lib/scalapack/libscalapack.a

# include makefile export files for Trilinos
include $(TRILINOS_DIR)/Makefile.export.Trilinos_install

# Set compiler to Trilinos compilers
CXX = $(Trilinos_CXX_COMPILER)
CPP = $(Trilinos_C_COMPILER)

# Set flags
CPPFLAGS = $(Trilinos_C_COMPILER_FLAGS) 
CXXFLAGS = $(Trilinos_CXX_COMPILER_FLAGS) 

#OBJ_DIR = $(BUILD_TYPE)
OBJ_DIR = .

# List sources for compilation
OBJ_NAMES = DofOrdering \
       BilinearForm \
       PoissonBilinearForm \
       BilinearFormUtility \
       BasisFactory \
       BasisSumFunction \
       BurgersBilinearForm \
       ConfectionManufacturedSolution \
       ConfusionBilinearForm \
       ConfusionManufacturedSolution \
       ConfusionProblem \
       MathInnerProduct \
       Element \
       ElementTests \
       ElementTypeFactory \
       Mesh \
       MeshPartitionPolicy \
       Boundary \
       PoissonBCLinear \
       PoissonRHSLinear \
       Projector \
       MeshTestSuite \
       MeshRefinementTests \
       SolutionTests \
       VectorizedBasisTestSuite \
       MultiBasis \
       MultiBasisTests \
       PatchBasis \
       PatchBasisTests \
       Solution \
       DPGInnerProduct \
       ExactSolution \
       MultiOrderStudy \
       PenaltyMethodFilter \
       PoissonExactSolutionLinear \
       PoissonBCQuartic \
       PoissonRHSQuartic \
       PoissonExactSolutionQuartic \
       PoissonBCQuadratic \
       PoissonRHSQuadratic \
       PoissonExactSolutionQuadratic \
       PoissonBCCubic \
       PoissonRHSCubic \
       PoissonExactSolutionCubic \
       PoissonExactSolution \
       DofOrderingFactory \
       BasisCache \
       BasisEvaluation \
       RefinementPattern \
			 StokesBilinearForm \
			 StokesVVPBilinearForm \
			 StokesManufacturedSolution \
			 TransientConfusionBilinearForm \
			 HConvergenceStudy \
			 OptimalInnerProduct \
			 ZoltanMeshPartitionPolicy

OBJS = $(addsuffix .o, $(OBJ_NAMES))
# put all the object files in OBJ_DIR:
#OBJS_IN_PLACE = $(OBJ_DIR)/$(OBJS,W $(OBJ_DIR)\/)
OBJS_IN_PLACE = $(addprefix $(OBJ_DIR)/, $(OBJS))

INCLUDE_PATHS = -I$(SRCS_DIR)/include -I$(DRIVERS_DIR)/DPGTests -I$(DRIVERS_DIR)/DemoDrivers -I$(DRIVERS_DIR)/Stokes -I$(DRIVERS_DIR)/Confusion -I$(DRIVERS_DIR)/TransientConfusion -I$(DRIVERS_DIR)/Poisson -I$(DRIVERS_DIR)/MultiOrderStudy -I/Users/nroberts/src/MUMPS_4.9.2/include $(Trilinos_INCLUDE_DIRS)

EXES = DPGTests StokesStudy StokesStudyHybridMesh PoissonSolver ConfusionStudy AdaptiveConfusion AdaptiveTransientConfusion ScalingExperiment ConfectionDriver BurgersDriver
ECHOES = 

default: all

all: $(ECHOES) $(EXES)

MovedObjs:
	@echo OBJS_IN_PLACE: $(OBJS_IN_PLACE)

DPGTests: DPGTests.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/DPGTests.o -o DPGTests $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

DPGTests.o: $(DRIVERS_DIR)/DPGTests/DPGTests.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

BurgersDriver: BurgersDriver.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/BurgersDriver.o -o BurgersDriver $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

BurgersDriver.o: $(DRIVERS_DIR)/Burgers/BurgersDriver.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

ConfectionDriver: ConfectionDriver.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/ConfectionDriver.o -o ConfectionDriver $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

ConfectionDriver.o: $(DRIVERS_DIR)/Confusion/ConfectionDriver.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

ConfectionManufacturedSolution.o: $(DRIVERS_DIR)/Confusion/ConfectionManufacturedSolution.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

StokesStudy: StokesStudy.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/StokesStudy.o -o StokesStudy $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

StokesStudyHybridMesh: StokesStudyHybridMesh.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/StokesStudyHybridMesh.o -o StokesStudyHybridMesh $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

ConfusionStudy: ConfusionStudy.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/ConfusionStudy.o -o ConfusionStudy $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

ScalingExperiment: ScalingExperiment.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/ScalingExperiment.o -o ScalingExperiment $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -lmpi_f77 -o $(OBJ_DIR)/$@

AdaptiveConfusion: AdaptiveConfusion.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/AdaptiveConfusion.o -o AdaptiveConfusion $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

AdaptiveTransientConfusion: AdaptiveTransientConfusion.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/AdaptiveTransientConfusion.o -o AdaptiveTransientConfusion $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

exampleMesh: exampleMesh.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/exampleMesh.o -o exampleMesh $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

PoissonSolver: PoissonSolver.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/PoissonSolver.o -o PoissonSolver $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

PoissonStudy: PoissonStudy.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/PoissonStudy.o -o PoissonStudy $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

MultiOrderStudy.o: $(DRIVERS_DIR)/MultiOrderStudy/MultiOrderStudy.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

MeshTestSuite.o: $(DRIVERS_DIR)/DPGTests/MeshTestSuite.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

MeshRefinementTests.o: $(DRIVERS_DIR)/DPGTests/MeshRefinementTests.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

SolutionTests.o: $(DRIVERS_DIR)/DPGTests/SolutionTests.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

ElementTests.o: $(DRIVERS_DIR)/DPGTests/ElementTests.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

PatchBasisTests.o: $(DRIVERS_DIR)/DPGTests/PatchBasisTests.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

MultiBasisTests.o: $(DRIVERS_DIR)/DPGTests/MultiBasisTests.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

VectorizedBasisTestSuite.o: $(DRIVERS_DIR)/DPGTests/VectorizedBasisTestSuite.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

BurgersBilinearForm.o: $(DRIVERS_DIR)/Burgers/BurgersBilinearForm.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

ConfusionBilinearForm.o: $(DRIVERS_DIR)/Confusion/ConfusionBilinearForm.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

PenaltyMethodFilter.o: $(SRCS_DIR)/Solution/PenaltyMethodFilter.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

TransientConfusionBilinearForm.o: $(DRIVERS_DIR)/TransientConfusion/TransientConfusionBilinearForm.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

ConfusionManufacturedSolution.o: $(DRIVERS_DIR)/Confusion/ConfusionManufacturedSolution.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@

ConfusionProblem.o: $(DRIVERS_DIR)/Confusion/ConfusionProblem.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@

ConfusionStudy.o: $(DRIVERS_DIR)/Confusion/ConfusionStudy.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

ScalingExperiment.o: $(DRIVERS_DIR)/ScalingExperiment/ScalingExperiment.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

AdaptiveConfusion.o: $(DRIVERS_DIR)/Confusion/AdaptiveConfusion.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

AdaptiveTransientConfusion.o: $(DRIVERS_DIR)/TransientConfusion/AdaptiveTransientConfusion.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

exampleMesh.o: $(DRIVERS_DIR)/RefTreeExperiment/exampleMesh.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

PoissonSolver.o: $(DRIVERS_DIR)/DemoDrivers/PoissonSolver.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

Poisson%.o: $(DRIVERS_DIR)/Poisson/Poisson%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

Stokes%.o: $(DRIVERS_DIR)/Stokes/Stokes%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

Basis%.o: $(SRCS_DIR)/Basis/Basis%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

%Basis.o: $(SRCS_DIR)/Basis/%Basis.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

%InnerProduct.o: $(SRCS_DIR)/InnerProduct/%InnerProduct.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

Bilinear%.o: $(SRCS_DIR)/Problem/Bilinear%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

%olution.o: $(SRCS_DIR)/Solution/%olution.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

Elem%.o: $(SRCS_DIR)/Mesh/Elem%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

Mesh.o: $(SRCS_DIR)/Mesh/Mesh.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

Projector.o: $(SRCS_DIR)/Projection/Projector.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

MeshPartitionPolicy.o: $(SRCS_DIR)/Mesh/MeshPartitionPolicy.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

ZoltanMeshPartitionPolicy.o: $(SRCS_DIR)/Mesh/ZoltanMeshPartitionPolicy.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

RefinementPattern.o: $(SRCS_DIR)/Mesh/RefinementPattern.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

Boundary.o: $(SRCS_DIR)/Mesh/Boundary.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

DofOrder%.o: $(SRCS_DIR)/DofOrdering/DofOrder%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

%ConvergenceStudy.o: $(SRCS_DIR)/ConvergenceStudy/%ConvergenceStudy.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

%.o: $(SRCS_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

clean:
	$(RM) $(OBJS_IN_PLACE) $(OBJ_DIR)/DPGTests.o $(OBJ_DIR)/PoissonStudy.o $(OBJ_DIR)/StokesStudy.o $(OBJ_DIR)/StokesStudyHybridMesh.o $(EXES)
