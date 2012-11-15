#
# This makefile compiles a program that has dependencies
# on Trilinos packages.
#

# 
# Trilinos directory

ifndef BUILD_TYPE
BUILD_TYPE = serial-debug
endif

TRILINOS_DIR = /Users/nroberts/Documents/Sandia/trilinos-10.10.2-Source/build/$(BUILD_TYPE)
#TRILINOS_DIR = /workspace/jchan/trilinos-10.8.3-Source/trilinos_builds/$(BUILD_TYPE)
SRCS_DIR = ../../src
DRIVERS_DIR = ../../Drivers
MPI_LIBRARIES = -L/Users/nroberts/lib/openmpi-1.4.3-built-with-llvm/lib -lmpi -lmpi_f77 -lgfortran /Users/nroberts/lib/scalapack/libscalapack.a

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
       BC \
       BCFunction \
       ConfectionManufacturedSolution \
       ConfusionBilinearForm \
       ConfusionManufacturedSolution \
       ConfusionProblemLegacy \
       Constraint \
       MathInnerProduct \
       Element \
       ElementTypeFactory \
       EricksonManufacturedSolution \
       EricksonConfectionSolution \
       ExperimentalInnerProduct \
       LagrangeConstraints \
       Mesh \
       MeshPartitionPolicy \
       MeshPolyOrderFunction \
       MeshTestUtility \
       Boundary \
       PoissonBCLinear \
       PoissonRHSLinear \
       Projector \
       MultiBasis \
       PatchBasis \
       RieszRep \
       Solution \
       WriteSolution \
       DPGInnerProduct \
       ExactSolution \
       MultiOrderStudy \
       NonlinearSolveStrategy \
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
       RefinementStrategy \
       RHS \
       SchwarzSolver \
       CGSolver \
			 StokesBilinearForm \
			 StokesMathBilinearForm \
			 StokesVVPBilinearForm \
			 StokesManufacturedSolution \
			 LidDrivenFlowRefinementStrategy \
       HessianFilter \
       TimeMarchingProblem \
			 HConvergenceStudy \
			 OptimalInnerProduct \
			 BF \
			 LinearTerm \
			 Var \
			 BCEasy \
			 SpatiallyFilteredFunction \
			 Function \
			 RHSEasy \
			 IP \
			 ZoltanMeshPartitionPolicy

OBJS = $(addsuffix .o, $(OBJ_NAMES))
# put all the object files in OBJ_DIR:
OBJS_IN_PLACE = $(addprefix $(OBJ_DIR)/, $(OBJS))

TEST_OBJ_NAMES = ElementTests \
       MeshTestSuite \
       FunctionTests \
       IncompressibleFormulationsTests \
       LinearTermTests \
       MeshRefinementTests \
       ScratchPadTests \
       SolutionTests \
       VectorizedBasisTestSuite \
       MultiBasisTests \
       PatchBasisTests \
       TestBilinearFormDx \
       TestBilinearFormFlux \
       RHSTests \
       BasisCacheTests
TEST_OBJS = $(addsuffix .o, $(TEST_OBJ_NAMES))
# put all the object files in OBJ_DIR:
TEST_OBJS_IN_PLACE = $(addprefix $(OBJ_DIR)/, $(TEST_OBJS))

BURGERS_OBJ_NAMES = BurgersBilinearForm \
       BurgersInnerProduct \
       BurgersProblem \
       TimeMarchingBurgersProblem
BURGERS_OBJS = $(addsuffix .o, $(BURGERS_OBJ_NAMES))
# put all the object files in OBJ_DIR:
BURGERS_OBJS_IN_PLACE = $(addprefix $(OBJ_DIR)/, $(BURGERS_OBJS))

INCLUDE_PATHS = -I$(SRCS_DIR)/include -I$(DRIVERS_DIR)/DPGTests -I$(DRIVERS_DIR)/DemoDrivers -I$(DRIVERS_DIR)/Stokes -I$(DRIVERS_DIR)/Confusion -I$(DRIVERS_DIR)/Poisson -I$(DRIVERS_DIR)/MultiOrderStudy -I/Users/nroberts/src/MUMPS_4.9.2/include $(Trilinos_INCLUDE_DIRS)

EXES = DPGTests StokesBackwardFacingStepDriver StokesCavityFlowDriver NavierStokesCavityFlowDriver StokesStudy NavierStokesStudy StokesMathStudy StokesStudyHybridMesh PoissonSolver ConfusionStudy AdaptiveConfusion ScalingExperiment ConfectionDriver BurgersDriver TimeMarchingBurgersDriver EricksonDriver ScratchPadDriver NewConfusionDriver NewBurgersDriver 
EXE_OBJS = $(addsuffix .o, $(EXES))

ECHOES = 

default: all

all: $(ECHOES) $(EXES)

MovedObjs:
	@echo OBJS_IN_PLACE: $(OBJS_IN_PLACE)
  @echo TEST_OBJS_IN_PLACE: $(TEST_OBJS_IN_PLACE)
  @echo BURGERS_OBJS_IN_PLACE: $(BURGERS_OBJS_IN_PLACE)

DPGTests: DPGTests.o $(OBJS) $(TEST_OBJS) $(BURGERS_OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(TEST_OBJS_IN_PLACE) $(BURGERS_OBJS_IN_PLACE) $(OBJ_DIR)/DPGTests.o -o DPGTests $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

DPGTests.o: $(DRIVERS_DIR)/DPGTests/DPGTests.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

BurgersDriver: BurgersDriver.o $(OBJS) $(BURGERS_OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(BURGERS_OBJS_IN_PLACE) $(OBJ_DIR)/BurgersDriver.o -o BurgersDriver $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

BurgersDriver.o: $(DRIVERS_DIR)/Burgers/BurgersDriver.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

ScratchPadDriver: ScratchPadDriver.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/ScratchPadDriver.o -o ScratchPadDriver $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

ScratchPadDriver.o: $(DRIVERS_DIR)/ScratchPad/ScratchPadDriver.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 
  
EricksonDriver: EricksonDriver.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/EricksonDriver.o -o EricksonDriver $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

EricksonDriver.o: $(DRIVERS_DIR)/Confusion/EricksonDriver.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

ExperimentalInnerProduct.o: $(DRIVERS_DIR)/Confusion/ExperimentalInnerProduct.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

TimeMarchingBurgersDriver: TimeMarchingBurgersDriver.o $(OBJS) $(BURGERS_OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(BURGERS_OBJS_IN_PLACE) $(OBJ_DIR)/TimeMarchingBurgersDriver.o -o TimeMarchingBurgersDriver $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

TimeMarchingBurgersDriver.o: $(DRIVERS_DIR)/Burgers/TimeMarchingBurgersDriver.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

ConfectionDriver: ConfectionDriver.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/ConfectionDriver.o -o ConfectionDriver $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

ConfectionDriver.o: $(DRIVERS_DIR)/Confusion/ConfectionDriver.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

ConfectionManufacturedSolution.o: $(DRIVERS_DIR)/Confusion/ConfectionManufacturedSolution.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

EricksonManufacturedSolution.o: $(DRIVERS_DIR)/Confusion/EricksonManufacturedSolution.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

EricksonConfectionSolution.o: $(DRIVERS_DIR)/Confusion/EricksonConfectionSolution.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

StokesBackwardFacingStepDriver: StokesBackwardFacingStepDriver.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/StokesBackwardFacingStepDriver.o -o StokesBackwardFacingStepDriver $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

StokesCavityFlowDriver: StokesCavityFlowDriver.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/StokesCavityFlowDriver.o -o StokesCavityFlowDriver $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

NavierStokesCavityFlowDriver: NavierStokesCavityFlowDriver.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/NavierStokesCavityFlowDriver.o -o NavierStokesCavityFlowDriver $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

NavierStokesStudy: NavierStokesStudy.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/NavierStokesStudy.o -o StokesStudy $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

StokesStudy: StokesStudy.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/StokesStudy.o -o StokesStudy $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

StokesMathStudy: StokesMathStudy.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/StokesMathStudy.o -o StokesMathStudy $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

StokesStudyHybridMesh: StokesStudyHybridMesh.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/StokesStudyHybridMesh.o -o StokesStudyHybridMesh $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

ConfusionStudy: ConfusionStudy.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/ConfusionStudy.o -o ConfusionStudy $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

ScalingExperiment: ScalingExperiment.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/ScalingExperiment.o -o ScalingExperiment $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -lmpi_f77 -o $(OBJ_DIR)/$@

NewConfusionDriver: NewConfusionDriver.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/NewConfusionDriver.o -o NewConfusionDriver $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

NewBurgersDriver: NewBurgersDriver.o $(OBJS) $(BURGERS_OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(BURGERS_OBJS_IN_PLACE) $(OBJ_DIR)/NewBurgersDriver.o -o NewBurgersDriver $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

AdaptiveConfusion: AdaptiveConfusion.o $(OBJS)
	$(CXX) $(OBJS_IN_PLACE) $(OBJ_DIR)/AdaptiveConfusion.o -o AdaptiveConfusion $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(MPI_LIBRARIES) $(Trilinos_TPL_LIBRARIES) -o $(OBJ_DIR)/$@

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

ScratchPadTests.o: $(DRIVERS_DIR)/DPGTests/ScratchPadTests.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

ElementTests.o: $(DRIVERS_DIR)/DPGTests/ElementTests.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

FunctionTests.o: $(DRIVERS_DIR)/DPGTests/FunctionTests.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

LinearTermTests.o: $(DRIVERS_DIR)/DPGTests/LinearTermTests.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

IncompressibleFormulationsTests.o: $(DRIVERS_DIR)/DPGTests/IncompressibleFormulationsTests.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

BasisCacheTests.o: $(DRIVERS_DIR)/DPGTests/BasisCacheTests.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

RHSTests.o: $(DRIVERS_DIR)/DPGTests/RHSTests.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

PatchBasisTests.o: $(DRIVERS_DIR)/DPGTests/PatchBasisTests.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

TestBilinearFormDx.o: $(DRIVERS_DIR)/DPGTests/TestBilinearFormDx.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

TestBilinearFormFlux.o: $(DRIVERS_DIR)/DPGTests/TestBilinearFormFlux.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

MultiBasisTests.o: $(DRIVERS_DIR)/DPGTests/MultiBasisTests.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

VectorizedBasisTestSuite.o: $(DRIVERS_DIR)/DPGTests/VectorizedBasisTestSuite.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

BurgersBilinearForm.o: $(DRIVERS_DIR)/Burgers/BurgersBilinearForm.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

BurgersInnerProduct.o: $(DRIVERS_DIR)/Burgers/BurgersInnerProduct.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

BurgersProblem.o: $(DRIVERS_DIR)/Burgers/BurgersProblem.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

TimeMarchingBurgersProblem.o: $(DRIVERS_DIR)/Burgers/TimeMarchingBurgersProblem.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

ConfusionBilinearForm.o: $(DRIVERS_DIR)/Confusion/ConfusionBilinearForm.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

RieszRep.o: $(SRCS_DIR)/Solution/RieszRep.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

CGSolver.o: $(SRCS_DIR)/Solution/CGSolver.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

SchwarzSolver.o: $(SRCS_DIR)/Solution/SchwarzSolver.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

PenaltyMethodFilter.o: $(SRCS_DIR)/Solution/PenaltyMethodFilter.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@ 

ConfusionManufacturedSolution.o: $(DRIVERS_DIR)/Confusion/ConfusionManufacturedSolution.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@

ConfusionProblemLegacy.o: $(DRIVERS_DIR)/Confusion/ConfusionProblemLegacy.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $< -o $(OBJ_DIR)/$@

ConfusionStudy.o: $(DRIVERS_DIR)/Confusion/ConfusionStudy.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

ScalingExperiment.o: $(DRIVERS_DIR)/ScalingExperiment/ScalingExperiment.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

NewConfusionDriver.o: $(DRIVERS_DIR)/ScratchPad/NewConfusionDriver.cpp $(SRCS_DIR)/include/PreviousSolutionFunction.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

NewBurgersDriver.o: $(DRIVERS_DIR)/ScratchPad/NewBurgersDriver.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

AdaptiveConfusion.o: $(DRIVERS_DIR)/Confusion/AdaptiveConfusion.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

exampleMesh.o: $(DRIVERS_DIR)/RefTreeExperiment/exampleMesh.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

PoissonSolver.o: $(DRIVERS_DIR)/DemoDrivers/PoissonSolver.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

Poisson%.o: $(DRIVERS_DIR)/Poisson/Poisson%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

LidDrivenFlowRefinementStrategy.o: $(DRIVERS_DIR)/Stokes/LidDrivenFlowRefinementStrategy.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

NavierStokesCavityFlowDriver.o: $(DRIVERS_DIR)/Stokes/NavierStokesCavityFlowDriver.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

NavierStokesStudy.o: $(DRIVERS_DIR)/IncompressibleNS/NavierStokesStudy.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

Stokes%.o: $(DRIVERS_DIR)/Stokes/Stokes%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

Basis%.o: $(SRCS_DIR)/Basis/Basis%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

%Basis.o: $(SRCS_DIR)/Basis/%Basis.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

%InnerProduct.o: $(SRCS_DIR)/InnerProduct/%InnerProduct.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

HessianFilter.o: $(SRCS_DIR)/Problem/HessianFilter.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

TimeMarchingProblem.o: $(SRCS_DIR)/Problem/TimeMarchingProblem.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

RHS.o: $(SRCS_DIR)/Problem/RHS.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

Bilinear%.o: $(SRCS_DIR)/Problem/Bilinear%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

%olution.o: $(SRCS_DIR)/Solution/%olution.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

LagrangeConstraints.o: $(SRCS_DIR)/Solution/LagrangeConstraints.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

Elem%.o: $(SRCS_DIR)/Mesh/Elem%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

MeshPolyOrderFunction.o: $(SRCS_DIR)/Mesh/MeshPolyOrderFunction.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

MeshTestUtility.o: $(SRCS_DIR)/Mesh/MeshTestUtility.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

Mesh.o: $(SRCS_DIR)/Mesh/Mesh.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

Projector.o: $(SRCS_DIR)/Projection/Projector.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

BCEasy.o: $(SRCS_DIR)/ScratchPad/BCEasy.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

Constraint.o: $(SRCS_DIR)/ScratchPad/Constraint.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@
  
Function.o: $(SRCS_DIR)/ScratchPad/Function.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

SpatiallyFilteredFunction.o: $(SRCS_DIR)/ScratchPad/SpatiallyFilteredFunction.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

BF.o: $(SRCS_DIR)/ScratchPad/BF.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

IP.o: $(SRCS_DIR)/ScratchPad/IP.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

LinearTerm.o: $(SRCS_DIR)/ScratchPad/LinearTerm.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

RHSEasy.o: $(SRCS_DIR)/ScratchPad/RHSEasy.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

Var.o: $(SRCS_DIR)/ScratchPad/Var.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

MeshPartitionPolicy.o: $(SRCS_DIR)/Mesh/MeshPartitionPolicy.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

ZoltanMeshPartitionPolicy.o: $(SRCS_DIR)/Mesh/ZoltanMeshPartitionPolicy.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@

RefinementPattern.o: $(SRCS_DIR)/Mesh/RefinementPattern.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@
  
BC.o: $(SRCS_DIR)/Problem/BC.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@
  
BCFunction.o: $(SRCS_DIR)/Problem/BCFunction.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@
  
RefinementStrategy.o: $(SRCS_DIR)/Solution/RefinementStrategy.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(INCLUDE_PATHS) $<  -o $(OBJ_DIR)/$@
  
NonlinearSolveStrategy.o: $(SRCS_DIR)/Solution/NonlinearSolveStrategy.cpp
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
	$(RM) $(OBJS_IN_PLACE) $(TEST_OBJS_IN_PLACE) $(BURGERS_OBJS_IN_PLACE) $(EXES) $(EXE_OBJS)
