project(UnitTests)

FILE(GLOB TEST_SOURCES CONFIGURE_DEPENDS "*.cpp")
list(FILTER TEST_SOURCES EXCLUDE REGEX "Epetra_Operator_to_Epetra_MatrixTests\\.cpp$|OverlappingRowMatrixTests\\.cpp$|TensorPAMatrixTests\\.cpp$|TensorPAMatrixHGRADTests\\.cpp$|TensorPAMatrixBridgeTests\\.cpp$")
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/Epetra_Operator_to_Epetra_MatrixTests.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/OverlappingRowMatrixTests.cpp"
  PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE
)

add_executable(runTests ${TEST_SOURCES})

target_link_libraries(runTests ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} Camellia
)

add_test(NAME runTests COMMAND runTests)

set(TENSOR_TEST_SOURCES
  Teuchos_StandardParallelUnitTestMain.cpp
  TensorBasisCacheTests.cpp
  TensorCoefficientValueEvaluatorTests.cpp
  TensorCoefficientTpetraImportTests.cpp
  TensorSolutionTests.cpp
  TensorBasisTests.cpp
  TensorProductLocalOperatorTests.cpp
  TensorLocalSchurComplementTests.cpp
  TensorFunctionTests.cpp
  TensorFunctionEvaluationContextTests.cpp
  TensorCoupledCoefficientEvaluatorTests.cpp
  TensorFixedPointLifecycleTests.cpp
  TensorMeshTests.cpp
  TensorMomentDescriptorTests.cpp
  TensorMomentReductionTests.cpp
  TensorMaxwellRHSMomentDataTests.cpp
  TensorMaxwell1D2VMomentReductionPipelineTests.cpp
  TensorOperatorDataTests.cpp
  TensorPAMatrixTests.cpp
  TensorPAMatrixHGRADTests.cpp
  TensorPAMatrixBridgeTests.cpp
  TensorTraceDescriptorTests.cpp
  TensorTraceLayoutTests.cpp
)

add_executable(runTensorTests ${TENSOR_TEST_SOURCES})

set(TENSOR_INTREPID2_SOURCE_DIR
  "/Users/nvrober/fork-Trilinos/packages/intrepid2/src"
  CACHE PATH "Intrepid2 source directory from the PAMatrix-enabled Trilinos fork")
set(TENSOR_INTREPID2_PAMATRIX_INCLUDE_DIR
  "${TENSOR_INTREPID2_SOURCE_DIR}/Discretization/Integration"
  CACHE PATH "Directory containing experimental Intrepid2 PAMatrix headers")
target_include_directories(runTensorTests BEFORE PRIVATE
  ${TENSOR_INTREPID2_PAMATRIX_INCLUDE_DIR}
  ${TENSOR_INTREPID2_SOURCE_DIR}/Discretization/Basis
  ${TENSOR_INTREPID2_SOURCE_DIR}/Shared
  ${TENSOR_INTREPID2_SOURCE_DIR}/Cell
  ${TENSOR_INTREPID2_SOURCE_DIR}/../assembly-examples
)

target_link_libraries(runTensorTests ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} Camellia
)

if (TARGET TensorPoissonPAMatrixDriver)
  add_dependencies(runTensorTests TensorPoissonPAMatrixDriver)
endif()

add_test(NAME runTensorTests COMMAND runTensorTests)

install (TARGETS runTests runTensorTests DESTINATION bin/tests)