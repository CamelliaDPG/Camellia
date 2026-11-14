project(UnitTests)

FILE(GLOB TEST_SOURCES CONFIGURE_DEPENDS "*.cpp")
list(FILTER TEST_SOURCES EXCLUDE REGEX "Epetra_Operator_to_Epetra_MatrixTests\\.cpp$|OverlappingRowMatrixTests\\.cpp$")

# Tests of the tensor-product (matrix-free) infrastructure are built into runTensorTests, not runTests.
set(TENSOR_TEST_SOURCES
  Teuchos_StandardParallelUnitTestMain.cpp
  MatrixFreeInfrastructureTests.cpp
  MatrixFreeGMGPreconditionerTests.cpp
  MatrixFreeGMGPreconditionerComparisonTests.cpp
  MatrixFreeGMGPreconditionerStageTraceComparisonTests.cpp
  TensorHGradPTransferTests.cpp
  TensorLocalPMultigridTests.cpp
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
  TensorPlanarMITLMomentComparisonAdapterTests.cpp
  TensorOperatorDataTests.cpp
  TensorTraceDescriptorTests.cpp
  TensorTraceLayoutTests.cpp
)
# PAMatrix tests: need Intrepid2::PAMatrix and Intrepid2's (uninstalled) reference assembly header; see the top-level
# CMakeLists.txt (CAMELLIA_ENABLE_PAMATRIX, TRILINOS_SOURCE_DIR).
set(TENSOR_PAMATRIX_TEST_SOURCES
  TensorPAMatrixTests.cpp
  TensorPAMatrixHGRADTests.cpp
  TensorPAMatrixBridgeTests.cpp
)
foreach(tensorTestSource IN LISTS TENSOR_TEST_SOURCES TENSOR_PAMATRIX_TEST_SOURCES)
  if(NOT tensorTestSource STREQUAL "Teuchos_StandardParallelUnitTestMain.cpp")
    list(REMOVE_ITEM TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${tensorTestSource}")
  endif()
endforeach()
if(CAMELLIA_INTREPID2_ASSEMBLY_EXAMPLES_DIR)
  list(APPEND TENSOR_TEST_SOURCES ${TENSOR_PAMATRIX_TEST_SOURCES})
endif()
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/Epetra_Operator_to_Epetra_MatrixTests.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/OverlappingRowMatrixTests.cpp"
  PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE
)

add_executable(runTests ${TEST_SOURCES})

target_link_libraries(runTests ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} Camellia
)

add_test(NAME runTests COMMAND runTests)

add_executable(runTensorTests ${TENSOR_TEST_SOURCES})

if(CAMELLIA_INTREPID2_ASSEMBLY_EXAMPLES_DIR)
  target_include_directories(runTensorTests PRIVATE ${CAMELLIA_INTREPID2_ASSEMBLY_EXAMPLES_DIR})
endif()

target_link_libraries(runTensorTests ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} Camellia
)

add_test(NAME runTensorTests COMMAND runTensorTests)

install (TARGETS runTests runTensorTests DESTINATION bin/tests)
