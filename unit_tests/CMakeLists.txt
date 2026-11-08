project(UnitTests)

FILE(GLOB TEST_SOURCES CONFIGURE_DEPENDS "*.cpp")
list(FILTER TEST_SOURCES EXCLUDE REGEX "Epetra_Operator_to_Epetra_MatrixTests\\.cpp$|OverlappingRowMatrixTests\\.cpp$")

# Tests of the tensor-product infrastructure are built into runTensorTests, not runTests.
set(TENSOR_TEST_SOURCES
  Teuchos_StandardParallelUnitTestMain.cpp
  TensorBasisCacheTests.cpp
  TensorCoefficientValueEvaluatorTests.cpp
  TensorSolutionTests.cpp
  TensorBasisTests.cpp
  TensorFunctionTests.cpp
  TensorFunctionEvaluationContextTests.cpp
  TensorFixedPointLifecycleTests.cpp
  TensorMeshTests.cpp
  TensorTraceDescriptorTests.cpp
  TensorTraceLayoutTests.cpp
)
foreach(tensorTestSource IN LISTS TENSOR_TEST_SOURCES)
  if(NOT tensorTestSource STREQUAL "Teuchos_StandardParallelUnitTestMain.cpp")
    list(REMOVE_ITEM TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${tensorTestSource}")
  endif()
endforeach()
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

target_link_libraries(runTensorTests ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} Camellia
)

add_test(NAME runTensorTests COMMAND runTensorTests)

install (TARGETS runTests runTensorTests DESTINATION bin/tests)
