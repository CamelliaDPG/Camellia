project(UnitTests)

FILE(GLOB TEST_SOURCES CONFIGURE_DEPENDS "*.cpp")
list(FILTER TEST_SOURCES EXCLUDE REGEX "Epetra_Operator_to_Epetra_MatrixTests\\.cpp$|OverlappingRowMatrixTests\\.cpp$")
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/Epetra_Operator_to_Epetra_MatrixTests.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/OverlappingRowMatrixTests.cpp"
  PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE
)

add_executable(runTests ${TEST_SOURCES})

target_link_libraries(runTests ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} Camellia
)

add_test(NAME runTests COMMAND runTests)

set(PHASE0_TEST_SOURCES
  Teuchos_StandardParallelUnitTestMain.cpp
  TensorBasisCacheTests.cpp
  TensorBasisTests.cpp
  TensorProductLocalOperatorTests.cpp
  TensorLocalSchurComplementTests.cpp
  TensorFunctionTests.cpp
  TensorMeshTests.cpp
  TensorOperatorDataTests.cpp
)

add_executable(runPhase0Tests ${PHASE0_TEST_SOURCES})
add_dependencies(runPhase0Tests runTests)

target_link_libraries(runPhase0Tests ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} Camellia
)

add_test(NAME runPhase0Tests COMMAND runPhase0Tests)

install (TARGETS runTests runPhase0Tests DESTINATION bin/tests)