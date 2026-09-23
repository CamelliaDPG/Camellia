# Standalone readiness smoke for comparing p=2 matrix-free and assembled GMG traces.
# This checks trace availability/schema rather than numerical equivalence: both paths
# must expose nonempty residual/operator data that later tests can compare quantitatively.

foreach(required_var IN ITEMS MATRIX_FREE_DRIVER ASSEMBLED_DRIVER TRACE_WORK_DIR)
  if(NOT DEFINED ${required_var})
    message(FATAL_ERROR "${required_var} is required")
  endif()
endforeach()

file(MAKE_DIRECTORY "${TRACE_WORK_DIR}")
set(MATRIX_FREE_TRACE_FILE "${TRACE_WORK_DIR}/vlasov-gram-trace-readiness-matrix-free.txt")
set(ASSEMBLED_TRACE_FILE "${TRACE_WORK_DIR}/vlasov-gram-trace-readiness-assembled.txt")
file(REMOVE "${MATRIX_FREE_TRACE_FILE}" "${ASSEMBLED_TRACE_FILE}")

execute_process(
  COMMAND "${MATRIX_FREE_DRIVER}"
    --fineH1Order=2
    --quick
    --maxIterations=20
    --matrixFreeGMGStageTrace=1
    --keepChildOutput=1
    --maxAssembledEntries=10000
    --outputFile=${MATRIX_FREE_TRACE_FILE}
  RESULT_VARIABLE MATRIX_FREE_RESULT
  OUTPUT_VARIABLE MATRIX_FREE_STDOUT
  ERROR_VARIABLE MATRIX_FREE_STDERR)
if(NOT MATRIX_FREE_RESULT EQUAL 0)
  message(FATAL_ERROR
    "matrix-free GMG trace producer failed with ${MATRIX_FREE_RESULT}\nstdout:\n${MATRIX_FREE_STDOUT}\nstderr:\n${MATRIX_FREE_STDERR}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env VLASOV_ASSEMBLED_GMG_EMIT_STAGE_TRACE=1
    "${ASSEMBLED_DRIVER}"
    --fineH1Order=2
    --maxIterations=5
    --outputFile=${ASSEMBLED_TRACE_FILE}
  RESULT_VARIABLE ASSEMBLED_RESULT
  OUTPUT_VARIABLE ASSEMBLED_STDOUT
  ERROR_VARIABLE ASSEMBLED_STDERR)
if(NOT ASSEMBLED_RESULT EQUAL 0)
  message(FATAL_ERROR
    "assembled GMG trace producer failed with ${ASSEMBLED_RESULT}\nstdout:\n${ASSEMBLED_STDOUT}\nstderr:\n${ASSEMBLED_STDERR}")
endif()

foreach(trace_file IN ITEMS "${MATRIX_FREE_TRACE_FILE}" "${ASSEMBLED_TRACE_FILE}")
  if(NOT EXISTS "${trace_file}")
    message(FATAL_ERROR "Expected GMG trace output file does not exist: ${trace_file}")
  endif()
endforeach()
file(READ "${MATRIX_FREE_TRACE_FILE}" MATRIX_FREE_FILE_TEXT)
file(READ "${ASSEMBLED_TRACE_FILE}" ASSEMBLED_FILE_TEXT)
set(MATRIX_FREE_TRACE_TEXT "${MATRIX_FREE_STDOUT}\n${MATRIX_FREE_FILE_TEXT}")
set(ASSEMBLED_TRACE_TEXT "${ASSEMBLED_STDOUT}\n${ASSEMBLED_FILE_TEXT}")

function(require_matrix_free_stage_rows TRACE_TEXT OUT_COUNT)
  string(REPLACE "\n" ";" TRACE_LINES "${TRACE_TEXT}")
  set(row_count 0)
  set(initial_count 0)
  foreach(trace_line IN LISTS TRACE_LINES)
    if(NOT trace_line MATCHES "^VLASOV_MATRIX_FREE_GMG_STAGE_TRACE")
      continue()
    endif()
    if(NOT trace_line MATCHES "fineP=2")
      continue()
    endif()
    if(NOT trace_line MATCHES "stage=[A-Za-z0-9_]+")
      message(FATAL_ERROR "matrix-free GMG trace row is missing stage label: ${trace_line}")
    endif()
    if(NOT trace_line MATCHES "residualInfNorm=[-+0-9.eE]+")
      message(FATAL_ERROR "matrix-free GMG trace row is missing residualInfNorm value: ${trace_line}")
    endif()
    if(trace_line MATCHES "stage=initialResidualNorm")
      math(EXPR initial_count "${initial_count} + 1")
    endif()
    math(EXPR row_count "${row_count} + 1")
  endforeach()
  if(row_count EQUAL 0)
    message(FATAL_ERROR "matrix-free GMG trace did not contain any p=2 VLASOV_MATRIX_FREE_GMG_STAGE_TRACE rows")
  endif()
  if(initial_count EQUAL 0)
    message(FATAL_ERROR "matrix-free GMG trace did not contain an initialResidualNorm stage row")
  endif()
  set(${OUT_COUNT} "${row_count}" PARENT_SCOPE)
endfunction()

function(require_assembled_vcycle_rows TRACE_TEXT OUT_COUNT)
  string(REPLACE "\n" ";" TRACE_LINES "${TRACE_TEXT}")
  set(row_count 0)
  foreach(trace_line IN LISTS TRACE_LINES)
    if(NOT trace_line MATCHES "^VLASOV_ASSEMBLED_GMG_VCYCLE_STAGE")
      continue()
    endif()
    foreach(required_field IN ITEMS
        "initialResidualNorm"
        "residualAfterPreSmooth"
        "residualAfterCoarseCorrection"
        "residualAfterPostSmooth")
      if(NOT trace_line MATCHES "${required_field}=[-+0-9.eE]+")
        message(FATAL_ERROR
          "assembled GMG V-cycle trace row is missing field '${required_field}': ${trace_line}")
      endif()
    endforeach()
    math(EXPR row_count "${row_count} + 1")
  endforeach()
  if(row_count EQUAL 0)
    message(FATAL_ERROR "assembled GMG trace did not contain any VLASOV_ASSEMBLED_GMG_VCYCLE_STAGE rows")
  endif()
  set(${OUT_COUNT} "${row_count}" PARENT_SCOPE)
endfunction()

require_matrix_free_stage_rows("${MATRIX_FREE_TRACE_TEXT}" MATRIX_FREE_TRACE_ROW_COUNT)
require_assembled_vcycle_rows("${ASSEMBLED_TRACE_TEXT}" ASSEMBLED_VCYCLE_ROW_COUNT)

message(STATUS
  "GMG trace readiness: matrix_free_stage_rows=${MATRIX_FREE_TRACE_ROW_COUNT}; assembled_vcycle_rows=${ASSEMBLED_VCYCLE_ROW_COUNT}; matrix_free_residual_field=residualInfNorm; assembled_fields=initialResidualNorm,residualAfterPreSmooth,residualAfterCoarseCorrection,residualAfterPostSmooth")
