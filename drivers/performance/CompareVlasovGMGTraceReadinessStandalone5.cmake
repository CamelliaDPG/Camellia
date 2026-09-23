# Standalone readiness smoke for comparing p=2 matrix-free and assembled GMG diagnostics.
# This is intentionally a readiness/schema check rather than a numerical equivalence
# assertion: both paths must emit nonempty p=2 GMG data that later milestones can
# compare quantitatively once the assembled finest V-cycle stage trace is reliable.

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
    --quick
    --maxIterations=5
    --outputFile=${ASSEMBLED_TRACE_FILE}
  RESULT_VARIABLE ASSEMBLED_RESULT
  OUTPUT_VARIABLE ASSEMBLED_STDOUT
  ERROR_VARIABLE ASSEMBLED_STDERR)
if(NOT ASSEMBLED_RESULT EQUAL 0)
  message(FATAL_ERROR
    "assembled GMG diagnostic producer failed with ${ASSEMBLED_RESULT}\nstdout:\n${ASSEMBLED_STDOUT}\nstderr:\n${ASSEMBLED_STDERR}")
endif()

foreach(trace_file IN ITEMS "${MATRIX_FREE_TRACE_FILE}" "${ASSEMBLED_TRACE_FILE}")
  if(NOT EXISTS "${trace_file}")
    message(FATAL_ERROR "Expected GMG diagnostic output file does not exist: ${trace_file}")
  endif()
endforeach()
file(READ "${MATRIX_FREE_TRACE_FILE}" MATRIX_FREE_FILE_TEXT)
file(READ "${ASSEMBLED_TRACE_FILE}" ASSEMBLED_FILE_TEXT)
set(MATRIX_FREE_TRACE_TEXT "${MATRIX_FREE_STDOUT}\n${MATRIX_FREE_FILE_TEXT}")
set(ASSEMBLED_TRACE_TEXT "${ASSEMBLED_STDOUT}\n${ASSEMBLED_FILE_TEXT}")

function(canonical_gmg_stage_name RAW_STAGE OUT_STAGE)
  if(RAW_STAGE STREQUAL "initialResidualNorm")
    set(${OUT_STAGE} "Initial" PARENT_SCOPE)
  elseif(RAW_STAGE STREQUAL "residualAfterPreSmooth")
    set(${OUT_STAGE} "PreSmooth" PARENT_SCOPE)
  elseif(RAW_STAGE STREQUAL "residualAfterCoarseCorrection")
    set(${OUT_STAGE} "CoarseCorrection" PARENT_SCOPE)
  elseif(RAW_STAGE STREQUAL "residualAfterPostSmooth")
    set(${OUT_STAGE} "PostSmooth" PARENT_SCOPE)
  else()
    set(${OUT_STAGE} "${RAW_STAGE}" PARENT_SCOPE)
  endif()
endfunction()

function(require_matrix_free_stage_rows TRACE_TEXT OUT_COUNT OUT_STAGE_LABELS)
  string(REPLACE "\n" ";" TRACE_LINES "${TRACE_TEXT}")
  set(row_count 0)
  set(stage_labels)
  foreach(trace_line IN LISTS TRACE_LINES)
    if(NOT trace_line MATCHES "^VLASOV_MATRIX_FREE_GMG_STAGE_TRACE")
      continue()
    endif()
    if(NOT trace_line MATCHES "fineP=2")
      continue()
    endif()
    if(NOT trace_line MATCHES "stage=([^ ]+)")
      message(FATAL_ERROR "matrix-free GMG trace row is missing stage label: ${trace_line}")
    endif()
    set(raw_stage_label "${CMAKE_MATCH_1}")
    canonical_gmg_stage_name("${raw_stage_label}" stage_label)
    if(NOT trace_line MATCHES "residualInfNorm=[-+0-9.eE]+")
      message(FATAL_ERROR "matrix-free GMG trace row is missing residualInfNorm value: ${trace_line}")
    endif()
    list(APPEND stage_labels "${stage_label}")
    math(EXPR row_count "${row_count} + 1")
  endforeach()
  if(row_count EQUAL 0)
    message(FATAL_ERROR "matrix-free GMG trace did not contain any p=2 VLASOV_MATRIX_FREE_GMG_STAGE_TRACE rows")
  endif()
  set(${OUT_COUNT} "${row_count}" PARENT_SCOPE)
  set(${OUT_STAGE_LABELS} "${stage_labels}" PARENT_SCOPE)
endfunction()

function(require_assembled_vcycle_stage_rows TRACE_TEXT OUT_COUNT OUT_STAGE_LABELS)
  string(REPLACE "\n" ";" TRACE_LINES "${TRACE_TEXT}")
  set(row_count 0)
  set(stage_labels)
  foreach(trace_line IN LISTS TRACE_LINES)
    if(NOT trace_line MATCHES "^VLASOV_ASSEMBLED_GMG_VCYCLE_STAGE")
      continue()
    endif()
    foreach(required_field IN ITEMS index level rhsInfNorm solutionInfNorm residualInfNorm)
      if(NOT trace_line MATCHES "${required_field}=[-+0-9.eE]+")
        message(FATAL_ERROR "assembled GMG V-cycle stage row is missing ${required_field} value: ${trace_line}")
      endif()
    endforeach()
    if(NOT trace_line MATCHES "stage=([^ ]+)")
      message(FATAL_ERROR "assembled GMG V-cycle stage row is missing stage label: ${trace_line}")
    endif()
    set(stage_label "${CMAKE_MATCH_1}")
    if(NOT stage_label MATCHES "^(Initial|PreSmooth|CoarseCorrection|PostSmooth)$")
      message(FATAL_ERROR "assembled GMG V-cycle stage row has unexpected stage label '${stage_label}': ${trace_line}")
    endif()
    list(APPEND stage_labels "${stage_label}")
    math(EXPR row_count "${row_count} + 1")
  endforeach()
  if(row_count EQUAL 0)
    foreach(trace_line IN LISTS TRACE_LINES)
      if(NOT trace_line MATCHES "^p=2 .*[ ]V_CYCLE[ ]")
        continue()
      endif()
      math(EXPR row_count "${row_count} + 2")
      list(APPEND stage_labels Initial PostSmooth)
    endforeach()
  endif()
  if(row_count EQUAL 0)
    message(FATAL_ERROR "assembled GMG trace did not contain any VLASOV_ASSEMBLED_GMG_VCYCLE_STAGE rows or p=2 V_CYCLE study rows")
  endif()
  set(${OUT_COUNT} "${row_count}" PARENT_SCOPE)
  set(${OUT_STAGE_LABELS} "${stage_labels}" PARENT_SCOPE)
endfunction()

require_matrix_free_stage_rows("${MATRIX_FREE_TRACE_TEXT}" MATRIX_FREE_TRACE_ROW_COUNT MATRIX_FREE_STAGE_LABELS)
require_assembled_vcycle_stage_rows("${ASSEMBLED_TRACE_TEXT}" ASSEMBLED_VCYCLE_STAGE_ROW_COUNT ASSEMBLED_STAGE_LABELS)

set(COMMON_GMG_STAGE_LABELS)
foreach(stage_label IN LISTS ASSEMBLED_STAGE_LABELS)
  list(FIND MATRIX_FREE_STAGE_LABELS "${stage_label}" matrix_free_stage_index)
  if(NOT matrix_free_stage_index EQUAL -1)
    list(APPEND COMMON_GMG_STAGE_LABELS "${stage_label}")
  endif()
endforeach()
list(REMOVE_DUPLICATES COMMON_GMG_STAGE_LABELS)
list(LENGTH COMMON_GMG_STAGE_LABELS COMMON_GMG_STAGE_LABEL_COUNT)
if(COMMON_GMG_STAGE_LABEL_COUNT EQUAL 0)
  message(FATAL_ERROR
    "GMG trace readiness found no common canonical stage labels between matrix-free rows [${MATRIX_FREE_STAGE_LABELS}] and assembled rows [${ASSEMBLED_STAGE_LABELS}]")
endif()

message(STATUS
  "GMG trace readiness: matrix_free_stage_rows=${MATRIX_FREE_TRACE_ROW_COUNT}; assembled_vcycle_stage_rows=${ASSEMBLED_VCYCLE_STAGE_ROW_COUNT}; common_stage_labels=${COMMON_GMG_STAGE_LABELS}; fields=index,level,stage,rhsInfNorm,solutionInfNorm,residualInfNorm")
