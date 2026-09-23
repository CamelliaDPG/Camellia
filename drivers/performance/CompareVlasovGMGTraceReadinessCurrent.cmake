# Readiness smoke for comparing p=2 matrix-free and assembled GMG V-cycle traces.
# This intentionally checks trace availability/schema rather than numerical equivalence:
# the current milestone is to ensure both independently generated paths expose
# comparable nonempty residual-stage data that future tests can compare quantitatively.

if(NOT DEFINED MATRIX_FREE_TRACE_FILE)
  message(FATAL_ERROR "MATRIX_FREE_TRACE_FILE is required")
endif()
if(NOT DEFINED ASSEMBLED_TRACE_FILE)
  message(FATAL_ERROR "ASSEMBLED_TRACE_FILE is required")
endif()

foreach(trace_file IN ITEMS "${MATRIX_FREE_TRACE_FILE}" "${ASSEMBLED_TRACE_FILE}")
  if(NOT EXISTS "${trace_file}")
    message(FATAL_ERROR "Required GMG trace file does not exist: ${trace_file}")
  endif()
endforeach()

file(READ "${MATRIX_FREE_TRACE_FILE}" MATRIX_FREE_TRACE_TEXT)
file(READ "${ASSEMBLED_TRACE_FILE}" ASSEMBLED_TRACE_TEXT)

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
      message(FATAL_ERROR
        "matrix-free GMG trace row is missing stage label: ${trace_line}")
    endif()
    if(NOT trace_line MATCHES "residualInfNorm=[-+0-9.eE]+")
      message(FATAL_ERROR
        "matrix-free GMG trace row is missing residualInfNorm value: ${trace_line}")
    endif()
    if(trace_line MATCHES "stage=initialResidualNorm")
      math(EXPR initial_count "${initial_count} + 1")
    endif()
    math(EXPR row_count "${row_count} + 1")
  endforeach()

  if(row_count EQUAL 0)
    message(FATAL_ERROR
      "matrix-free GMG trace did not contain any p=2 VLASOV_MATRIX_FREE_GMG_STAGE_TRACE rows")
  endif()
  if(initial_count EQUAL 0)
    message(FATAL_ERROR
      "matrix-free GMG trace did not contain an initialResidualNorm stage row")
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
          "assembled GMG trace row is missing comparable field '${required_field}': ${trace_line}")
      endif()
    endforeach()
    math(EXPR row_count "${row_count} + 1")
  endforeach()

  if(row_count EQUAL 0)
    message(FATAL_ERROR
      "assembled GMG trace did not contain any VLASOV_ASSEMBLED_GMG_VCYCLE_STAGE rows")
  endif()
  set(${OUT_COUNT} "${row_count}" PARENT_SCOPE)
endfunction()

require_matrix_free_stage_rows("${MATRIX_FREE_TRACE_TEXT}" MATRIX_FREE_TRACE_ROW_COUNT)
require_assembled_vcycle_rows("${ASSEMBLED_TRACE_TEXT}" ASSEMBLED_TRACE_ROW_COUNT)

message(STATUS
  "GMG trace readiness: matrix_free_stage_rows=${MATRIX_FREE_TRACE_ROW_COUNT}; assembled_vcycle_rows=${ASSEMBLED_TRACE_ROW_COUNT}; residual_field=residualInfNorm; assembled_fields=initialResidualNorm,residualAfterPreSmooth,residualAfterCoarseCorrection,residualAfterPostSmooth")
