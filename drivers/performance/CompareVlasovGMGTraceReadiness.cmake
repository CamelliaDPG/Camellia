# Readiness smoke for comparing p=2 matrix-free and assembled GMG V-cycle traces.
# This intentionally checks trace availability/schema rather than numerical equivalence:
# the current milestone is to ensure both independently generated paths expose
# comparable nonempty stage data that future tests can compare quantitatively.

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

function(count_comparable_gmg_trace_rows LABEL TRACE_TEXT PREFIX REQUIRE_FINE_P2 OUT_COUNT)
  string(REPLACE "\n" ";" TRACE_LINES "${TRACE_TEXT}")
  set(row_count 0)
  foreach(trace_line IN LISTS TRACE_LINES)
    if(NOT trace_line MATCHES "^${PREFIX}")
      continue()
    endif()
    if(REQUIRE_FINE_P2 AND NOT trace_line MATCHES "fineP=2")
      continue()
    endif()
    foreach(required_field IN ITEMS
        "initialResidualNorm"
        "residualAfterPreSmooth"
        "residualAfterCoarseCorrection"
        "residualAfterPostSmooth")
      if(NOT trace_line MATCHES "${required_field}=[-+0-9.eE]+")
        message(FATAL_ERROR
          "${LABEL} GMG trace row is missing comparable field '${required_field}': ${trace_line}")
      endif()
    endforeach()
    math(EXPR row_count "${row_count} + 1")
  endforeach()

  if(row_count EQUAL 0)
    message(FATAL_ERROR
      "${LABEL} GMG trace did not contain any comparable p=2 residual-stage rows with prefix ${PREFIX}")
  endif()
  set(${OUT_COUNT} "${row_count}" PARENT_SCOPE)
endfunction()

count_comparable_gmg_trace_rows(
  "matrix-free"
  "${MATRIX_FREE_TRACE_TEXT}"
  "VLASOV_MATRIX_FREE_GMG_STAGE_TRACE"
  TRUE
  MATRIX_FREE_TRACE_ROW_COUNT)
count_comparable_gmg_trace_rows(
  "assembled"
  "${ASSEMBLED_TRACE_TEXT}"
  "VLASOV_ASSEMBLED_GMG_VCYCLE_STAGE"
  FALSE
  ASSEMBLED_TRACE_ROW_COUNT)

message(STATUS
  "GMG trace readiness: matrix_free_rows=${MATRIX_FREE_TRACE_ROW_COUNT}; assembled_rows=${ASSEMBLED_TRACE_ROW_COUNT}; comparable_fields=initialResidualNorm,residualAfterPreSmooth,residualAfterCoarseCorrection,residualAfterPostSmooth")
