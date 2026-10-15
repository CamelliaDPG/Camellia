# Standalone readiness smoke for comparing p=${TRACE_FINE_P} matrix-free and assembled GMG diagnostics.
# This is intentionally a readiness/schema check rather than a numerical equivalence
# assertion: both paths must emit nonempty p=${TRACE_FINE_P} GMG data that later milestones can
# compare quantitatively once the assembled finest V-cycle stage trace is reliable.
# Both producers are pinned to --basisFamily=hierarchical: the assembled driver
# defaults to nodal while the matrix-free driver defaults to hierarchical, and
# mixing them makes the Initial rhsInfNorm differ (16/81 vs 1/16).

foreach(required_var IN ITEMS MATRIX_FREE_DRIVER ASSEMBLED_DRIVER TRACE_WORK_DIR)
  if(NOT DEFINED ${required_var})
    message(FATAL_ERROR "${required_var} is required")
  endif()
endforeach()

if(NOT DEFINED TRACE_STRATEGY)
  set(TRACE_STRATEGY pmg-chebyshev-smoother-direct-coarse)
endif()
if(NOT DEFINED TRACE_FINE_P)
  set(TRACE_FINE_P 2)
endif()
if(NOT DEFINED TRACE_MAX_ITERATIONS)
  set(TRACE_MAX_ITERATIONS 20)
endif()
if(NOT DEFINED TRACE_MATRIX_FREE_MAX_ITERATIONS)
  set(TRACE_MATRIX_FREE_MAX_ITERATIONS ${TRACE_MAX_ITERATIONS})
endif()
if(NOT DEFINED TRACE_ASSEMBLED_MAX_ITERATIONS)
  set(TRACE_ASSEMBLED_MAX_ITERATIONS ${TRACE_MAX_ITERATIONS})
endif()
if(NOT DEFINED TRACE_MAX_ASSEMBLED_ENTRIES)
  set(TRACE_MAX_ASSEMBLED_ENTRIES 10000)
endif()

file(MAKE_DIRECTORY "${TRACE_WORK_DIR}")
set(MATRIX_FREE_TRACE_FILE "${TRACE_WORK_DIR}/vlasov-gram-trace-readiness-matrix-free.txt")
set(ASSEMBLED_TRACE_FILE "${TRACE_WORK_DIR}/vlasov-gram-trace-readiness-assembled.txt")
set(ASSEMBLED_GERSHGORIN_TRACE_FILE "${TRACE_WORK_DIR}/vlasov-gram-trace-readiness-assembled-gershgorin.txt")
set(MATRIX_FREE_GERSHGORIN_TRACE_FILE "${TRACE_WORK_DIR}/vlasov-gram-trace-readiness-matrix-free-gershgorin.txt")
set(ASSEMBLED_DETERMINISTIC_POWER_TRACE_FILE "${TRACE_WORK_DIR}/vlasov-gram-trace-readiness-assembled-deterministic-power.txt")
set(MATRIX_FREE_DETERMINISTIC_POWER_TRACE_FILE "${TRACE_WORK_DIR}/vlasov-gram-trace-readiness-matrix-free-deterministic-power.txt")
set(STAGE_COMPARISON_FILE "${TRACE_WORK_DIR}/vlasov-gram-trace-readiness-p${TRACE_FINE_P}-stage-comparison.txt")
file(REMOVE "${MATRIX_FREE_TRACE_FILE}" "${ASSEMBLED_TRACE_FILE}" "${ASSEMBLED_GERSHGORIN_TRACE_FILE}" "${MATRIX_FREE_GERSHGORIN_TRACE_FILE}" "${ASSEMBLED_DETERMINISTIC_POWER_TRACE_FILE}" "${MATRIX_FREE_DETERMINISTIC_POWER_TRACE_FILE}" "${STAGE_COMPARISON_FILE}")

execute_process(
  COMMAND "${MATRIX_FREE_DRIVER}"
    --fineH1Order=${TRACE_FINE_P}
    --quick
    --maxIterations=${TRACE_MATRIX_FREE_MAX_ITERATIONS}
    --basisFamily=hierarchical
    --matrixFreeGMGStageTrace=1
    --keepChildOutput=1
    --maxAssembledEntries=${TRACE_MAX_ASSEMBLED_ENTRIES}
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
    --fineH1Order=${TRACE_FINE_P}
    --basisFamily=hierarchical
    --emitStageTrace
    --stageTraceFineP=${TRACE_FINE_P}
    --quick
    --maxIterations=${TRACE_ASSEMBLED_MAX_ITERATIONS}
    --outputFile=${ASSEMBLED_TRACE_FILE}
  RESULT_VARIABLE ASSEMBLED_RESULT
  OUTPUT_VARIABLE ASSEMBLED_STDOUT
  ERROR_VARIABLE ASSEMBLED_STDERR)
if(NOT ASSEMBLED_RESULT EQUAL 0)
  message(FATAL_ERROR
    "assembled GMG diagnostic producer failed with ${ASSEMBLED_RESULT}\nstdout:\n${ASSEMBLED_STDOUT}\nstderr:\n${ASSEMBLED_STDERR}")
endif()

# Controlled experiment, matrix-free side: keep the default Ifpack2 fine smoother and
# seed its Chebyshev lambdaMax/min eigenvalue bounds from a dense-free RowMatrix
# Gershgorin bound of D^-1 A (no Ifpack2 power iteration).
execute_process(
  COMMAND "${MATRIX_FREE_DRIVER}"
    --fineH1Order=${TRACE_FINE_P}
    --quick
    --maxIterations=${TRACE_MATRIX_FREE_MAX_ITERATIONS}
    --basisFamily=hierarchical
    --matrixFreeGMGStageTrace=1
    --keepChildOutput=1
    --maxAssembledEntries=${TRACE_MAX_ASSEMBLED_ENTRIES}
    --pMultigridChebyshevLambdaMax=gershgorin
    --outputFile=${MATRIX_FREE_GERSHGORIN_TRACE_FILE}
  RESULT_VARIABLE MATRIX_FREE_GERSHGORIN_RESULT
  OUTPUT_VARIABLE MATRIX_FREE_GERSHGORIN_STDOUT
  ERROR_VARIABLE MATRIX_FREE_GERSHGORIN_STDERR)

# Controlled experiment, assembled side: identical assembled run, but lambdaMax = Gershgorin bound of D^-1 A
# (lambdaMin = lambdaMax/ratio, no power iterations). Both sides are then Ifpack2 Chebyshev with an identical
# deterministic lambdaMax.
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env VLASOV_ASSEMBLED_GMG_EMIT_STAGE_TRACE=1
    "${ASSEMBLED_DRIVER}"
    --fineH1Order=${TRACE_FINE_P}
    --basisFamily=hierarchical
    --emitStageTrace
    --stageTraceFineP=${TRACE_FINE_P}
    --quick
    --maxIterations=${TRACE_ASSEMBLED_MAX_ITERATIONS}
    --assembledChebyshevLambdaMax=gershgorin
    --outputFile=${ASSEMBLED_GERSHGORIN_TRACE_FILE}
  RESULT_VARIABLE ASSEMBLED_GERSHGORIN_RESULT
  OUTPUT_VARIABLE ASSEMBLED_GERSHGORIN_STDOUT
  ERROR_VARIABLE ASSEMBLED_GERSHGORIN_STDERR)

# Controlled experiment, matrix-free side: seed Chebyshev lambdaMax/min eigenvalue bounds
# from the shared deterministic D^-1 A power estimate (no Ifpack2 power iteration).
execute_process(
  COMMAND "${MATRIX_FREE_DRIVER}"
    --fineH1Order=${TRACE_FINE_P}
    --quick
    --maxIterations=${TRACE_MATRIX_FREE_MAX_ITERATIONS}
    --basisFamily=hierarchical
    --matrixFreeGMGStageTrace=1
    --keepChildOutput=1
    --maxAssembledEntries=${TRACE_MAX_ASSEMBLED_ENTRIES}
    --pMultigridChebyshevLambdaMax=deterministic-power
    --outputFile=${MATRIX_FREE_DETERMINISTIC_POWER_TRACE_FILE}
  RESULT_VARIABLE MATRIX_FREE_DETERMINISTIC_POWER_RESULT
  OUTPUT_VARIABLE MATRIX_FREE_DETERMINISTIC_POWER_STDOUT
  ERROR_VARIABLE MATRIX_FREE_DETERMINISTIC_POWER_STDERR)

# Controlled experiment, assembled side: identical assembled run, but lambdaMax = deterministic D^-1 A power estimate
# (lambdaMin = lambdaMax/ratio, no Ifpack2 power iterations).
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env VLASOV_ASSEMBLED_GMG_EMIT_STAGE_TRACE=1
    "${ASSEMBLED_DRIVER}"
    --fineH1Order=${TRACE_FINE_P}
    --basisFamily=hierarchical
    --emitStageTrace
    --stageTraceFineP=${TRACE_FINE_P}
    --quick
    --maxIterations=${TRACE_ASSEMBLED_MAX_ITERATIONS}
    --assembledChebyshevLambdaMax=deterministic-power
    --outputFile=${ASSEMBLED_DETERMINISTIC_POWER_TRACE_FILE}
  RESULT_VARIABLE ASSEMBLED_DETERMINISTIC_POWER_RESULT
  OUTPUT_VARIABLE ASSEMBLED_DETERMINISTIC_POWER_STDOUT
  ERROR_VARIABLE ASSEMBLED_DETERMINISTIC_POWER_STDERR)

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
    if(NOT trace_line MATCHES " strategy=${TRACE_STRATEGY}( |$)")
      continue()
    endif()
    if(NOT trace_line MATCHES " fineP=${TRACE_FINE_P}( |$)")
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
    message(FATAL_ERROR "matrix-free GMG trace did not contain any p=${TRACE_FINE_P} VLASOV_MATRIX_FREE_GMG_STAGE_TRACE rows")
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
    if(NOT trace_line MATCHES " strategy=${TRACE_STRATEGY}( |$)")
      continue()
    endif()
    if(NOT trace_line MATCHES " fineP=${TRACE_FINE_P}( |$)")
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
    message(FATAL_ERROR "assembled GMG trace did not contain any real VLASOV_ASSEMBLED_GMG_VCYCLE_STAGE rows")
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
foreach(required_stage_label IN ITEMS Initial PreSmooth CoarseCorrection PostSmooth)
  list(FIND COMMON_GMG_STAGE_LABELS "${required_stage_label}" required_stage_index)
  if(required_stage_index EQUAL -1)
    message(FATAL_ERROR
      "GMG trace readiness is missing required common canonical stage label '${required_stage_label}'; matrix_free rows [${MATRIX_FREE_STAGE_LABELS}]; assembled rows [${ASSEMBLED_STAGE_LABELS}]")
  endif()
endforeach()

function(record_gmg_stage_metrics TRACE_TEXT ROW_PREFIX OUT_PREFIX)
  string(REPLACE "\n" ";" TRACE_LINES "${TRACE_TEXT}")
  set(seen_stage_labels)
  set(recorded_metric_row_count 0)
  foreach(trace_line IN LISTS TRACE_LINES)
    if(NOT trace_line MATCHES "^${ROW_PREFIX}")
      continue()
    endif()
    if(NOT trace_line MATCHES " strategy=${TRACE_STRATEGY}( |$)")
      continue()
    endif()
    if((ROW_PREFIX STREQUAL "VLASOV_ASSEMBLED_GMG_VCYCLE_STAGE") OR (ROW_PREFIX STREQUAL "VLASOV_MATRIX_FREE_GMG_STAGE_TRACE"))
      if(NOT trace_line MATCHES " fineP=${TRACE_FINE_P}( |$)")
        continue()
      endif()
    endif()
    if(NOT trace_line MATCHES " stage=([^ ]+)")
      message(FATAL_ERROR "${ROW_PREFIX} stage row is missing stage label: ${trace_line}")
    endif()
    set(raw_stage_label "${CMAKE_MATCH_1}")
    canonical_gmg_stage_name("${raw_stage_label}" stage_label)
    if(stage_label IN_LIST seen_stage_labels)
      continue()
    endif()
    string(MAKE_C_IDENTIFIER "${stage_label}" stage_key)
    foreach(metric IN ITEMS rhsInfNorm solutionInfNorm residualInfNorm)
      if(NOT trace_line MATCHES " ${metric}=([^ ]+)")
        message(FATAL_ERROR "${ROW_PREFIX} stage row is missing ${metric}: ${trace_line}")
      endif()
      set(${OUT_PREFIX}_${stage_key}_${metric} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    endforeach()
    set(${OUT_PREFIX}_${stage_key}_rawRow "${trace_line}" PARENT_SCOPE)
    list(APPEND seen_stage_labels "${stage_label}")
    math(EXPR recorded_metric_row_count "${recorded_metric_row_count} + 1")
  endforeach()
  set(${OUT_PREFIX}_labels "${seen_stage_labels}" PARENT_SCOPE)
  set(${OUT_PREFIX}_metric_row_count "${recorded_metric_row_count}" PARENT_SCOPE)
endfunction()

record_gmg_stage_metrics("${MATRIX_FREE_TRACE_TEXT}" "VLASOV_MATRIX_FREE_GMG_STAGE_TRACE" MATRIX_FREE_STAGE_METRIC)
record_gmg_stage_metrics("${ASSEMBLED_TRACE_TEXT}" "VLASOV_ASSEMBLED_GMG_VCYCLE_STAGE" ASSEMBLED_STAGE_METRIC)
if(ASSEMBLED_STAGE_METRIC_metric_row_count EQUAL 0)
  message(FATAL_ERROR "assembled GMG trace did not contain any real VLASOV_ASSEMBLED_GMG_VCYCLE_STAGE rows with numeric metrics; p=${TRACE_FINE_P} V_CYCLE summary rows cannot be used for rhsInfNorm/solutionInfNorm/residualInfNorm comparison")
endif()

set(COMPARED_GMG_STAGE_METRIC_COUNT 0)
set(GMG_STAGE_NUMERICAL_MISMATCH_COUNT 0)
set(GMG_STAGE_NUMERICAL_FIRST_MISMATCH "none")
set(GMG_STAGE_NUMERICAL_FIRST_MISMATCH_ASSEMBLED_ROW "none")
set(GMG_STAGE_NUMERICAL_FIRST_MISMATCH_MATRIX_FREE_ROW "none")
foreach(stage_label IN LISTS COMMON_GMG_STAGE_LABELS)
  string(MAKE_C_IDENTIFIER "${stage_label}" stage_key)
  foreach(metric IN ITEMS rhsInfNorm solutionInfNorm residualInfNorm)
    set(assembled_metric_var "ASSEMBLED_STAGE_METRIC_${stage_key}_${metric}")
    set(matrix_free_metric_var "MATRIX_FREE_STAGE_METRIC_${stage_key}_${metric}")
    if(NOT DEFINED ${assembled_metric_var})
      message(FATAL_ERROR "assembled GMG trace is missing ${metric} for common stage label '${stage_label}'")
    endif()
    if(NOT DEFINED ${matrix_free_metric_var})
      message(FATAL_ERROR "matrix-free GMG trace is missing ${metric} for common stage label '${stage_label}'")
    endif()
    if(NOT "${${assembled_metric_var}}" EQUAL "${${matrix_free_metric_var}}")
      math(EXPR GMG_STAGE_NUMERICAL_MISMATCH_COUNT "${GMG_STAGE_NUMERICAL_MISMATCH_COUNT} + 1")
      if(GMG_STAGE_NUMERICAL_FIRST_MISMATCH STREQUAL "none")
        set(GMG_STAGE_NUMERICAL_FIRST_MISMATCH "stage=${stage_label};metric=${metric};assembled=${${assembled_metric_var}};matrix_free=${${matrix_free_metric_var}}")
        set(assembled_raw_row_var "ASSEMBLED_STAGE_METRIC_${stage_key}_rawRow")
        set(matrix_free_raw_row_var "MATRIX_FREE_STAGE_METRIC_${stage_key}_rawRow")
        if(DEFINED ${assembled_raw_row_var})
          set(GMG_STAGE_NUMERICAL_FIRST_MISMATCH_ASSEMBLED_ROW "${${assembled_raw_row_var}}")
        endif()
        if(DEFINED ${matrix_free_raw_row_var})
          set(GMG_STAGE_NUMERICAL_FIRST_MISMATCH_MATRIX_FREE_ROW "${${matrix_free_raw_row_var}}")
        endif()
      endif()
    endif()
    math(EXPR COMPARED_GMG_STAGE_METRIC_COUNT "${COMPARED_GMG_STAGE_METRIC_COUNT} + 1")
  endforeach()
endforeach()

set(ASSEMBLED_CHEBYSHEV_BOUNDS_TAIL "missing")
string(REGEX MATCHALL "VLASOV_ASSEMBLED_GMG_CHEBYSHEV_BOUNDS[^\n]*" ASSEMBLED_CHEBYSHEV_BOUNDS_ROWS "${ASSEMBLED_TRACE_TEXT}")
foreach(bounds_line IN LISTS ASSEMBLED_CHEBYSHEV_BOUNDS_ROWS)
  if(NOT bounds_line MATCHES " strategy=${TRACE_STRATEGY}( |$)")
    continue()
  endif()
  if(NOT bounds_line MATCHES " fineP=${TRACE_FINE_P}( |$)")
    continue()
  endif()
  string(REGEX REPLACE "^VLASOV_ASSEMBLED_GMG_CHEBYSHEV_BOUNDS *" "" ASSEMBLED_CHEBYSHEV_BOUNDS_TAIL "${bounds_line}")
  string(STRIP "${ASSEMBLED_CHEBYSHEV_BOUNDS_TAIL}" ASSEMBLED_CHEBYSHEV_BOUNDS_TAIL)
  break()
endforeach()

set(MATRIX_FREE_CHEBYSHEV_BOUNDS_TAIL "missing")
string(REGEX MATCHALL "VLASOV_MATRIX_FREE_CHEBYSHEV_BOUNDS[^\n]*" MATRIX_FREE_CHEBYSHEV_BOUNDS_ROWS "${MATRIX_FREE_TRACE_TEXT}")
foreach(bounds_line IN LISTS MATRIX_FREE_CHEBYSHEV_BOUNDS_ROWS)
  if((bounds_line MATCHES " strategy=${TRACE_STRATEGY}( |$)") AND (bounds_line MATCHES " fineP=${TRACE_FINE_P}( |$)"))
    string(REGEX REPLACE "^VLASOV_MATRIX_FREE_CHEBYSHEV_BOUNDS *" "" MATRIX_FREE_CHEBYSHEV_BOUNDS_TAIL "${bounds_line}")
    string(STRIP "${MATRIX_FREE_CHEBYSHEV_BOUNDS_TAIL}" MATRIX_FREE_CHEBYSHEV_BOUNDS_TAIL)
    break()
  endif()
endforeach()
if(MATRIX_FREE_CHEBYSHEV_BOUNDS_TAIL STREQUAL "missing")
  foreach(bounds_line IN LISTS MATRIX_FREE_CHEBYSHEV_BOUNDS_ROWS)
    if(NOT bounds_line MATCHES " fineP=${TRACE_FINE_P}( |$)")
      continue()
    endif()
    string(REGEX REPLACE "^VLASOV_MATRIX_FREE_CHEBYSHEV_BOUNDS *" "" MATRIX_FREE_CHEBYSHEV_BOUNDS_TAIL "${bounds_line}")
    string(STRIP "${MATRIX_FREE_CHEBYSHEV_BOUNDS_TAIL}" MATRIX_FREE_CHEBYSHEV_BOUNDS_TAIL)
    break()
  endforeach()
endif()

# Convert a decimal float string to a 15-significant-digit integer mantissa and decimal exponent
# (value ~= 0.<mantissa> * 10^exponent). CMake has no float math, so the relative comparison below
# is done in 64-bit integers with math(EXPR).
function(gmg_sig_mantissa VALUE OUT_MANTISSA OUT_EXPONENT)
  string(STRIP "${VALUE}" v)
  if(NOT v MATCHES "^([-+]?)([0-9]*)([.]([0-9]*))?([eE]([-+]?[0-9]+))?$")
    set(${OUT_MANTISSA} "nan" PARENT_SCOPE)
    set(${OUT_EXPONENT} "0" PARENT_SCOPE)
    return()
  endif()
  set(sign "${CMAKE_MATCH_1}")
  set(int_part "${CMAKE_MATCH_2}")
  set(frac_part "${CMAKE_MATCH_4}")
  set(exp_part "${CMAKE_MATCH_6}")
  if(exp_part STREQUAL "")
    set(exp_part 0)
  endif()
  string(REGEX REPLACE "^[+]" "" exp_part "${exp_part}")
  string(REGEX REPLACE "^(-?)0+([0-9])" "\\1\\2" exp_part "${exp_part}")
  string(LENGTH "${int_part}" int_len)
  set(digits "${int_part}${frac_part}")
  math(EXPR point "${int_len} + ${exp_part}")
  while(digits MATCHES "^0(.*)$")
    set(digits "${CMAKE_MATCH_1}")
    math(EXPR point "${point} - 1")
  endwhile()
  if(digits STREQUAL "")
    set(${OUT_MANTISSA} "0" PARENT_SCOPE)
    set(${OUT_EXPONENT} "zero" PARENT_SCOPE)
    return()
  endif()
  string(SUBSTRING "${digits}000000000000000" 0 15 mantissa)
  if(sign STREQUAL "-")
    set(mantissa "-${mantissa}")
  endif()
  set(${OUT_MANTISSA} "${mantissa}" PARENT_SCOPE)
  set(${OUT_EXPONENT} "${point}" PARENT_SCOPE)
endfunction()

# agree=1 iff |a-b| <= 1e-8 * max(|a|,|b|) (evaluated on 15-digit integer mantissas).
function(gmg_relative_agree A B OUT_AGREE)
  gmg_sig_mantissa("${A}" ma ea)
  gmg_sig_mantissa("${B}" mb eb)
  if((ma STREQUAL "nan") OR (mb STREQUAL "nan"))
    set(${OUT_AGREE} 0 PARENT_SCOPE)
    return()
  endif()
  if((ea STREQUAL "zero") AND (eb STREQUAL "zero"))
    set(${OUT_AGREE} 1 PARENT_SCOPE)
    return()
  endif()
  if(ea STREQUAL "zero")
    set(ea "${eb}")
  endif()
  if(eb STREQUAL "zero")
    set(eb "${ea}")
  endif()
  math(EXPR shift "${ea} - ${eb}")
  if(shift EQUAL 1)
    math(EXPR ma "${ma} * 10")
  elseif(shift EQUAL -1)
    math(EXPR mb "${mb} * 10")
  elseif(NOT shift EQUAL 0)
    set(${OUT_AGREE} 0 PARENT_SCOPE)
    return()
  endif()
  math(EXPR diff "${ma} - ${mb}")
  if(diff LESS 0)
    math(EXPR diff "${mb} - ${ma}")
  endif()
  string(REGEX REPLACE "^-" "" abs_a "${ma}")
  string(REGEX REPLACE "^-" "" abs_b "${mb}")
  set(scale "${abs_a}")
  if(abs_b GREATER abs_a)
    set(scale "${abs_b}")
  endif()
  math(EXPR tol "${scale} / 100000000")
  if(diff GREATER tol)
    set(${OUT_AGREE} 0 PARENT_SCOPE)
  else()
    set(${OUT_AGREE} 1 PARENT_SCOPE)
  endif()
endfunction()

set(GMG_GERSHGORIN_EXPERIMENT_SUMMARY "not_run")
set(GMG_GERSHGORIN_EXPERIMENT_ALL_AGREE 0)
set(ASSEMBLED_GERSHGORIN_BOUNDS_TAIL "missing")
set(MATRIX_FREE_GERSHGORIN_BOUNDS_TAIL "missing")
if((ASSEMBLED_GERSHGORIN_RESULT EQUAL 0) AND (EXISTS "${ASSEMBLED_GERSHGORIN_TRACE_FILE}")
   AND (MATRIX_FREE_GERSHGORIN_RESULT EQUAL 0) AND (EXISTS "${MATRIX_FREE_GERSHGORIN_TRACE_FILE}"))
  file(READ "${MATRIX_FREE_GERSHGORIN_TRACE_FILE}" MATRIX_FREE_GERSHGORIN_FILE_TEXT)
  set(MATRIX_FREE_GERSHGORIN_TRACE_TEXT "${MATRIX_FREE_GERSHGORIN_STDOUT}\n${MATRIX_FREE_GERSHGORIN_FILE_TEXT}")
  record_gmg_stage_metrics("${MATRIX_FREE_GERSHGORIN_TRACE_TEXT}" "VLASOV_MATRIX_FREE_GMG_STAGE_TRACE" MATRIX_FREE_GERSHGORIN_STAGE_METRIC)
  string(REGEX MATCHALL "VLASOV_MATRIX_FREE_CHEBYSHEV_BOUNDS[^\n]*" MATRIX_FREE_GERSHGORIN_BOUNDS_ROWS "${MATRIX_FREE_GERSHGORIN_TRACE_TEXT}")
  foreach(bounds_line IN LISTS MATRIX_FREE_GERSHGORIN_BOUNDS_ROWS)
    if((bounds_line MATCHES " strategy=${TRACE_STRATEGY}( |$)") AND (bounds_line MATCHES " fineP=${TRACE_FINE_P}( |$)"))
      string(REGEX REPLACE "^VLASOV_MATRIX_FREE_CHEBYSHEV_BOUNDS *" "" MATRIX_FREE_GERSHGORIN_BOUNDS_TAIL "${bounds_line}")
      string(STRIP "${MATRIX_FREE_GERSHGORIN_BOUNDS_TAIL}" MATRIX_FREE_GERSHGORIN_BOUNDS_TAIL)
      break()
    endif()
  endforeach()
  if(MATRIX_FREE_GERSHGORIN_BOUNDS_TAIL STREQUAL "missing")
    foreach(bounds_line IN LISTS MATRIX_FREE_GERSHGORIN_BOUNDS_ROWS)
      if(NOT bounds_line MATCHES " fineP=${TRACE_FINE_P}( |$)")
        continue()
      endif()
      string(REGEX REPLACE "^VLASOV_MATRIX_FREE_CHEBYSHEV_BOUNDS *" "" MATRIX_FREE_GERSHGORIN_BOUNDS_TAIL "${bounds_line}")
      string(STRIP "${MATRIX_FREE_GERSHGORIN_BOUNDS_TAIL}" MATRIX_FREE_GERSHGORIN_BOUNDS_TAIL)
      break()
    endforeach()
  endif()
  file(READ "${ASSEMBLED_GERSHGORIN_TRACE_FILE}" ASSEMBLED_GERSHGORIN_FILE_TEXT)
  set(ASSEMBLED_GERSHGORIN_TRACE_TEXT "${ASSEMBLED_GERSHGORIN_STDOUT}\n${ASSEMBLED_GERSHGORIN_FILE_TEXT}")
  record_gmg_stage_metrics("${ASSEMBLED_GERSHGORIN_TRACE_TEXT}" "VLASOV_ASSEMBLED_GMG_VCYCLE_STAGE" ASSEMBLED_GERSHGORIN_STAGE_METRIC)
  string(REGEX MATCHALL "VLASOV_ASSEMBLED_GMG_CHEBYSHEV_BOUNDS[^\n]*" ASSEMBLED_GERSHGORIN_BOUNDS_ROWS "${ASSEMBLED_GERSHGORIN_TRACE_TEXT}")
  foreach(bounds_line IN LISTS ASSEMBLED_GERSHGORIN_BOUNDS_ROWS)
    if((bounds_line MATCHES " strategy=${TRACE_STRATEGY}( |$)") AND (bounds_line MATCHES " fineP=${TRACE_FINE_P}( |$)"))
      string(REGEX REPLACE "^VLASOV_ASSEMBLED_GMG_CHEBYSHEV_BOUNDS *" "" ASSEMBLED_GERSHGORIN_BOUNDS_TAIL "${bounds_line}")
      string(STRIP "${ASSEMBLED_GERSHGORIN_BOUNDS_TAIL}" ASSEMBLED_GERSHGORIN_BOUNDS_TAIL)
      break()
    endif()
  endforeach()
  set(GMG_GERSHGORIN_EXPERIMENT_SUMMARY "all_agree")
  set(GMG_GERSHGORIN_EXPERIMENT_ALL_AGREE 1)
  foreach(stage_label IN ITEMS Initial PreSmooth CoarseCorrection PostSmooth)
    string(MAKE_C_IDENTIFIER "${stage_label}" stage_key)
    foreach(metric IN ITEMS rhsInfNorm solutionInfNorm residualInfNorm)
      set(gershgorin_metric_var "ASSEMBLED_GERSHGORIN_STAGE_METRIC_${stage_key}_${metric}")
      set(matrix_free_metric_var "MATRIX_FREE_GERSHGORIN_STAGE_METRIC_${stage_key}_${metric}")
      set(gershgorin_value "missing")
      set(matrix_free_value "missing")
      if(DEFINED ${gershgorin_metric_var})
        set(gershgorin_value "${${gershgorin_metric_var}}")
      endif()
      if(DEFINED ${matrix_free_metric_var})
        set(matrix_free_value "${${matrix_free_metric_var}}")
      endif()
      set(pair_agree 0)
      if((NOT gershgorin_value STREQUAL "missing") AND (NOT matrix_free_value STREQUAL "missing"))
        gmg_relative_agree("${gershgorin_value}" "${matrix_free_value}" pair_agree)
      endif()
      message(STATUS "GMG trace gershgorin-lambda experiment: stage=${stage_label};metric=${metric};assembled_gershgorin=${gershgorin_value};matrix_free=${matrix_free_value};agree=${pair_agree}")
      if((pair_agree EQUAL 0) AND (GMG_GERSHGORIN_EXPERIMENT_SUMMARY STREQUAL "all_agree"))
        set(GMG_GERSHGORIN_EXPERIMENT_ALL_AGREE 0)
        set(GMG_GERSHGORIN_EXPERIMENT_SUMMARY "first_disagreement=stage:${stage_label},metric:${metric},assembled_gershgorin:${gershgorin_value},matrix_free:${matrix_free_value}")
      endif()
    endforeach()
  endforeach()
else()
  set(GMG_GERSHGORIN_EXPERIMENT_SUMMARY "producer_failed_assembled_result_${ASSEMBLED_GERSHGORIN_RESULT}_matrix_free_result_${MATRIX_FREE_GERSHGORIN_RESULT}")
endif()
message(STATUS "GMG trace gershgorin-lambda experiment summary: ${GMG_GERSHGORIN_EXPERIMENT_SUMMARY}; assembled_gershgorin_bounds=${ASSEMBLED_GERSHGORIN_BOUNDS_TAIL}; matrix_free_gershgorin_bounds=${MATRIX_FREE_GERSHGORIN_BOUNDS_TAIL}")

set(GMG_DETERMINISTIC_POWER_EXPERIMENT_SUMMARY "not_run")
set(GMG_DETERMINISTIC_POWER_EXPERIMENT_ALL_AGREE 0)
set(ASSEMBLED_DETERMINISTIC_POWER_BOUNDS_TAIL "missing")
set(MATRIX_FREE_DETERMINISTIC_POWER_BOUNDS_TAIL "missing")
if((ASSEMBLED_DETERMINISTIC_POWER_RESULT EQUAL 0) AND (EXISTS "${ASSEMBLED_DETERMINISTIC_POWER_TRACE_FILE}")
   AND (MATRIX_FREE_DETERMINISTIC_POWER_RESULT EQUAL 0) AND (EXISTS "${MATRIX_FREE_DETERMINISTIC_POWER_TRACE_FILE}"))
  file(READ "${MATRIX_FREE_DETERMINISTIC_POWER_TRACE_FILE}" MATRIX_FREE_DETERMINISTIC_POWER_FILE_TEXT)
  set(MATRIX_FREE_DETERMINISTIC_POWER_TRACE_TEXT "${MATRIX_FREE_DETERMINISTIC_POWER_STDOUT}\n${MATRIX_FREE_DETERMINISTIC_POWER_FILE_TEXT}")
  record_gmg_stage_metrics("${MATRIX_FREE_DETERMINISTIC_POWER_TRACE_TEXT}" "VLASOV_MATRIX_FREE_GMG_STAGE_TRACE" MATRIX_FREE_DETERMINISTIC_POWER_STAGE_METRIC)
  string(REGEX MATCHALL "VLASOV_MATRIX_FREE_CHEBYSHEV_BOUNDS[^\n]*" MATRIX_FREE_DETERMINISTIC_POWER_BOUNDS_ROWS "${MATRIX_FREE_DETERMINISTIC_POWER_TRACE_TEXT}")
  foreach(bounds_line IN LISTS MATRIX_FREE_DETERMINISTIC_POWER_BOUNDS_ROWS)
    if((bounds_line MATCHES " strategy=${TRACE_STRATEGY}( |$)") AND (bounds_line MATCHES " fineP=${TRACE_FINE_P}( |$)"))
      string(REGEX REPLACE "^VLASOV_MATRIX_FREE_CHEBYSHEV_BOUNDS *" "" MATRIX_FREE_DETERMINISTIC_POWER_BOUNDS_TAIL "${bounds_line}")
      string(STRIP "${MATRIX_FREE_DETERMINISTIC_POWER_BOUNDS_TAIL}" MATRIX_FREE_DETERMINISTIC_POWER_BOUNDS_TAIL)
      break()
    endif()
  endforeach()
  if(MATRIX_FREE_DETERMINISTIC_POWER_BOUNDS_TAIL STREQUAL "missing")
    foreach(bounds_line IN LISTS MATRIX_FREE_DETERMINISTIC_POWER_BOUNDS_ROWS)
      if(NOT bounds_line MATCHES " fineP=${TRACE_FINE_P}( |$)")
        continue()
      endif()
      string(REGEX REPLACE "^VLASOV_MATRIX_FREE_CHEBYSHEV_BOUNDS *" "" MATRIX_FREE_DETERMINISTIC_POWER_BOUNDS_TAIL "${bounds_line}")
      string(STRIP "${MATRIX_FREE_DETERMINISTIC_POWER_BOUNDS_TAIL}" MATRIX_FREE_DETERMINISTIC_POWER_BOUNDS_TAIL)
      break()
    endforeach()
  endif()
  file(READ "${ASSEMBLED_DETERMINISTIC_POWER_TRACE_FILE}" ASSEMBLED_DETERMINISTIC_POWER_FILE_TEXT)
  set(ASSEMBLED_DETERMINISTIC_POWER_TRACE_TEXT "${ASSEMBLED_DETERMINISTIC_POWER_STDOUT}\n${ASSEMBLED_DETERMINISTIC_POWER_FILE_TEXT}")
  record_gmg_stage_metrics("${ASSEMBLED_DETERMINISTIC_POWER_TRACE_TEXT}" "VLASOV_ASSEMBLED_GMG_VCYCLE_STAGE" ASSEMBLED_DETERMINISTIC_POWER_STAGE_METRIC)
  string(REGEX MATCHALL "VLASOV_ASSEMBLED_GMG_CHEBYSHEV_BOUNDS[^\n]*" ASSEMBLED_DETERMINISTIC_POWER_BOUNDS_ROWS "${ASSEMBLED_DETERMINISTIC_POWER_TRACE_TEXT}")
  foreach(bounds_line IN LISTS ASSEMBLED_DETERMINISTIC_POWER_BOUNDS_ROWS)
    if((bounds_line MATCHES " strategy=${TRACE_STRATEGY}( |$)") AND (bounds_line MATCHES " fineP=${TRACE_FINE_P}( |$)"))
      string(REGEX REPLACE "^VLASOV_ASSEMBLED_GMG_CHEBYSHEV_BOUNDS *" "" ASSEMBLED_DETERMINISTIC_POWER_BOUNDS_TAIL "${bounds_line}")
      string(STRIP "${ASSEMBLED_DETERMINISTIC_POWER_BOUNDS_TAIL}" ASSEMBLED_DETERMINISTIC_POWER_BOUNDS_TAIL)
      break()
    endif()
  endforeach()
  set(GMG_DETERMINISTIC_POWER_EXPERIMENT_SUMMARY "all_agree")
  set(GMG_DETERMINISTIC_POWER_EXPERIMENT_ALL_AGREE 1)
  foreach(stage_label IN ITEMS Initial PreSmooth CoarseCorrection PostSmooth)
    string(MAKE_C_IDENTIFIER "${stage_label}" stage_key)
    foreach(metric IN ITEMS rhsInfNorm solutionInfNorm residualInfNorm)
      set(deterministic_power_metric_var "ASSEMBLED_DETERMINISTIC_POWER_STAGE_METRIC_${stage_key}_${metric}")
      set(matrix_free_metric_var "MATRIX_FREE_DETERMINISTIC_POWER_STAGE_METRIC_${stage_key}_${metric}")
      set(deterministic_power_value "missing")
      set(matrix_free_value "missing")
      if(DEFINED ${deterministic_power_metric_var})
        set(deterministic_power_value "${${deterministic_power_metric_var}}")
      endif()
      if(DEFINED ${matrix_free_metric_var})
        set(matrix_free_value "${${matrix_free_metric_var}}")
      endif()
      set(pair_agree 0)
      if((NOT deterministic_power_value STREQUAL "missing") AND (NOT matrix_free_value STREQUAL "missing"))
        gmg_relative_agree("${deterministic_power_value}" "${matrix_free_value}" pair_agree)
      endif()
      message(STATUS "GMG trace deterministic-power-lambda experiment: stage=${stage_label};metric=${metric};assembled_deterministic_power=${deterministic_power_value};matrix_free=${matrix_free_value};agree=${pair_agree}")
      if((pair_agree EQUAL 0) AND (GMG_DETERMINISTIC_POWER_EXPERIMENT_SUMMARY STREQUAL "all_agree"))
        set(GMG_DETERMINISTIC_POWER_EXPERIMENT_ALL_AGREE 0)
        set(GMG_DETERMINISTIC_POWER_EXPERIMENT_SUMMARY "first_disagreement=stage:${stage_label},metric:${metric},assembled_deterministic_power:${deterministic_power_value},matrix_free:${matrix_free_value}")
      endif()
    endforeach()
  endforeach()
else()
  set(GMG_DETERMINISTIC_POWER_EXPERIMENT_SUMMARY "producer_failed_assembled_result_${ASSEMBLED_DETERMINISTIC_POWER_RESULT}_matrix_free_result_${MATRIX_FREE_DETERMINISTIC_POWER_RESULT}")
endif()
message(STATUS "GMG trace deterministic-power-lambda experiment summary: ${GMG_DETERMINISTIC_POWER_EXPERIMENT_SUMMARY}; assembled_deterministic_power_bounds=${ASSEMBLED_DETERMINISTIC_POWER_BOUNDS_TAIL}; matrix_free_deterministic_power_bounds=${MATRIX_FREE_DETERMINISTIC_POWER_BOUNDS_TAIL}")

set(ASSEMBLED_DETERMINISTIC_POWER_COARSE_CORRECTION_TRACE_ROW "missing")
if(DEFINED ASSEMBLED_DETERMINISTIC_POWER_TRACE_TEXT)
  string(REGEX MATCHALL "VLASOV_ASSEMBLED_GMG_COARSE_CORRECTION_TRACE[^\n]*" ASSEMBLED_DETERMINISTIC_POWER_COARSE_CORRECTION_ROWS "${ASSEMBLED_DETERMINISTIC_POWER_TRACE_TEXT}")
  foreach(trace_line IN LISTS ASSEMBLED_DETERMINISTIC_POWER_COARSE_CORRECTION_ROWS)
    if((trace_line MATCHES " strategy=${TRACE_STRATEGY}( |$)") AND (trace_line MATCHES " fineP=${TRACE_FINE_P}( |$)"))
      set(ASSEMBLED_DETERMINISTIC_POWER_COARSE_CORRECTION_TRACE_ROW "${trace_line}")
      break()
    endif()
  endforeach()
endif()
set(MATRIX_FREE_DETERMINISTIC_POWER_RESTRICTED_RESIDUAL_ROW "missing")
set(MATRIX_FREE_DETERMINISTIC_POWER_COARSE_CORRECTION_ROW "missing")
set(MATRIX_FREE_DETERMINISTIC_POWER_PROLONGATED_CORRECTION_ROW "missing")
set(MATRIX_FREE_DETERMINISTIC_POWER_RESIDUAL_AFTER_COARSE_ROW "missing")
if(DEFINED MATRIX_FREE_DETERMINISTIC_POWER_TRACE_TEXT)
  string(REGEX MATCHALL "VLASOV_MATRIX_FREE_GMG_STAGE_TRACE[^\n]*" MATRIX_FREE_DETERMINISTIC_POWER_STAGE_ROWS "${MATRIX_FREE_DETERMINISTIC_POWER_TRACE_TEXT}")
  foreach(trace_line IN LISTS MATRIX_FREE_DETERMINISTIC_POWER_STAGE_ROWS)
    if(NOT trace_line MATCHES " strategy=${TRACE_STRATEGY}( |$)")
      continue()
    endif()
    if(NOT trace_line MATCHES " fineP=${TRACE_FINE_P}( |$)")
      continue()
    endif()
    if(trace_line MATCHES " stage=restrictedResidualNorm( |$)")
      set(MATRIX_FREE_DETERMINISTIC_POWER_RESTRICTED_RESIDUAL_ROW "${trace_line}")
    elseif(trace_line MATCHES " stage=coarseCorrectionNorm( |$)")
      set(MATRIX_FREE_DETERMINISTIC_POWER_COARSE_CORRECTION_ROW "${trace_line}")
    elseif(trace_line MATCHES " stage=prolongateCorrection( |$)")
      set(MATRIX_FREE_DETERMINISTIC_POWER_PROLONGATED_CORRECTION_ROW "${trace_line}")
    elseif(trace_line MATCHES " stage=residualAfterCoarseCorrection( |$)")
      set(MATRIX_FREE_DETERMINISTIC_POWER_RESIDUAL_AFTER_COARSE_ROW "${trace_line}")
    endif()
  endforeach()
endif()
message(STATUS "GMG deterministic-power coarse-correction diagnostic rows:\nassembled: ${ASSEMBLED_DETERMINISTIC_POWER_COARSE_CORRECTION_TRACE_ROW}\nmatrix_free_restricted: ${MATRIX_FREE_DETERMINISTIC_POWER_RESTRICTED_RESIDUAL_ROW}\nmatrix_free_coarse_solution: ${MATRIX_FREE_DETERMINISTIC_POWER_COARSE_CORRECTION_ROW}\nmatrix_free_prolongated: ${MATRIX_FREE_DETERMINISTIC_POWER_PROLONGATED_CORRECTION_ROW}\nmatrix_free_residual_after_coarse: ${MATRIX_FREE_DETERMINISTIC_POWER_RESIDUAL_AFTER_COARSE_ROW}")

function(gmg_extract_trace_field TRACE_ROW FIELD_NAME OUT_VALUE)
  if(TRACE_ROW MATCHES " ${FIELD_NAME}=([^ ]+)")
    set(${OUT_VALUE} "${CMAKE_MATCH_1}" PARENT_SCOPE)
  else()
    set(${OUT_VALUE} "missing" PARENT_SCOPE)
  endif()
endfunction()

gmg_extract_trace_field("${ASSEMBLED_DETERMINISTIC_POWER_COARSE_CORRECTION_TRACE_ROW}" "restrictedResidualChecksum" ASSEMBLED_DETERMINISTIC_POWER_RESTRICTED_CHECKSUM)
gmg_extract_trace_field("${ASSEMBLED_DETERMINISTIC_POWER_COARSE_CORRECTION_TRACE_ROW}" "coarseCorrectionChecksum" ASSEMBLED_DETERMINISTIC_POWER_COARSE_CHECKSUM)
gmg_extract_trace_field("${ASSEMBLED_DETERMINISTIC_POWER_COARSE_CORRECTION_TRACE_ROW}" "prolongatedCorrectionChecksum" ASSEMBLED_DETERMINISTIC_POWER_PROLONGATED_CHECKSUM)
gmg_extract_trace_field("${ASSEMBLED_DETERMINISTIC_POWER_COARSE_CORRECTION_TRACE_ROW}" "residualAfterCorrectionChecksum" ASSEMBLED_DETERMINISTIC_POWER_RESIDUAL_CHECKSUM)
gmg_extract_trace_field("${MATRIX_FREE_DETERMINISTIC_POWER_RESTRICTED_RESIDUAL_ROW}" "solutionChecksum" MATRIX_FREE_DETERMINISTIC_POWER_RESTRICTED_CHECKSUM)
gmg_extract_trace_field("${MATRIX_FREE_DETERMINISTIC_POWER_COARSE_CORRECTION_ROW}" "solutionChecksum" MATRIX_FREE_DETERMINISTIC_POWER_COARSE_CHECKSUM)
gmg_extract_trace_field("${MATRIX_FREE_DETERMINISTIC_POWER_PROLONGATED_CORRECTION_ROW}" "solutionChecksum" MATRIX_FREE_DETERMINISTIC_POWER_PROLONGATED_CHECKSUM)
gmg_extract_trace_field("${MATRIX_FREE_DETERMINISTIC_POWER_RESIDUAL_AFTER_COARSE_ROW}" "residualChecksum" MATRIX_FREE_DETERMINISTIC_POWER_RESIDUAL_CHECKSUM)
message(STATUS "GMG deterministic-power coarse-correction checksum fingerprints: restricted assembled=${ASSEMBLED_DETERMINISTIC_POWER_RESTRICTED_CHECKSUM} matrix_free=${MATRIX_FREE_DETERMINISTIC_POWER_RESTRICTED_CHECKSUM}; coarse assembled=${ASSEMBLED_DETERMINISTIC_POWER_COARSE_CHECKSUM} matrix_free=${MATRIX_FREE_DETERMINISTIC_POWER_COARSE_CHECKSUM}; prolongated assembled=${ASSEMBLED_DETERMINISTIC_POWER_PROLONGATED_CHECKSUM} matrix_free=${MATRIX_FREE_DETERMINISTIC_POWER_PROLONGATED_CHECKSUM}; residual assembled=${ASSEMBLED_DETERMINISTIC_POWER_RESIDUAL_CHECKSUM} matrix_free=${MATRIX_FREE_DETERMINISTIC_POWER_RESIDUAL_CHECKSUM}")

# Main auto-vs-auto pairs: tolerance-based agreement using the same 1e-8 relative helper.
set(GMG_AUTO_LAMBDA_COMPARISON_SUMMARY "all_agree")
set(GMG_AUTO_LAMBDA_ALL_AGREE 1)
set(GMG_AUTO_LAMBDA_RELATIVE_DISAGREEMENT_COUNT 0)
set(GMG_AUTO_LAMBDA_FIRST_RELATIVE_DISAGREEMENT "none")
set(GMG_AUTO_LAMBDA_FIRST_RELATIVE_DISAGREEMENT_ASSEMBLED_ROW "none")
set(GMG_AUTO_LAMBDA_FIRST_RELATIVE_DISAGREEMENT_MATRIX_FREE_ROW "none")
foreach(stage_label IN LISTS COMMON_GMG_STAGE_LABELS)
  string(MAKE_C_IDENTIFIER "${stage_label}" stage_key)
  foreach(metric IN ITEMS rhsInfNorm solutionInfNorm residualInfNorm)
    set(auto_assembled_value "${ASSEMBLED_STAGE_METRIC_${stage_key}_${metric}}")
    set(auto_matrix_free_value "${MATRIX_FREE_STAGE_METRIC_${stage_key}_${metric}}")
    set(auto_pair_agree 0)
    gmg_relative_agree("${auto_assembled_value}" "${auto_matrix_free_value}" auto_pair_agree)
    message(STATUS "GMG trace auto-lambda comparison: stage=${stage_label};metric=${metric};assembled=${auto_assembled_value};matrix_free=${auto_matrix_free_value};agree=${auto_pair_agree}")
    if(auto_pair_agree EQUAL 0)
      math(EXPR GMG_AUTO_LAMBDA_RELATIVE_DISAGREEMENT_COUNT "${GMG_AUTO_LAMBDA_RELATIVE_DISAGREEMENT_COUNT} + 1")
      set(GMG_AUTO_LAMBDA_ALL_AGREE 0)
      if(GMG_AUTO_LAMBDA_FIRST_RELATIVE_DISAGREEMENT STREQUAL "none")
        set(GMG_AUTO_LAMBDA_FIRST_RELATIVE_DISAGREEMENT "stage=${stage_label};metric=${metric};assembled=${auto_assembled_value};matrix_free=${auto_matrix_free_value}")
        set(GMG_AUTO_LAMBDA_COMPARISON_SUMMARY "first_relative_disagreement=${GMG_AUTO_LAMBDA_FIRST_RELATIVE_DISAGREEMENT}")
        set(auto_assembled_raw_row_var "ASSEMBLED_STAGE_METRIC_${stage_key}_rawRow")
        set(auto_matrix_free_raw_row_var "MATRIX_FREE_STAGE_METRIC_${stage_key}_rawRow")
        if(DEFINED ${auto_assembled_raw_row_var})
          set(GMG_AUTO_LAMBDA_FIRST_RELATIVE_DISAGREEMENT_ASSEMBLED_ROW "${${auto_assembled_raw_row_var}}")
        endif()
        if(DEFINED ${auto_matrix_free_raw_row_var})
          set(GMG_AUTO_LAMBDA_FIRST_RELATIVE_DISAGREEMENT_MATRIX_FREE_ROW "${${auto_matrix_free_raw_row_var}}")
        endif()
      endif()
    endif()
  endforeach()
endforeach()
message(STATUS "GMG trace auto-lambda comparison summary: ${GMG_AUTO_LAMBDA_COMPARISON_SUMMARY}; relative_disagreements=${GMG_AUTO_LAMBDA_RELATIVE_DISAGREEMENT_COUNT}")

if((GMG_AUTO_LAMBDA_RELATIVE_DISAGREEMENT_COUNT GREATER 0) AND (NOT GMG_GERSHGORIN_EXPERIMENT_ALL_AGREE EQUAL 1))
  message(STATUS
    "GMG trace first relative disagreement retained as readiness diagnostic: strategy=${TRACE_STRATEGY}; relative_tolerance=1e-8; ${GMG_AUTO_LAMBDA_FIRST_RELATIVE_DISAGREEMENT}; relative_disagreements=${GMG_AUTO_LAMBDA_RELATIVE_DISAGREEMENT_COUNT}; exact_value_mismatches=${GMG_STAGE_NUMERICAL_MISMATCH_COUNT}; common_stage_labels=${COMMON_GMG_STAGE_LABELS}; assembled_bounds=${ASSEMBLED_CHEBYSHEV_BOUNDS_TAIL}; matrix_free_bounds=${MATRIX_FREE_CHEBYSHEV_BOUNDS_TAIL}; gershgorin_experiment=${GMG_GERSHGORIN_EXPERIMENT_SUMMARY}; assembled_gershgorin_bounds=${ASSEMBLED_GERSHGORIN_BOUNDS_TAIL}; matrix_free_gershgorin_bounds=${MATRIX_FREE_GERSHGORIN_BOUNDS_TAIL}; auto_lambda_comparison=${GMG_AUTO_LAMBDA_COMPARISON_SUMMARY}")
elseif(GMG_STAGE_NUMERICAL_MISMATCH_COUNT GREATER 0)
  if(GMG_AUTO_LAMBDA_ALL_AGREE EQUAL 1)
    message(STATUS "GMG trace stages agree within tolerance: strategy=${TRACE_STRATEGY}; relative_tolerance=1e-8; exact_value_mismatches=${GMG_STAGE_NUMERICAL_MISMATCH_COUNT}; first_exact_value_mismatch=${GMG_STAGE_NUMERICAL_FIRST_MISMATCH}; auto_lambda_comparison=${GMG_AUTO_LAMBDA_COMPARISON_SUMMARY}; gershgorin_experiment=${GMG_GERSHGORIN_EXPERIMENT_SUMMARY}")
  elseif(GMG_GERSHGORIN_EXPERIMENT_ALL_AGREE EQUAL 1)
    message(STATUS "GMG trace stages agree in deterministic gershgorin-lambda direct-coarse experiment: strategy=${TRACE_STRATEGY}; relative_tolerance=1e-8; exact_value_mismatches=${GMG_STAGE_NUMERICAL_MISMATCH_COUNT}; first_exact_value_mismatch=${GMG_STAGE_NUMERICAL_FIRST_MISMATCH}; auto_lambda_comparison=${GMG_AUTO_LAMBDA_COMPARISON_SUMMARY}; gershgorin_experiment=${GMG_GERSHGORIN_EXPERIMENT_SUMMARY}; assembled_gershgorin_bounds=${ASSEMBLED_GERSHGORIN_BOUNDS_TAIL}; matrix_free_gershgorin_bounds=${MATRIX_FREE_GERSHGORIN_BOUNDS_TAIL}")
  endif()
endif()

set(GMG_TRACE_COMPARISON_SUMMARY
"trace_fine_p=${TRACE_FINE_P}
strategy=${TRACE_STRATEGY}
matrix_free_stage_rows=${MATRIX_FREE_TRACE_ROW_COUNT}
assembled_vcycle_stage_rows=${ASSEMBLED_VCYCLE_STAGE_ROW_COUNT}
required_common_stage_labels=Initial;PreSmooth;CoarseCorrection;PostSmooth
common_stage_labels=${COMMON_GMG_STAGE_LABELS}
compared_stage_metrics=${COMPARED_GMG_STAGE_METRIC_COUNT}
relative_tolerance=1e-8
relative_disagreements=${GMG_AUTO_LAMBDA_RELATIVE_DISAGREEMENT_COUNT}
first_relative_disagreement=${GMG_AUTO_LAMBDA_FIRST_RELATIVE_DISAGREEMENT}
first_relative_disagreement_assembled_row=${GMG_AUTO_LAMBDA_FIRST_RELATIVE_DISAGREEMENT_ASSEMBLED_ROW}
first_relative_disagreement_matrix_free_row=${GMG_AUTO_LAMBDA_FIRST_RELATIVE_DISAGREEMENT_MATRIX_FREE_ROW}
exact_value_mismatches=${GMG_STAGE_NUMERICAL_MISMATCH_COUNT}
first_exact_value_mismatch=${GMG_STAGE_NUMERICAL_FIRST_MISMATCH}
assembled_chebyshev_bounds=${ASSEMBLED_CHEBYSHEV_BOUNDS_TAIL}
matrix_free_chebyshev_bounds=${MATRIX_FREE_CHEBYSHEV_BOUNDS_TAIL}
gershgorin_experiment=${GMG_GERSHGORIN_EXPERIMENT_SUMMARY}
assembled_gershgorin_bounds=${ASSEMBLED_GERSHGORIN_BOUNDS_TAIL}
matrix_free_gershgorin_bounds=${MATRIX_FREE_GERSHGORIN_BOUNDS_TAIL}
deterministic_power_experiment=${GMG_DETERMINISTIC_POWER_EXPERIMENT_SUMMARY}
assembled_deterministic_power_bounds=${ASSEMBLED_DETERMINISTIC_POWER_BOUNDS_TAIL}
matrix_free_deterministic_power_bounds=${MATRIX_FREE_DETERMINISTIC_POWER_BOUNDS_TAIL}
assembled_deterministic_power_coarse_correction_row=${ASSEMBLED_DETERMINISTIC_POWER_COARSE_CORRECTION_TRACE_ROW}
matrix_free_deterministic_power_restricted_residual_row=${MATRIX_FREE_DETERMINISTIC_POWER_RESTRICTED_RESIDUAL_ROW}
matrix_free_deterministic_power_coarse_correction_row=${MATRIX_FREE_DETERMINISTIC_POWER_COARSE_CORRECTION_ROW}
matrix_free_deterministic_power_prolongated_correction_row=${MATRIX_FREE_DETERMINISTIC_POWER_PROLONGATED_CORRECTION_ROW}
matrix_free_deterministic_power_residual_after_coarse_row=${MATRIX_FREE_DETERMINISTIC_POWER_RESIDUAL_AFTER_COARSE_ROW}
auto_lambda_comparison=${GMG_AUTO_LAMBDA_COMPARISON_SUMMARY}
matrix_free_trace_file=${MATRIX_FREE_TRACE_FILE}
assembled_trace_file=${ASSEMBLED_TRACE_FILE}
rhs_inputs=assembled:RHS::addTerm(Function::constant(1.0)*w),basis=Intrepid2HierarchicalHGRAD,massWeight=1,stiffness=VlasovGraphLfLw;matrix_free:--inverseRhs=constant-test-function,--basisMode=hgrad,--operator=vlasov-graph-norm-1d1t2v,--massWeight=1,--stiffnessWeight=1
")
file(WRITE "${STAGE_COMPARISON_FILE}" "${GMG_TRACE_COMPARISON_SUMMARY}")
message(STATUS "GMG trace comparison diagnostics written: ${STAGE_COMPARISON_FILE}")
message(STATUS "GMG trace p=${TRACE_FINE_P} stage comparison diagnostics:\n${GMG_TRACE_COMPARISON_SUMMARY}")
if(NOT GMG_AUTO_LAMBDA_ALL_AGREE EQUAL 1)
  message(STATUS
    "GMG trace first relative disagreement rows:\nassembled: ${GMG_AUTO_LAMBDA_FIRST_RELATIVE_DISAGREEMENT_ASSEMBLED_ROW}\nmatrix_free: ${GMG_AUTO_LAMBDA_FIRST_RELATIVE_DISAGREEMENT_MATRIX_FREE_ROW}")
endif()
