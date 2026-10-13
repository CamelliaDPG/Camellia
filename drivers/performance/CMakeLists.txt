project(performance)

add_executable(StokesAssemblyPerformanceTest "StokesAssemblyPerformanceTest.cpp")
target_link_libraries(StokesAssemblyPerformanceTest Camellia)

add_executable(TensorPoissonPAMatrixDriver "TensorPoissonPAMatrixDriver.cpp")
add_executable(VlasovGramGMGStudyDriver "VlasovGramGMGStudyDriver.cpp")
add_executable(VlasovGramMatrixFreeStudyDriver "VlasovGramMatrixFreeStudyDriver.cpp")
target_compile_definitions(VlasovGramMatrixFreeStudyDriver PRIVATE VLASOV_MF_DRIVER_CURRENT_SOURCE=1)
set_target_properties(VlasovGramMatrixFreeStudyDriver PROPERTIES EXCLUDE_FROM_ALL FALSE)
target_link_libraries(VlasovGramGMGStudyDriver Camellia ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
set(TENSOR_INTREPID2_SOURCE_DIR
  "/Users/nvrober/fork-Trilinos/packages/intrepid2/src"
  CACHE PATH "Intrepid2 source directory from the PAMatrix-enabled Trilinos fork")
set(TENSOR_INTREPID2_PAMATRIX_INCLUDE_DIR
  "${TENSOR_INTREPID2_SOURCE_DIR}/Discretization/Integration"
  CACHE PATH "Directory containing experimental Intrepid2 PAMatrix headers")
target_include_directories(TensorPoissonPAMatrixDriver BEFORE PRIVATE
  ${TENSOR_INTREPID2_PAMATRIX_INCLUDE_DIR}
  ${TENSOR_INTREPID2_SOURCE_DIR}/Discretization/Basis
  ${TENSOR_INTREPID2_SOURCE_DIR}/Shared
  ${TENSOR_INTREPID2_SOURCE_DIR}/Cell
  ${TENSOR_INTREPID2_SOURCE_DIR}/../assembly-examples
)
target_link_libraries(TensorPoissonPAMatrixDriver Camellia ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})

add_test(NAME TensorPoissonPAMatrixDriverSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=2
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=mass-plus-stiffness
    --basisMode=hgrad
    --mode=compare
    --numApply=2
    --maxAssembledEntries=10000)

add_test(NAME VlasovGramMatrixFreeStudyDriverSmoke
  COMMAND $<TARGET_FILE:VlasovGramMatrixFreeStudyDriver>
    --fineH1Order=1
    --quick
    --maxIterations=5
    --outputFile=vlasov-gram-matrix-free-study-smoke.txt)

add_test(NAME TensorVlasovGramMatrixFreePMultigridP2DiagnosticSmoke
  COMMAND $<TARGET_FILE:VlasovGramMatrixFreeStudyDriver>
    --fineH1Order=2
    --quick
    --maxIterations=200
    --outputFile=${CMAKE_BINARY_DIR}/vlasov-gram-matrix-free-study.txt)

add_test(NAME TensorVlasovGramMatrixFreePMultigridP4DiagnosticSmoke
  COMMAND $<TARGET_FILE:VlasovGramMatrixFreeStudyDriver>
    --fineH1Order=4
    --quick
    --maxIterations=200
    --keepChildOutput=1
    --outputFile=${CMAKE_BINARY_DIR}/vlasov-gram-matrix-free-p4-diagnostic-smoke.txt)
set_tests_properties(TensorVlasovGramMatrixFreePMultigridP4DiagnosticSmoke PROPERTIES
  PASS_REGULAR_EXPRESSION "P4_PMG_JACOBI_CHEBYSHEV_COARSE_RESULT.*pMultigridLevels=3.*pLevelSchedule=4->2->1"
  FAIL_REGULAR_EXPRESSION "P4_PMG_JACOBI_CHEBYSHEV_COARSE_RESULT.*status=converged iterations=1 ")

add_test(NAME TensorVlasovGramMatrixFreeGMGStageTraceSmoke
  COMMAND $<TARGET_FILE:VlasovGramMatrixFreeStudyDriver>
    --fineH1Order=2
    --quick
    --maxIterations=20
    --matrixFreeGMGStageTrace=1
    --keepChildOutput=1
    --maxAssembledEntries=10000
    --outputFile=${CMAKE_BINARY_DIR}/vlasov-gram-matrix-free-stage-trace-smoke.txt)
set_tests_properties(TensorVlasovGramMatrixFreeGMGStageTraceSmoke PROPERTIES
  PASS_REGULAR_EXPRESSION "P2_PMG_FINEWT_RESULT.*matrixFreeGMGStageTrace=1 matrixFreeGMGStageTraceEntries=[1-9][0-9]*")
add_test(NAME TensorVlasovGramAssembledGMGP2TraceSmoke
  COMMAND $<TARGET_FILE:VlasovGramGMGStudyDriver>
    --fineH1Order=2
    --quick
    --maxIterations=5
    --outputFile=${CMAKE_BINARY_DIR}/vlasov-gram-assembled-gmg-p2-trace-smoke.txt)
set_tests_properties(TensorVlasovGramAssembledGMGP2TraceSmoke PROPERTIES
  ENVIRONMENT "VLASOV_ASSEMBLED_GMG_EMIT_STAGE_TRACE=1"
  # Temporary relaxation: keep the trace path enabled, but do not treat the
  # currently-missing real GMGOperator stage row as a Tensor-suite failure.
  # Do not add sentinel stage rows here; the readiness comparison below requires
  # real VLASOV_ASSEMBLED_GMG_VCYCLE_STAGE rows when both trace producers run.
  PASS_REGULAR_EXPRESSION "Wrote Vlasov Gram GMG study results")

add_test(NAME TensorVlasovGramGMGTraceReadinessCompare
  COMMAND ${CMAKE_COMMAND}
    -DMATRIX_FREE_DRIVER=$<TARGET_FILE:VlasovGramMatrixFreeStudyDriver>
    -DASSEMBLED_DRIVER=$<TARGET_FILE:VlasovGramGMGStudyDriver>
    -DTRACE_WORK_DIR=${CMAKE_BINARY_DIR}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/CompareVlasovGMGTraceReadinessStandalone5.cmake)

add_test(NAME TensorH1GramPAMatrixDriverSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=3
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=compare
    --numApply=2
    --massWeight=2.0
    --stiffnessWeight=0.5
    --maxAssembledEntries=10000)

add_test(NAME TensorH1GramPAMatrixMatrixFreeGMGStageTraceSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=3
    --polyOrder=2
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=gmres-pmultigrid-chebyshev
    --matrixFreeGMGStageTrace=1
    --numInverseIterations=20)
set_tests_properties(TensorH1GramPAMatrixMatrixFreeGMGStageTraceSmoke PROPERTIES
  PASS_REGULAR_EXPRESSION "matrixFreeGMGStageTrace=1 matrixFreeGMGStageTraceEntries=[1-9][0-9]*")

add_test(NAME TensorH1GramPAMatrixAssemblyCompareSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=3
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=assembly-compare
    --checkAssemblyAgreement=1
    --massWeight=2.0
    --stiffnessWeight=0.5
    --maxAssembledEntries=10000)

add_test(NAME TensorVlasovPAMatrixBridgeCompareSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=2
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=vlasov-dt
    --basisMode=hgrad
    --mode=compare
    --numApply=2
    --maxAssembledEntries=10000)

add_test(NAME TensorVlasovCoordinatePAMatrixBridgeCompareSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=2
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=vlasov-coordinate
    --basisMode=hgrad
    --mode=compare
    --numApply=2
    --maxAssembledEntries=10000)

add_test(NAME TensorVlasovLinearCoordinatePAMatrixBridgeCompareSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=3
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=vlasov-linear-x0-x1-d2
    --basisMode=hgrad
    --mode=compare
    --numApply=2
    --maxAssembledEntries=10000)

add_test(NAME TensorVlasovCoordinateX1D3PAMatrixBridgeCompareSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=4
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=vlasov-coordinate-x1-d3
    --basisMode=hgrad
    --mode=compare
    --numApply=2
    --maxAssembledEntries=10000)

add_test(NAME TensorH1GramPAMatrixSynthetic4DSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=4
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=synthetic
    --mode=apply
    --numApply=1
    --massWeight=2.0
    --stiffnessWeight=0.5
    --maxAssembledEntries=1)

add_test(NAME TensorH1GramPAMatrixInverseDiagnosticSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=2
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --numInverseIterations=5
    --massWeight=2.0
    --stiffnessWeight=0.5
    --maxAssembledEntries=10000)

add_test(NAME TensorH1GramPAMatrixChebyshevDiagnosticSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=2
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=chebyshev
    --numInverseIterations=5
    --inverseEigenRatio=30.0
    --massWeight=2.0
    --stiffnessWeight=0.5
    --maxAssembledEntries=10000)

add_test(NAME TensorH1GramPAMatrixPowerSpectralDiagnosticSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=2
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=chebyshev
    --inverseSpectralEstimate=power
    --inverseSpectralEstimateIterations=6
    --inversePowerSafetyFactor=1.10
    --inverseDiagnosticRequireAssembledReference=0
    --compareTpetraPowerEstimate=1
    --numInverseIterations=5
    --inverseEigenRatio=30.0
    --inverseLambdaMin=0.01
    --massWeight=2.0
    --stiffnessWeight=0.5
    --maxAssembledEntries=1)

add_test(NAME TensorH1GramPAMatrixUserSpectralNoReferenceSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=2
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=richardson
    --inverseSpectralEstimate=user
    --inverseDiagnosticRequireAssembledReference=0
    --inverseInitialGuess=zero
    --numInverseIterations=5
    --inverseTolerance=1e-6
    --inverseLambdaMin=0.01
    --inverseLambdaMax=10.0
    --massWeight=2.0
    --stiffnessWeight=0.5
    --maxAssembledEntries=1)

add_test(NAME TensorH1GramPAMatrixTpetraOperatorSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=2
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=tpetra-operator-compare
    --numApply=3
    --massWeight=2.0
    --stiffnessWeight=0.5
    --maxAssembledEntries=10000)

add_test(NAME TensorH1GramPAMatrixTpetraRowMatrixSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=2
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=tpetra-rowmatrix-compare
    --numApply=3
    --massWeight=2.0
    --stiffnessWeight=0.5
    --maxAssembledEntries=10000)

add_test(NAME TensorVlasovGraphNormCamelliaP1P2MatrixCompareSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=4
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=vlasov-graph-norm-1d1t2v
    --basisMode=hgrad
    --mode=camellia-vlasov-matrix-compare
    --massWeight=1.0
    --stiffnessWeight=1.0
    --camelliaMatrixCompareFailOnMismatch=1
    --maxAssembledEntries=1000000)

add_test(NAME TensorH1GramPAMatrixTpetraRowMatrixGuardSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=2
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=tpetra-rowmatrix-compare
    --numApply=3
    --massWeight=2.0
    --stiffnessWeight=0.5
    --maxAssembledEntries=1)

add_test(NAME TensorH1GramPAMatrixIfpack2ChebyshevSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=2
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=chebyshev
    --ifpack2ChebyshevSmoke=1
    --inverseInitialGuess=zero
    --numInverseIterations=5
    --inverseEigenRatio=30.0
    --massWeight=2.0
    --stiffnessWeight=0.5
    --maxAssembledEntries=10000)

add_test(NAME TensorH1GramPAMatrixIfpack2PowerNoReferenceSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=2
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=chebyshev
    --inverseSpectralEstimate=power
    --inverseSpectralEstimateIterations=6
    --inverseDiagnosticRequireAssembledReference=0
    --compareTpetraPowerEstimate=1
    --compareTpetraRowMatrixPowerEstimate=1
    --ifpack2ChebyshevSmoke=1
    --inverseInitialGuess=zero
    --numInverseIterations=5
    --inverseEigenRatio=30.0
    --massWeight=2.0
    --stiffnessWeight=0.5
    --maxAssembledEntries=1)

add_test(NAME TensorH1GramPAMatrixLowerSpectralDiagnosticSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=2
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=chebyshev
    --inverseLowerSpectralEstimate=assembled-inverse-iteration
    --inverseLowerSpectralEstimateIterations=4
    --inverseLowerPowerSafetyFactor=0.95
    --ifpack2ChebyshevSmoke=1
    --inverseInitialGuess=zero
    --numInverseIterations=5
    --massWeight=2.0
    --stiffnessWeight=0.5
    --maxAssembledEntries=10000)

add_test(NAME TensorVlasovGraphNormPAMatrixChebyshevPowerSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=4
    --polyOrder=1
    --cellsPerDim=1
    --operator=vlasov-graph-norm-1d1t2v
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=chebyshev
    --inverseSpectralEstimate=power
    --inverseSpectralEstimateIterations=6
    --inverseDiagnosticRequireAssembledReference=0
    --compareTpetraPowerEstimate=1
    --compareTpetraRowMatrixPowerEstimate=1
    --ifpack2ChebyshevSmoke=1
    --inverseInitialGuess=zero
    --numInverseIterations=5
    --inverseEigenRatio=30.0
    --massWeight=2.0
    --stiffnessWeight=0.5
    --maxAssembledEntries=1)

add_test(NAME TensorVlasovGraphNormPAMatrixCellPowerChebyshevSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=4
    --polyOrder=1
    --cellsPerDim=2
    --operator=vlasov-graph-norm-1d1t2v
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=chebyshev
    --inverseSpectralEstimate=cell-power
    --inverseSpectralEstimateIterations=5
    --inverseDiagnosticRequireAssembledReference=0
    --inverseInitialGuess=zero
    --numInverseIterations=5
    --inverseEigenRatio=30.0
    --massWeight=2.0
    --stiffnessWeight=0.5
    --maxAssembledEntries=1)

add_test(NAME TensorVlasovGraphNormSolutionCoeffPAMatrixChebyshevSmoke
  COMMAND TensorPoissonPAMatrixDriver
    --dim=4
    --polyOrder=1
    --cellsPerDim=1
    --operator=vlasov-graph-norm-solution-coeff-1d1t2v
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=chebyshev
    --inverseSpectralEstimate=power
    --inverseSpectralEstimateIterations=5
    --inverseDiagnosticRequireAssembledReference=0
    --inverseInitialGuess=zero
    --numInverseIterations=5
    --inverseEigenRatio=30.0
    --massWeight=2.0
    --stiffnessWeight=0.5
    --maxAssembledEntries=1)

add_custom_target(runTensorPoissonPAMatrixSmoke
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=2
    --polyOrder=1:3
    --cellsPerDim=1
    --operator=mass-plus-stiffness
    --basisMode=hgrad
    --mode=compare
    --numApply=5
    --maxAssembledEntries=50000
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running small direct-PAMatrix tensor Poisson apply/compare smoke test"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixSmoke
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=3
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=compare
    --numApply=3
    --maxAssembledEntries=10000
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running small direct-PAMatrix H1 Gram assembly/apply smoke test"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixTpetraOperatorSmoke
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=3
    --polyOrder=1:3
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=tpetra-operator-compare
    --massWeight=1.0
    --stiffnessWeight=1.0
    --maxAssembledEntries=10000
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running local-flat-map Tpetra operator comparison for direct-PAMatrix H1 Gram apply"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixTpetraRowMatrixSmoke
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=3
    --polyOrder=1:3
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=tpetra-rowmatrix-compare
    --massWeight=1.0
    --stiffnessWeight=1.0
    --maxAssembledEntries=10000
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running local-flat-map Tpetra RowMatrix comparison for direct-PAMatrix H1 Gram apply/diagonal"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixInverseDiagnosticSweep
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=3
    --polyOrder=2:5
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --numInverseIterations=20
    --massWeight=1.0
    --stiffnessWeight=1.0
    --maxAssembledEntries=20000000
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running opt-in direct-PAMatrix H1 Gram local matrix-free inverse diagnostic p-sweep"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixChebyshevDiagnosticSweep
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=3
    --polyOrder=2:5
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=chebyshev
    --numInverseIterations=20
    --inverseEigenRatio=30.0
    --massWeight=1.0
    --stiffnessWeight=1.0
    --maxAssembledEntries=20000000
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running opt-in direct-PAMatrix H1 Gram local Chebyshev inverse diagnostic p-sweep"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixIfpack2ChebyshevSmoke
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=3
    --polyOrder=1:3
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=chebyshev
    --ifpack2ChebyshevSmoke=1
    --inverseInitialGuess=zero
    --numInverseIterations=5
    --inverseEigenRatio=30.0
    --massWeight=1.0
    --stiffnessWeight=1.0
    --maxAssembledEntries=10000
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running opt-in Ifpack2 Chebyshev smoke on the local-flat PAMatrix RowMatrix adapter"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixIfpack2PowerNoReferenceSmoke
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=3
    --polyOrder=1:3
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=chebyshev
    --inverseSpectralEstimate=power
    --inverseSpectralEstimateIterations=8
    --inverseDiagnosticRequireAssembledReference=0
    --compareTpetraPowerEstimate=1
    --compareTpetraRowMatrixPowerEstimate=1
    --ifpack2ChebyshevSmoke=1
    --inverseInitialGuess=zero
    --numInverseIterations=5
    --inverseEigenRatio=30.0
    --massWeight=1.0
    --stiffnessWeight=1.0
    --maxAssembledEntries=1
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running opt-in local-flat Ifpack2 Chebyshev smoke with matrix-free power bounds and no assembled reference"
  VERBATIM)

add_dependencies(runTensorH1GramPAMatrixInverseDiagnosticSweep
  runTensorH1GramPAMatrixTpetraRowMatrixSmoke
  runTensorH1GramPAMatrixIfpack2ChebyshevSmoke
  runTensorH1GramPAMatrixIfpack2PowerNoReferenceSmoke)

add_custom_target(runTensorH1GramPAMatrixPowerSpectralChebyshevSweep
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=3
    --polyOrder=2:5
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=chebyshev
    --inverseSpectralEstimate=power
    --inverseSpectralEstimateIterations=10
    --inversePowerSafetyFactor=1.05
    --numInverseIterations=20
    --inverseEigenRatio=30.0
    --massWeight=1.0
    --stiffnessWeight=1.0
    --maxAssembledEntries=20000000
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running opt-in direct-PAMatrix H1 Gram Chebyshev diagnostic with matrix-free power spectral estimate"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixNoReferenceChebyshevSweep
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=3
    --polyOrder=2:5
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=chebyshev
    --inverseSpectralEstimate=power
    --inverseSpectralEstimateIterations=10
    --inverseDiagnosticRequireAssembledReference=0
    --numInverseIterations=20
    --inverseEigenRatio=30.0
    --massWeight=1.0
    --stiffnessWeight=1.0
    --maxAssembledEntries=1
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running opt-in direct-PAMatrix H1 Gram Chebyshev diagnostic without assembled reference/oracle blocks"
  VERBATIM)

add_custom_target(runTensorVlasovGraphNormPAMatrixChebyshevViabilitySweep ALL
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=4
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=vlasov-graph-norm-1d1t2v
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=chebyshev
    --inverseSpectralEstimate=power
    --inverseSpectralEstimateIterations=10
    --inverseDiagnosticRequireAssembledReference=0
    --compareTpetraPowerEstimate=1
    --compareTpetraRowMatrixPowerEstimate=1
    --ifpack2ChebyshevSmoke=1
    --inverseInitialGuess=zero
    --numInverseIterations=10
    --inverseMaxResidualReduction=-1
    --inverseEigenRatio=1.1
    --massWeight=1e6
    --stiffnessWeight=0.5
    --maxAssembledEntries=1
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=4
    --polyOrder=1:2
    --cellsPerDim=2
    --operator=vlasov-graph-norm-1d1t2v
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=chebyshev
    --inverseSpectralEstimate=cell-power
    --inverseSpectralEstimateIterations=8
    --inverseDiagnosticRequireAssembledReference=0
    --inverseInitialGuess=zero
    --numInverseIterations=10
    --inverseMaxResidualReduction=-1
    --inverseEigenRatio=1.1
    --massWeight=1e6
    --stiffnessWeight=0.5
    --maxAssembledEntries=1
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=4
    --polyOrder=1:2
    --cellsPerDim=2
    --operator=vlasov-graph-norm-1d1t2v
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=chebyshev
    --inverseSpectralEstimate=cell-power
    --inverseSpectralEstimateIterations=8
    --inverseDiagnosticRequireAssembledReference=0
    --inverseInitialGuess=zero
    --numInverseIterations=20
    --inverseMaxResidualReduction=-1
    --inverseEigenRatio=1.1
    --massWeight=1e6
    --stiffnessWeight=0.5
    --maxAssembledEntries=1
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running opt-in Vlasov graph-norm Chebyshev G-inverse viability sweep with global and per-cell matrix-free spectral calibration"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixLowerSpectralChebyshevSweep
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=3
    --polyOrder=2:5
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=chebyshev
    --inverseLowerSpectralEstimate=assembled-inverse-iteration
    --inverseLowerSpectralEstimateIterations=8
    --inverseLowerPowerSafetyFactor=0.95
    --ifpack2ChebyshevSmoke=1
    --inverseInitialGuess=zero
    --numInverseIterations=20
    --massWeight=1.0
    --stiffnessWeight=1.0
    --maxAssembledEntries=20000000
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running opt-in direct-PAMatrix H1 Gram Chebyshev diagnostic with assembled inverse-iteration lower spectral estimate and Ifpack2 recurrence smoke"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixUserSpectralNoReferenceSweep
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=3
    --polyOrder=2:5
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=inverse-diagnostic
    --inverseDiagnosticAlgorithm=richardson
    --inverseSpectralEstimate=user
    --inverseDiagnosticRequireAssembledReference=0
    --inverseInitialGuess=diagonal
    --numInverseIterations=20
    --inverseLambdaMin=0.01
    --inverseLambdaMax=10.0
    --massWeight=1.0
    --stiffnessWeight=1.0
    --maxAssembledEntries=1
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running opt-in direct-PAMatrix H1 Gram inverse diagnostic with user spectral bounds and no assembled reference"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixFactorSmoke
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=3
    --polyOrder=1:2
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=factor
    --massWeight=2.0
    --stiffnessWeight=0.5
    --numSolve=3
    --maxAssembledEntries=10000
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running small direct-PAMatrix H1 Gram local assembly/factor smoke test"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixAssemble2Sweep
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=3
    --polyOrder=8:12
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=assemble2
    --massWeight=1.0
    --stiffnessWeight=1.0
    --maxAssembledEntries=200000000
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running opt-in direct-PAMatrix H1 Gram column-apply assembly p-sweep"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixAssembleSweep
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=3
    --polyOrder=8:12
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=assemble
    --massWeight=1.0
    --stiffnessWeight=1.0
    --maxAssembledEntries=200000000
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running opt-in direct-PAMatrix H1 Gram standard assemble p-sweep"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixFactorSweep
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=3
    --polyOrder=3:5
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=factor
    --massWeight=1.0
    --stiffnessWeight=1.0
    --numSolve=10
    --maxAssembledEntries=20000000
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running opt-in direct-PAMatrix H1 Gram local assembly/factor/solve p-sweep"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixAssemblyCompareSweep
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=3
    --polyOrder=8:12
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=hgrad
    --mode=assembly-compare
    --checkAssemblyAgreement=1
    --massWeight=1.0
    --stiffnessWeight=1.0
    --maxAssembledEntries=200000000
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running opt-in direct-PAMatrix H1 Gram standard-vs-column-apply assembly p-sweep with agreement check"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixSynthetic4DAssemble2Sweep
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=4
    --polyOrder=4:8
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=synthetic
    --mode=assemble2
    --massWeight=1.0
    --stiffnessWeight=1.0
    --maxAssembledEntries=200000000
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running opt-in synthetic 4D H1 Gram column-apply assembly p-sweep"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixSynthetic4DAssembleSweep
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=4
    --polyOrder=4:8
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=synthetic
    --mode=assemble
    --massWeight=1.0
    --stiffnessWeight=1.0
    --maxAssembledEntries=200000000
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running opt-in synthetic 4D H1 Gram standard assemble p-sweep"
  VERBATIM)

add_custom_target(runTensorH1GramPAMatrixSynthetic4DAssemblyCompareSweep
  COMMAND $<TARGET_FILE:TensorPoissonPAMatrixDriver>
    --dim=4
    --polyOrder=4:8
    --cellsPerDim=1
    --operator=h1-gram
    --basisMode=synthetic
    --mode=assembly-compare
    --checkAssemblyAgreement=1
    --massWeight=1.0
    --stiffnessWeight=1.0
    --maxAssembledEntries=200000000
  DEPENDS TensorPoissonPAMatrixDriver
  COMMENT "Running opt-in synthetic 4D H1 Gram standard-vs-column-apply assembly p-sweep with agreement check"
  VERBATIM)

if (TARGET PlanarMITLRelativisticSpaceTime)
  set(PHASE0_PLANAR_MITL_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/phase0-planar-mitl" CACHE PATH "Output directory for Phase 0 Planar MITL performance runs")

  if (MPIEXEC_EXECUTABLE)
    set(PHASE0_MPI_1RANK_COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1)
    set(PHASE0_MPI_2RANK_COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2)
  else()
    set(PHASE0_MPI_1RANK_COMMAND)
    set(PHASE0_MPI_2RANK_COMMAND)
  endif()

  add_custom_target(runPhase0PerformanceSmall
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PHASE0_PLANAR_MITL_OUTPUT_DIR}/small"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${PHASE0_PLANAR_MITL_OUTPUT_DIR}/small/timing.log"
    COMMAND ${CMAKE_COMMAND} -E env OMP_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=threads
            ${PHASE0_MPI_1RANK_COMMAND} $<TARGET_FILE:PlanarMITLRelativisticSpaceTime>
            --vElements=2
            --xElements=8
            --maxDofsForDirectSolve=0
            --tElements=1
            --numUniformRefinements=0
            --p=4
            --dp=2
            --tWidth=1e-1
            --numTimeSlabs=2
            --ratio=1.5
            --outputDir=${PHASE0_PLANAR_MITL_OUTPUT_DIR}/small
            --minSubsteps=2
            --fixedPointTol=1e-4
            --useSerendipityBasis
            --J=5e8
            --V=1.5e7
            --noViz
            --phase0Instrumentation
            --phase0TensorGraphSchurBackend=assembled_crs
            --phase0TimingFile=${PHASE0_PLANAR_MITL_OUTPUT_DIR}/small/timing.log
    DEPENDS PlanarMITLRelativisticSpaceTime
    COMMENT "Running Phase 0 small Planar MITL performance baseline on one MPI rank"
    VERBATIM)

  add_custom_target(runPhase0PerformanceSmallMatrixFreeGraphSchur
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PHASE0_PLANAR_MITL_OUTPUT_DIR}/small-matrix-free-graph-schur"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${PHASE0_PLANAR_MITL_OUTPUT_DIR}/small-matrix-free-graph-schur/timing.log"
    COMMAND ${CMAKE_COMMAND} -E env OMP_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=threads
            ${PHASE0_MPI_1RANK_COMMAND} $<TARGET_FILE:PlanarMITLRelativisticSpaceTime>
            --vElements=2
            --xElements=8
            --maxDofsForDirectSolve=0
            --tElements=1
            --numUniformRefinements=0
            --p=4
            --dp=2
            --tWidth=1e-1
            --numTimeSlabs=2
            --ratio=1.5
            --outputDir=${PHASE0_PLANAR_MITL_OUTPUT_DIR}/small-matrix-free-graph-schur
            --minSubsteps=2
            --fixedPointTol=1e-4
            --useSerendipityBasis
            --J=5e8
            --V=1.5e7
            --noViz
            --phase0Instrumentation
            --phase0TensorGraphSchurBackend=matrix_free
            --phase0TimingFile=${PHASE0_PLANAR_MITL_OUTPUT_DIR}/small-matrix-free-graph-schur/timing.log
    DEPENDS PlanarMITLRelativisticSpaceTime
    COMMENT "Running Phase 0 small Planar MITL matrix-free graph-norm Schur micro-apply on one MPI rank"
    VERBATIM)

  add_custom_target(runPhase0PerformanceSmallPhase8MomentCompare
    COMMAND ${CMAKE_COMMAND} -E echo "runPhase0PerformanceSmallPhase8MomentCompare is disabled: accepted accuracy data is recorded, and the legacy direct-quadrature comparison path is too slow for routine runs."
    COMMENT "Phase 8 small Planar MITL strict tensor-vs-legacy moment comparison disabled"
    VERBATIM)

  add_custom_target(runPhase0PerformanceCompareGraphSchur
    COMMAND ${CMAKE_COMMAND}
            -DASSEMBLED_LOG=${PHASE0_PLANAR_MITL_OUTPUT_DIR}/small/timing.log
            -DMATRIX_FREE_LOG=${PHASE0_PLANAR_MITL_OUTPUT_DIR}/small-matrix-free-graph-schur/timing.log
            -P ${CMAKE_CURRENT_SOURCE_DIR}/ComparePhase0GraphSchurTimings.cmake
    DEPENDS runPhase0PerformanceSmall runPhase0PerformanceSmallMatrixFreeGraphSchur
    COMMENT "Run Phase 0 small Planar MITL assembled-CRS and matrix-free graph-norm Schur micro-apply targets for timing-log comparison")

  add_custom_target(runPhase0PerformanceExpandedGraphSchurAssembled
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PHASE0_PLANAR_MITL_OUTPUT_DIR}/expanded-assembled-graph-schur"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${PHASE0_PLANAR_MITL_OUTPUT_DIR}/expanded-assembled-graph-schur/timing.log"
    COMMAND ${CMAKE_COMMAND} -E env OMP_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=threads
            ${PHASE0_MPI_1RANK_COMMAND} $<TARGET_FILE:PlanarMITLRelativisticSpaceTime>
            --vElements=2
            --xElements=8
            --maxDofsForDirectSolve=0
            --tElements=1
            --numUniformRefinements=0
            --p=4
            --dp=2
            --tWidth=1e-1
            --numTimeSlabs=0
            --ratio=1.5
            --outputDir=${PHASE0_PLANAR_MITL_OUTPUT_DIR}/expanded-assembled-graph-schur
            --minSubsteps=2
            --fixedPointTol=1e-4
            --useSerendipityBasis
            --J=5e8
            --V=1.5e7
            --noViz
            --phase0Instrumentation
            --phase0TensorGraphSchurBackend=assembled_crs
            --phase0TensorGraphSchurXElements=4
            --phase0TensorGraphSchurTElements=1
            --phase0TensorGraphSchurUxElements=2
            --phase0TensorGraphSchurUyElements=4
            --phase0TensorGraphSchurComponentP=2
            --phase0TensorGraphSchurTestEnrichment=1
            --phase0TimingFile=${PHASE0_PLANAR_MITL_OUTPUT_DIR}/expanded-assembled-graph-schur/timing.log
    DEPENDS PlanarMITLRelativisticSpaceTime
    COMMENT "Running expanded Phase 0 assembled-CRS graph-norm Schur micro-apply on one MPI rank"
    VERBATIM)

  add_custom_target(runPhase0PerformanceExpandedGraphSchurMatrixFree
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PHASE0_PLANAR_MITL_OUTPUT_DIR}/expanded-matrix-free-graph-schur"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${PHASE0_PLANAR_MITL_OUTPUT_DIR}/expanded-matrix-free-graph-schur/timing.log"
    COMMAND ${CMAKE_COMMAND} -E env OMP_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=threads
            ${PHASE0_MPI_1RANK_COMMAND} $<TARGET_FILE:PlanarMITLRelativisticSpaceTime>
            --vElements=2
            --xElements=8
            --maxDofsForDirectSolve=0
            --tElements=1
            --numUniformRefinements=0
            --p=4
            --dp=2
            --tWidth=1e-1
            --numTimeSlabs=0
            --ratio=1.5
            --outputDir=${PHASE0_PLANAR_MITL_OUTPUT_DIR}/expanded-matrix-free-graph-schur
            --minSubsteps=2
            --fixedPointTol=1e-4
            --useSerendipityBasis
            --J=5e8
            --V=1.5e7
            --noViz
            --phase0Instrumentation
            --phase0TensorGraphSchurBackend=matrix_free
            --phase0TensorGraphSchurXElements=4
            --phase0TensorGraphSchurTElements=1
            --phase0TensorGraphSchurUxElements=2
            --phase0TensorGraphSchurUyElements=4
            --phase0TensorGraphSchurComponentP=2
            --phase0TensorGraphSchurTestEnrichment=1
            --phase0TimingFile=${PHASE0_PLANAR_MITL_OUTPUT_DIR}/expanded-matrix-free-graph-schur/timing.log
    DEPENDS PlanarMITLRelativisticSpaceTime
    COMMENT "Running expanded Phase 0 CRS-free matrix-free graph-norm Schur micro-apply on one MPI rank"
    VERBATIM)

  add_custom_target(runPhase0PerformanceCompareExpandedGraphSchur
    COMMAND ${CMAKE_COMMAND}
            -DASSEMBLED_LOG=${PHASE0_PLANAR_MITL_OUTPUT_DIR}/expanded-assembled-graph-schur/timing.log
            -DMATRIX_FREE_LOG=${PHASE0_PLANAR_MITL_OUTPUT_DIR}/expanded-matrix-free-graph-schur/timing.log
            -P ${CMAKE_CURRENT_SOURCE_DIR}/ComparePhase0GraphSchurTimings.cmake
    DEPENDS runPhase0PerformanceExpandedGraphSchurAssembled runPhase0PerformanceExpandedGraphSchurMatrixFree
    COMMENT "Run expanded Phase 0 assembled-CRS and matrix-free graph-norm Schur micro-apply targets for timing-log comparison")

  add_custom_target(runPhase0PerformanceMedium
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PHASE0_PLANAR_MITL_OUTPUT_DIR}/medium"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${PHASE0_PLANAR_MITL_OUTPUT_DIR}/medium/timing.log"
    COMMAND ${CMAKE_COMMAND} -E env OMP_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=threads
            ${PHASE0_MPI_2RANK_COMMAND} $<TARGET_FILE:PlanarMITLRelativisticSpaceTime>
            --vElements=2
            --xElements=8
            --maxDofsForDirectSolve=0
            --tElements=1
            --numUniformRefinements=1
            --p=4
            --dp=2
            --tWidth=1e-1
            --numTimeSlabs=5
            --ratio=1.5
            --outputDir=${PHASE0_PLANAR_MITL_OUTPUT_DIR}/medium
            --minSubsteps=2
            --fixedPointTol=1e-4
            --useSerendipityBasis
            --J=5e8
            --V=1.5e7
            --noViz
            --phase0Instrumentation
            --phase0TimingFile=${PHASE0_PLANAR_MITL_OUTPUT_DIR}/medium/timing.log
    DEPENDS PlanarMITLRelativisticSpaceTime
    COMMENT "Running Phase 0 medium Planar MITL performance baseline on two MPI ranks"
    VERBATIM)

  add_custom_target(runPhase0PerformanceMediumPhase8MomentCompare
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PHASE0_PLANAR_MITL_OUTPUT_DIR}/medium-phase8-moment-compare"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${PHASE0_PLANAR_MITL_OUTPUT_DIR}/medium-phase8-moment-compare/timing.log"
    COMMAND ${CMAKE_COMMAND} -E env OMP_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=threads
            ${PHASE0_MPI_2RANK_COMMAND} $<TARGET_FILE:PlanarMITLRelativisticSpaceTime>
            --vElements=2
            --xElements=8
            --maxDofsForDirectSolve=0
            --tElements=1
            --numUniformRefinements=1
            --p=4
            --dp=2
            --tWidth=1e-1
            --numTimeSlabs=5
            --ratio=1.5
            --outputDir=${PHASE0_PLANAR_MITL_OUTPUT_DIR}/medium-phase8-moment-compare
            --minSubsteps=2
            --fixedPointTol=1e-4
            --useSerendipityBasis
            --J=5e8
            --V=1.5e7
            --noViz
            --phase0Instrumentation
            --phase0TensorGraphSchurBackend=off
            --phase8MomentBackend=compare
            --phase8MomentCompareStrict
            --phase0TimingFile=${PHASE0_PLANAR_MITL_OUTPUT_DIR}/medium-phase8-moment-compare/timing.log
    DEPENDS PlanarMITLRelativisticSpaceTime
    COMMENT "Running opt-in Phase 8 medium Planar MITL strict tensor-vs-legacy moment comparison on two MPI ranks"
    VERBATIM)
endif()
