project(performance)

add_executable(StokesAssemblyPerformanceTest "StokesAssemblyPerformanceTest.cpp")
target_link_libraries(StokesAssemblyPerformanceTest Camellia)

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
endif()
