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
            --phase0TimingFile=${PHASE0_PLANAR_MITL_OUTPUT_DIR}/small/timing.log
    DEPENDS PlanarMITLRelativisticSpaceTime
    COMMENT "Running Phase 0 small Planar MITL performance baseline on one MPI rank"
    VERBATIM)

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
