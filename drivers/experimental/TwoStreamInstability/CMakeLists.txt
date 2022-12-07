project(TwoStreamInstability)

MESSAGE("Adding executables for OLD-LandauDamping.")
add_executable( OLD-LandauDamping "LandauDamping.cpp" 
                Boltzmann1D1VFormulation.cpp Boltzmann1D1VFormulation.h
                PhasePhysicalMeshMap.cpp PhasePhysicalMeshMap.h
                VlasovPoissonSpaceTimeFormulation.cpp VlasovPoissonSpaceTimeFormulation.h
                PhasePhysicalMeshMap.cpp PhasePhysicalMeshMap.h )

target_link_libraries(OLD-LandauDamping 
  Camellia
)

MESSAGE("Adding executables for TwoStreamInstability.")
add_executable( TwoStreamInstability "TwoStreamInstability.cpp" 
                Boltzmann1D1VFormulation.cpp Boltzmann1D1VFormulation.h
                PhasePhysicalMeshMap.cpp PhasePhysicalMeshMap.h )

target_link_libraries(TwoStreamInstability 
  Camellia
)

MESSAGE("Adding executables for SpaceTimeVlasovPoisson.")
add_executable( SpaceTimeVlasovPoisson "SpaceTimeVlasovPoisson.cpp" 
                VlasovPoissonSpaceTimeFormulation.cpp VlasovPoissonSpaceTimeFormulation.h
                PhasePhysicalMeshMap.cpp PhasePhysicalMeshMap.h )

target_link_libraries(SpaceTimeVlasovPoisson 
  Camellia
)

MESSAGE("Adding executables for ColdDiodeProblemProblem.")
add_executable( ColdDiodeProblemProblem "ColdDiodeProblemProblem.cpp" 
                VlasovPoissonSpaceTimeFormulation.cpp VlasovPoissonSpaceTimeFormulation.h
                PhasePhysicalMeshMap.cpp PhasePhysicalMeshMap.h )

target_link_libraries(ColdDiodeProblemProblem 
  Camellia
)


add_executable( Boltzmann1D3V_TwoStreamVerification Boltzmann1D3V_TwoStreamVerification.cpp
                Boltzmann1D3VFormulation.cpp Boltzmann1D3VFormulation.h
                Boltzmann1D1VFormulation.cpp Boltzmann1D1VFormulation.h
                PhasePhysicalMeshMap.cpp PhasePhysicalMeshMap.h )

target_link_libraries(Boltzmann1D3V_TwoStreamVerification
  Camellia
)

add_executable( BoltzmannTests BoltzmannTests.cpp
                Boltzmann1D1VFormulation.cpp Boltzmann1D1VFormulation.h
                PhasePhysicalMeshMap.cpp PhasePhysicalMeshMap.h
                Boltzmann1D1VFormulationTests.cpp
                PhasePhysicalMeshMap1D1VTests.cpp
                Boltzmann1D3VFormulation.cpp Boltzmann1D3VFormulation.h
                VlasovPoissonSpaceTimeFormulation.cpp VlasovPoissonSpaceTimeFormulation.h
                VlasovSpaceTimeFormulation.cpp VlasovSpaceTimeFormulation.h
                Boltzmann1D3VFormulationTests.cpp
                PhasePhysicalMeshMap1D3VTests.cpp
                VlasovSpaceTimeFormulationTests.cpp
)

target_link_libraries(BoltzmannTests 
  Camellia
)

add_executable( Vlasov1D3VTimeMarchingWithCollisions Vlasov1D3VTimeMarchingWithCollisions.cpp
                Boltzmann1D3VFormulation.cpp Boltzmann1D3VFormulation.h
                PhasePhysicalMeshMap.cpp PhasePhysicalMeshMap.h )

target_link_libraries(Vlasov1D3VTimeMarchingWithCollisions
  Camellia
)

add_test(NAME BoltzmannUnitTests COMMAND BoltzmannTests)
add_test(NAME Boltzmann1D3V_TwoStreamVerification COMMAND Boltzmann1D3V_TwoStreamVerification)
