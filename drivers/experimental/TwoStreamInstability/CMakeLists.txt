project(TwoStreamInstability)

MESSAGE("Adding executables for TwoStreamInstability.")
add_executable( TwoStreamInstability "TwoStreamInstability.cpp" 
                Boltzmann1D1VFormulation.cpp Boltzmann1D1VFormulation.h
                PhasePhysicalMeshMap.cpp PhasePhysicalMeshMap.h
                QuadraturePointFunction.cpp QuadraturePointFunction.h )

target_link_libraries(TwoStreamInstability 
  Camellia
)

add_executable( Boltzmann1D3V_TwoStreamVerification Boltzmann1D3V_TwoStreamVerification.cpp
                Boltzmann1D3VFormulation.cpp Boltzmann1D3VFormulation.h
                Boltzmann1D1VFormulation.cpp Boltzmann1D1VFormulation.h
                PhasePhysicalMeshMap.cpp PhasePhysicalMeshMap.h
                QuadraturePointFunction.cpp QuadraturePointFunction.h )

target_link_libraries(Boltzmann1D3V_TwoStreamVerification
  Camellia
)

add_executable( BoltzmannTests BoltzmannTests.cpp
                Boltzmann1D1VFormulation.cpp Boltzmann1D1VFormulation.h
                PhasePhysicalMeshMap.cpp PhasePhysicalMeshMap.h
                QuadraturePointFunction.cpp QuadraturePointFunction.h
                Boltzmann1D1VFormulationTests.cpp
                PhasePhysicalMeshMap1D1VTests.cpp
                Boltzmann1D3VFormulation.cpp Boltzmann1D3VFormulation.h
                Boltzmann1D3VFormulationTests.cpp
                PhasePhysicalMeshMap1D3VTests.cpp
                QuadraturePointFunctionTests.cpp )

target_link_libraries(BoltzmannTests 
  Camellia
)

add_test(NAME BoltzmannUnitTests COMMAND BoltzmannTests)
add_test(NAME Boltzmann1D3V_TwoStreamVerification COMMAND Boltzmann1D3V_TwoStreamVerification)
