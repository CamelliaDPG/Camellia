project(TwoStreamInstability)

MESSAGE("Adding executable for TwoStreamInstability.")
add_executable( TwoStreamInstability "TwoStreamInstability.cpp" 
                Boltzmann1D1VFormulation.cpp Boltzmann1D1VFormulation.h
                PhasePhysicalMeshMap1D1V.cpp PhasePhysicalMeshMap1D1V.h
                QuadraturePointFunction.cpp QuadraturePointFunction.h )

target_link_libraries(TwoStreamInstability 
  Camellia
)

add_executable( TwoStreamInstabilityTests TwoStreamInstabilityTests.cpp
                Boltzmann1D1VFormulation.cpp Boltzmann1D1VFormulation.h
                PhasePhysicalMeshMap1D1V.cpp PhasePhysicalMeshMap1D1V.h
                QuadraturePointFunction.cpp QuadraturePointFunction.h
                Boltzmann1D1VFormulationTests.cpp
                PhasePhysicalMeshMap1D1VTests.cpp
                QuadraturePointFunctionTests.cpp )

target_link_libraries(TwoStreamInstabilityTests 
  Camellia
)
