project(Diode)

MESSAGE("Adding executables for TwoStream.")
add_executable( Diode "Diode.cpp" 
                Boltzmann1D1VFormulation.cpp Boltzmann1D1VFormulation.h
                PhasePhysicalMeshMap.cpp PhasePhysicalMeshMap.h )

target_link_libraries( Diode Camellia )
