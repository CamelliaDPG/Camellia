project(VlasovPoissonBackwardEuler)

MESSAGE("Adding executable for BumpOnTail.")
add_executable( BumpOnTail "BumpOnTail.cpp" )

target_link_libraries( BumpOnTail Camellia )

MESSAGE("Adding executable for LandauDamping.")
add_executable( LandauDamping "LandauDamping.cpp" )

target_link_libraries( LandauDamping Camellia )

MESSAGE("Adding executable for TwoStream.")
add_executable( TwoStream "TwoStream.cpp" )

target_link_libraries( TwoStream Camellia )
