project(VlasovPoissonBackwardEuler)

MESSAGE("Adding executables for BumpOnTail.")
add_executable( BumpOnTail "BumpOnTail.cpp" )

target_link_libraries( BumpOnTail Camellia )

MESSAGE("Adding executables for TwoStream.")
add_executable( TwoStream "TwoStream.cpp" )

target_link_libraries( TwoStream Camellia )
