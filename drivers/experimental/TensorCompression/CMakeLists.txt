project(TensorCompression)

MESSAGE("Adding executable for TwoStreamDPGVerifyTT.")
add_executable( TwoStreamDPGVerifyTT "TwoStreamDPGVerifyTT.cpp" )

target_link_libraries( TwoStreamDPGVerifyTT Camellia )
