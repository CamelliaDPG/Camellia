project(SpaceTimeMixedVlasovBobaDriver)

MESSAGE("Adding executable for SpaceTimeMixedVlasovBobaDriver.")
add_executable(SpaceTimeMixedVlasovBobaDriver "SpaceTimeMixedVlasovBobaDriver.cpp")
set_target_properties(SpaceTimeMixedVlasovBobaDriver PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
)
target_link_libraries(SpaceTimeMixedVlasovBobaDriver 
  Camellia
)

MESSAGE("Adding executable for SpaceTimeMixedVlasovCompressDriver.")
add_executable(SpaceTimeMixedVlasovCompressDriver
  SpaceTimeMixedVlasovSetup.cpp
  SpaceTimeMixedVlasovCompress.cpp
  SpaceTimeMixedVlasovCompressDriver.cpp
)
set_target_properties(SpaceTimeMixedVlasovCompressDriver PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
)
target_link_libraries(SpaceTimeMixedVlasovCompressDriver Camellia)

MESSAGE("Adding executable for SpaceTimeMixedVlasovBlockSolveDriver.")
add_executable(SpaceTimeMixedVlasovBlockSolveDriver
  SpaceTimeMixedVlasovSetup.cpp
  SpaceTimeMixedVlasovBlockSolveDriver.cpp
)
set_target_properties(SpaceTimeMixedVlasovBlockSolveDriver PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
)
target_link_libraries(SpaceTimeMixedVlasovBlockSolveDriver Camellia)

configure_file(assembleBlocks.m assembleBlocks.m COPYONLY)
configure_file(fgmres.m fgmres.m COPYONLY)
