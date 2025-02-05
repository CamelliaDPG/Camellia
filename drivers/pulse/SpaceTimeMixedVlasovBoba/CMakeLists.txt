project(SpaceTimeMixedVlasovBobaDriver)

MESSAGE("Adding executable for SpaceTimeMixedVlasovBobaDriver.")
add_executable(SpaceTimeMixedVlasovBobaDriver "SpaceTimeMixedVlasovBobaDriver.cpp")
target_link_libraries(SpaceTimeMixedVlasovBobaDriver 
  Camellia
)

MESSAGE("Adding executable for SpaceTimeMixedVlasovCompressDriver.")
add_executable(SpaceTimeMixedVlasovCompressDriver
  SpaceTimeMixedVlasovSetup.cpp
  SpaceTimeMixedVlasovCompress.cpp
  SpaceTimeMixedVlasovCompressDriver.cpp
)
target_link_libraries(SpaceTimeMixedVlasovCompressDriver Camellia)

MESSAGE("Adding executable for SpaceTimeMixedVlasovBlockSolveDriver.")
add_executable(SpaceTimeMixedVlasovBlockSolveDriver
  SpaceTimeMixedVlasovSetup.cpp
  SpaceTimeMixedVlasovBlockSolveDriver.cpp
)
target_link_libraries(SpaceTimeMixedVlasovBlockSolveDriver Camellia)

configure_file(assembleBlocks.m assembleBlocks.m COPYONLY)
configure_file(fgmres.m fgmres.m COPYONLY)
