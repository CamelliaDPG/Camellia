project(RelativisticVlasovSpaceTime)

MESSAGE("Adding executable for ColdDiodeRelativisticSpaceTime.")
add_executable( ColdDiodeRelativisticSpaceTime "ColdDiodeRelativisticSpaceTime.cpp" )

target_link_libraries( ColdDiodeRelativisticSpaceTime Camellia )

MESSAGE("Adding executable for GenerateTimeSteps.")
add_executable( GenerateTimeSteps "GenerateTimeSteps.cpp" )

target_link_libraries( GenerateTimeSteps Camellia )

MESSAGE("Adding executable for LandauRelativisticSpaceTime.")
add_executable( LandauRelativisticSpaceTime "LandauRelativisticSpaceTime.cpp" )

target_link_libraries( LandauRelativisticSpaceTime Camellia )
