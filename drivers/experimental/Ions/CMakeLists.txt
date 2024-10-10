project(IonsDriver)

MESSAGE("Adding executable for IonsDriver.")
add_executable(IonsDriver "IonsDriver.cpp")
target_link_libraries(IonsDriver 
  Camellia
)

MESSAGE("Adding executable for IonsDriverForRefinementBug.")
add_executable(IonsDriverForRefinementBug "IonsDriverForRefinementBug.cpp")
target_link_libraries(IonsDriverForRefinementBug 
  Camellia
)

MESSAGE("Adding executable for SpaceTimeIonsDriver.")
add_executable(SpaceTimeIonsDriver "SpaceTimeIonsDriver.cpp")
target_link_libraries(SpaceTimeIonsDriver 
  Camellia
)

MESSAGE("Adding executable for SpaceTimeIonsDriverDiagnostics.")
add_executable(SpaceTimeIonsDriverDiagnostics "SpaceTimeIonsDriverDiagnostics.cpp")
target_link_libraries(SpaceTimeIonsDriverDiagnostics 
  Camellia
)

#install (TARGETS IonsDriver DESTINATION bin/examples/)

