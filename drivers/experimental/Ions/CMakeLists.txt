project(IonsDriver)

MESSAGE("Adding executable for IonsDriver.")
add_executable(IonsDriver "IonsDriver.cpp")
target_link_libraries(IonsDriver 
  Camellia
)

#install (TARGETS IonsDriver DESTINATION bin/examples/)

