project(B-dotDriver)

MESSAGE("Adding executable for B-dotDriver.")
add_executable(B-dotDriver "B-dotDriver.cpp")
target_link_libraries(B-dotDriver 
  Camellia
)

#install (TARGETS B-dotDriver DESTINATION bin/examples/)

