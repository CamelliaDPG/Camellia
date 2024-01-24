project(ProximalDriver)

MESSAGE("Adding executable for ProximalDriver.")
add_executable(ProximalDriver "ProximalDriver.cpp")
target_link_libraries(ProximalDriver 
  Camellia
)

#install (TARGETS ProximalDriver DESTINATION bin/examples/)

