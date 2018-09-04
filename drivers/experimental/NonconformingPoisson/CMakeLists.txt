project(NonconformingPoissonDriver)

MESSAGE("Adding executable for NonconformingPoissonDriver.")
add_executable(NonconformingPoissonDriver "NonconformingPoissonDriver.cpp")
target_link_libraries(NonconformingPoissonDriver 
  Camellia
)

#install (TARGETS NonconformingPoissonDriver DESTINATION bin/examples/)

