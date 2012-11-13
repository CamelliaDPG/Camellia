project(Confusion_Jesse)

add_executable(Hughes Confusion_Hughes.cpp)
target_link_libraries(Hughes 
  Camellia
)

add_executable(Step Confusion_step.cpp)
target_link_libraries(Step
  Camellia
)
add_executable(Rates Rates.cpp)
target_link_libraries(Rates 
  Camellia
)

add_executable(Vortex Vortex.cpp)
target_link_libraries(Vortex
  Camellia
)

add_executable(Qopt QoptInflowTest.cpp)
target_link_libraries(Vortex
  Camellia
)

add_executable(SimpleTest SimpleTest.cpp)
target_link_libraries(SimpleTest
  Camellia
)
