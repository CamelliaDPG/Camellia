project(Confusion_Jesse)

add_executable(Time TransientTest.cpp)
target_link_libraries(Time
  Camellia
)

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
target_link_libraries(Qopt
  Camellia
)

add_executable(QoptMesh QoptInflowSchwabSuri.cpp)
target_link_libraries(QoptMesh
  Camellia
)

add_executable(SimpleTest SimpleTest.cpp)
target_link_libraries(SimpleTest
  Camellia
)

add_executable(Convection ConvectionTest.cpp)
target_link_libraries(Convection
  Camellia
)
