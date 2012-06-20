set(DRIVER_NAME ConservativeConfusion)
project(${DRIVER_NAME})

set(SRC
  ConfusionDriver.cpp
)
add_executable(${DRIVER_NAME} ${SRC})
target_link_libraries(${DRIVER_NAME}
  Camellia
)
