set(DRIVER_NAME NavierStokes)
project(${DRIVER_NAME})

set(SRC
  NSDriver.cpp
)
add_executable(${DRIVER_NAME} ${SRC})
