set(DRIVER_NAME NS)
project(${DRIVER_NAME})
set(SRC
  NSDriver.cpp
)
add_executable(${DRIVER_NAME} ${SRC})

set(DRIVER_NAME NSVis)
project(${DRIVER_NAME})
set(SRC
  PlateVisualizer.cpp
)
add_executable(${DRIVER_NAME} ${SRC})

set(DRIVER_NAME Ramp)
project(${DRIVER_NAME})
set(SRC
  RampDriver.cpp
)
add_executable(${DRIVER_NAME} ${SRC})
