set(DRIVER_NAME Burgers)
project(${DRIVER_NAME})

set(SRC
  BurgersBilinearForm.cpp
  BurgersProblem.cpp
  BurgersInnerProduct.cpp
  TimeMarchingBurgersProblem.cpp
)
add_executable(${DRIVER_NAME} BurgersDriver.cpp ${SRC})
add_executable(TimeBurgers TimeMarchingBurgersDriver.cpp ${SRC})
