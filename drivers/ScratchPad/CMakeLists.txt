project(ScratchPad)

add_executable(Burgers NewBurgersDriver.cpp)
target_link_libraries(Burgers
  Camellia
)
