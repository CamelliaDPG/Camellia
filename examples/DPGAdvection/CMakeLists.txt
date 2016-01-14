project(DPGAdvectionExample)

add_executable(DPGAdvectionExample DPGAdvectionExample.cpp)

target_link_libraries(DPGAdvectionExample
  Camellia
)
