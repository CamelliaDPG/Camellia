project(Poisson1D)

add_executable(Poisson1D "Poisson1D.cpp")
target_link_libraries(Poisson1D 
  Camellia
)

