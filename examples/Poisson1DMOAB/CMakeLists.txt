project(Poisson1DMOAB)

add_executable(Poisson1DMOAB "Poisson1DMOAB.cpp")
target_link_libraries(Poisson1DMOAB 
  Camellia
)

