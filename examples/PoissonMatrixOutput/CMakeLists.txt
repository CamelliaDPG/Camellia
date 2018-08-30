project(PoissonMatrixOutput)

add_executable(PoissonMatrixOutput PoissonMatrixOutput.cpp)
target_link_libraries(PoissonMatrixOutput 
  Camellia
)

