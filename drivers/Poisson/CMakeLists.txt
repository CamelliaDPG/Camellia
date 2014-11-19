project(StokesDrivers)

set( POISSON_DRIVER_GMG "PoissonGMGDriver.cpp" )

add_executable(PoissonGMGDriver ${POISSON_DRIVER_GMG})
target_link_libraries(PoissonGMGDriver 
  Camellia
)

add_executable(BarkerPoissonDriver "PoissonDriverBarkerAdditiveSchwarz.cpp")
target_link_libraries(BarkerPoissonDriver Camellia)

add_executable(PoissonPreconditioningTestsDriver "PoissonPreconditioningTestsDriver.cpp")
target_link_libraries(PoissonPreconditioningTestsDriver Camellia)

add_executable(PoissonDriver2D "PoissonDriver2D.cpp")
target_link_libraries(PoissonDriver2D Camellia)
