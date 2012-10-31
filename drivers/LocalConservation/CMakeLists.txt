project(LocalConservation)

add_executable(ConservativeConfusion ConfusionDriver.cpp)
target_link_libraries(ConservativeConfusion 
  Camellia
)

add_executable(SimpleConfusion SimpleConfusion.cpp)
target_link_libraries(SimpleConfusion 
  Camellia
)

add_executable(DoubleGlazingDriver DoubleGlazingDriver.cpp)
target_link_libraries(DoubleGlazingDriver 
  Camellia
)

add_executable(EricksonDriver EricksonDriver.cpp)
target_link_libraries(EricksonDriver 
  Camellia
)

add_executable(BendDriver BendDriver.cpp)
target_link_libraries(BendDriver 
  Camellia
)

add_executable(HemkerDriver HemkerDriver.cpp)
target_link_libraries(HemkerDriver 
  Camellia
)

add_executable(AirfoilDriver AirfoilDriver.cpp)
target_link_libraries(AirfoilDriver 
  Camellia
)

add_executable(VortexDriver VortexDriver.cpp)
target_link_libraries(VortexDriver 
  Camellia
)

add_executable(TransientConfusion TransientConfusion.cpp)
target_link_libraries(TransientConfusion 
  Camellia
)

add_executable(TransientHemker TransientHemker.cpp)
target_link_libraries(TransientHemker 
  Camellia
)

add_executable(InviscidBurgers InviscidBurgers.cpp)
target_link_libraries(InviscidBurgers 
  Camellia
)
