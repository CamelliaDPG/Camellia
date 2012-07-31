project(ConservativeConfusion)

add_executable(ConservativeConfusion ConfusionDriver.cpp)
target_link_libraries(ConservativeConfusion 
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
