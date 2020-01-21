project(StefanDrivers)

MESSAGE("Adding executable for HeatTransientDriver.")
add_executable(HeatTransientDriver "HeatTransientDriver.cpp")
target_link_libraries(HeatTransientDriver Camellia)

MESSAGE("Adding executable for HeatSteadyDriver.")
add_executable(HeatSteadyDriver "HeatSteadyDriver.cpp")
target_link_libraries(HeatSteadyDriver Camellia)