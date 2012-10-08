project(DPGDrivers)

# Include headers from DPGTests for some drivers
include_directories(DPGTests)

# Add each driver
add_subdirectory(ConservativeConfusion)
add_subdirectory(NavierStokes)
add_subdirectory(Burgers)
add_subdirectory(DPGTests)
