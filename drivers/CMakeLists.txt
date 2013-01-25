project(DPGDrivers)

# Include headers from DPGTests for some drivers
include_directories(DPGTests)

# Add each driver
add_subdirectory(LocalConservation)
add_subdirectory(NavierStokes)
add_subdirectory(Euler)
#add_subdirectory(Burgers)
add_subdirectory(ScratchPad)
add_subdirectory(IncompressibleNS)
#add_subdirectory(NonlinearTests)
add_subdirectory(DPGTests)
add_subdirectory(Confusion_Jesse)
add_subdirectory(Stokes)
