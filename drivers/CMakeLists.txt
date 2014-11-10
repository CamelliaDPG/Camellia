project(DPGDrivers)

# Include headers from DPGTests for some drivers
include_directories(DPGTests)

# Add each driver
#add_subdirectory(Burgers)
add_subdirectory(Confusion_Jesse)
add_subdirectory(DPGTests)
add_subdirectory(IncompressibleNS)
add_subdirectory(NavierStokes)
add_subdirectory(NonlinearTests)
add_subdirectory(Poisson)
add_subdirectory(ScratchPad)
add_subdirectory(Stokes)
add_subdirectory(Truman)
add_subdirectory(TutorialDriver)

