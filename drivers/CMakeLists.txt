project(DPGDrivers)

option(BUILD_DPGTESTS_DRIVER "Build DPGTests driver" ON)
option(BUILD_TRUMAN_DRIVERS "Build drivers in Truman directory" ON)
option(BUILD_TRUMAN_TESTS_DRIVERS "Build drivers in Truman/Tests directory" OFF)
option(BUILD_CONFUSION_JESSE_DRIVERS "Build drivers in Confusion_Jesse directory" ON)
option(BUILD_CONVECTION_DRIVERS "Build drivers in Convection directory" ON)
option(BUILD_INCOMPRESSIBLENS_DRIVERS "Build drivers in IncompressibleNS directory" ON)
option(BUILD_PRECONDITIONING_DRIVERS "Build drivers in Preconditioning directory" ON)

# Include headers from DPGTests for some drivers
include_directories(DPGTests)

# Add each driver
#add_subdirectory(Burgers)
if (BUILD_CONFUSION_JESSE_DRIVERS)
  add_subdirectory(Confusion_Jesse)
  MESSAGE("Setting up makefiles for drivers in drivers/Confusion_Jesse, because BUILD_CONFUSION_JESSE_DRIVERS is ON.")
else()
  MESSAGE("Not setting up makefiles for drivers in drivers/Confusion_Jesse, because BUILD_CONFUSION_JESSE_DRIVERS is OFF.")  
endif(BUILD_CONFUSION_JESSE_DRIVERS)

if (BUILD_CONVECTION_DRIVERS)
  add_subdirectory(Convection)
  MESSAGE("Setting up makefiles for drivers in drivers/Convection, because BUILD_CONVECTION_DRIVERS is ON.")
else()
  MESSAGE("Not setting up makefiles for drivers in drivers/Convection, because BUILD_CONVECTION_DRIVERS is OFF.")  
endif(BUILD_CONVECTION_DRIVERS)

if (BUILD_DPGTESTS_DRIVER)
  add_subdirectory(DPGTests)
  MESSAGE("Setting up makefile for DPGTests driver in drivers/DPGTests, because BUILD_DPGTESTS_DRIVER is ON.")
else()
  MESSAGE("Not setting up makefile for DPGTests driver in drivers/DPGTests, because BUILD_DPGTESTS_DRIVER is OFF.")  
endif(BUILD_DPGTESTS_DRIVER)

if (BUILD_INCOMPRESSIBLENS_DRIVERS)
  add_subdirectory(IncompressibleNS)
  MESSAGE("Setting up makefiles for drivers in drivers/IncompressibleNS, because BUILD_INCOMPRESSIBLENS_DRIVERS is ON.")
else()
  MESSAGE("Not setting up makefiles for drivers in drivers/IncompressibleNS, because BUILD_INCOMPRESSIBLENS_DRIVERS is OFF.")  
endif(BUILD_INCOMPRESSIBLENS_DRIVERS)

if (BUILD_PRECONDITIONING_DRIVERS)
  add_subdirectory(Preconditioning)
  MESSAGE("Setting up makefiles for drivers in drivers/Preconditioning, because BUILD_PRECONDITIONING_DRIVERS is ON.")
else()
  MESSAGE("Not setting up makefiles for drivers in drivers/Preconditioning, because BUILD_PRECONDITIONING_DRIVERS is OFF.")  
endif(BUILD_PRECONDITIONING_DRIVERS)

add_subdirectory(MeshMemorySize)
add_subdirectory(NavierStokes)
add_subdirectory(NonlinearTests)
add_subdirectory(Poisson)
add_subdirectory(ScratchPad)
add_subdirectory(Stokes)

if (BUILD_TRUMAN_DRIVERS)
  add_subdirectory(Truman)
  MESSAGE("Setting up makefiles for drivers in drivers/Truman, because BUILD_TRUMAN_DRIVERS is ON.")
else()
  MESSAGE("Not setting up makefiles for drivers in drivers/Truman, because BUILD_TRUMAN_DRIVERS is OFF.")  
endif(BUILD_TRUMAN_DRIVERS)

if (BUILD_TRUMAN_TESTS_DRIVERS)
  add_subdirectory(Truman/Tests)
  MESSAGE("Setting up makefiles for drivers in drivers/Truman/Tests, because BUILD_TRUMAN_TESTS_DRIVERS is ON.")
else()
  MESSAGE("Not setting up makefiles for drivers in drivers/Truman/Tests, because BUILD_TRUMAN_TESTS_DRIVERS is OFF.")  
endif(BUILD_TRUMAN_TESTS_DRIVERS)


add_subdirectory(TutorialDriver)

