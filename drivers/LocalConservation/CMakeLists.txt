project(LocalConservation)

add_executable(SimpleConvection SimpleConvection.cpp)

add_executable(RotatingCylinder RotatingCylinder.cpp)

add_executable(ConservativeConfusion ConfusionDriver.cpp)

add_executable(SimpleConfusion SimpleConfusion.cpp)

add_executable(DoubleGlazingDriver DoubleGlazingDriver.cpp)

add_executable(EricksonDriver EricksonDriver.cpp)

add_executable(BendDriver BendDriver.cpp)

add_executable(HemkerDriver HemkerDriver.cpp)

add_executable(AirfoilDriver AirfoilDriver.cpp)

add_executable(VortexDriver VortexDriver.cpp)

add_executable(SingularWedge SingularWedge.cpp)

add_executable(TransientConfusion TransientConfusion.cpp)

add_executable(TransientHemker TransientHemker.cpp)

add_executable(InviscidBurgers InviscidBurgers.cpp)
