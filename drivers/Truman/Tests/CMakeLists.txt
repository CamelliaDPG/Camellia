project(TrumanTests)

add_executable(ExporterTests ExporterTests.cpp)

add_executable(XDMFTest XDMFTest.cpp)
target_link_libraries(XDMFTest Xdmf)