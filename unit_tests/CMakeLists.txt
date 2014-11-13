project(UnitTests)

FILE(GLOB TEST_SOURCES "*.cpp")

add_executable(runTests ${TEST_SOURCES})

target_link_libraries(runTests ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} Camellia
)

add_test(NAME runTests COMMAND runTests)