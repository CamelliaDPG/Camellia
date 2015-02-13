project(SlowTests)

FILE(GLOB TEST_SOURCES "*.cpp")

add_executable(runSlowTests ${TEST_SOURCES})

target_link_libraries(runSlowTests ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} Camellia
)

add_test(NAME runSlowTests COMMAND runSlowTests)