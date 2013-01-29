project(NonlinearTests)

add_executable(InviscidBurgersTest InviscidBurgers.cpp)
target_link_libraries(InviscidBurgersTest
  Camellia
)
