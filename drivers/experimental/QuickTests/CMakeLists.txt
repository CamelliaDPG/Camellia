project(QuickTests)

MESSAGE("Adding executable for QuickTests.")
add_executable( QuickTests "mainTests.cpp" 
                QuickTests.cpp )

target_link_libraries(QuickTests 
  Camellia
)