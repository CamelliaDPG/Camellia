project(MeshMemorySize)

FILE(GLOB DRIVER_SOURCES "*.cpp")

add_executable(MeshMemorySize ${DRIVER_SOURCES})
target_link_libraries(MeshMemorySize 
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  Camellia
)
