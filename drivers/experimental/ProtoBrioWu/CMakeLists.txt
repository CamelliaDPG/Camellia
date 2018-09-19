project(ProtoBrioWu)

set( PROTO_BRIO_WU_EXAMPLE "ProtoBrioWuExample.cpp" )

add_executable(ProtoBrioWuExample ${PROTO_BRIO_WU_EXAMPLE})
target_link_libraries(ProtoBrioWuExample
  Camellia
)
