project(ShearAlfvenWave)

set( SHEAR_ALFVEN_WAVE_EXAMPLE "ShearAlfvenWaveExample.cpp" )

add_executable(ShearAlfvenWaveExample ${SHEAR_ALFVEN_WAVE_EXAMPLE})
target_link_libraries(ShearAlfvenWaveExample
  Camellia
)
