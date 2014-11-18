project(TutorialDriver)

set( TUTORIAL_DRIVER "TutorialDriver.cpp" )

add_executable(TutorialDriver ${TUTORIAL_DRIVER})
target_link_libraries(TutorialDriver 
  Camellia
)