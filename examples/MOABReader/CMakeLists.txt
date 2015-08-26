project(MOABReaderExamples)

# hard-code MOAB location, for now:
SET(MOAB_PATH "/Users/nroberts/lib/moab")

MESSAGE(" Looking for MOAB in ${MOAB_PATH}/lib")

# Find MOAB
find_package(MOAB PATHS ${MOAB_PATH}/lib)

if (MOAB_FOUND)
  include_directories(
    ${MOAB_INCLUDE_DIRS}
  )

  add_executable(MOABReaderExample MOABReaderExample.cpp)

  target_link_libraries(MOABReaderExample 
    Camellia
    ${MOAB_LIBRARIES}
  )
else()
  MESSAGE("MOAB not found; not configuring MOABReaderExample.")
endif(MOAB_FOUND)