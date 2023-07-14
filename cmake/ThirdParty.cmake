### for mnn and mnn_opencv
set(mnn ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/include/)
message(STATUS "mnn: ${mnn}")
include_directories(${mnn})

set(depLibs ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/lib/*)
file(GLOB thirdPartyLibs ${depLibs})
message(STATUS "depLibs: ${thirdPartyLibs}")












