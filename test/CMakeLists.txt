file(GLOB_RECURSE srcs CONFIGURE_DEPENDS *.cpp *.h)
add_executable(test ${srcs})
target_link_libraries(test PUBLIC modnet ${thirdPartyLibs})










