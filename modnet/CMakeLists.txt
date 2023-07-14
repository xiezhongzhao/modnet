file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(modnet SHARED ${srcs} include/modnet/mnn_handler.h src/mnn_handler.cpp include/modnet/common.h)
target_include_directories(modnet PUBLIC include)


 