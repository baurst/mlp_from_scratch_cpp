add_library(layer SHARED layer.cpp)
target_include_directories(layer PUBLIC include)
target_link_libraries(layer PRIVATE utils)
target_compile_options(layer PRIVATE -Wall -Wextra -pedantic -Werror)