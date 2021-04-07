add_library(mnist SHARED mnist.cpp)
target_include_directories(mnist PUBLIC include)
target_link_libraries(mnist PRIVATE utils)
target_compile_options(mnist PRIVATE -Wall -Wextra -pedantic -Werror)