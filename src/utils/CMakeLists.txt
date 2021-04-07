add_library(utils SHARED utils.cpp)
target_include_directories(utils PUBLIC include)
target_compile_options(utils PRIVATE -Wall -Wextra -pedantic -Werror)