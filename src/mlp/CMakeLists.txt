add_library(mlp SHARED mlp.cpp)
target_include_directories(mlp PUBLIC include)
target_link_libraries(mlp PRIVATE utils layer)
target_compile_options(mlp PRIVATE -Wall -Wextra -pedantic -Werror)