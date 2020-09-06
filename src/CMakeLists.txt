
add_subdirectory(utils)
add_subdirectory(layer)
add_subdirectory(mlp)

add_executable(main main.cpp)

target_link_libraries(main PUBLIC layer mlp PRIVATE utils)
target_compile_options(main PRIVATE -Wall -Wextra)