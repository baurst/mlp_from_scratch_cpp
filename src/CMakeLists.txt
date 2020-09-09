
add_subdirectory(utils)
add_subdirectory(mnist)
add_subdirectory(layer)
add_subdirectory(mlp)

add_executable(main main.cpp)

target_link_libraries(main PRIVATE layer mlp mnist utils)
target_compile_options(main PRIVATE -Wall -Wextra)