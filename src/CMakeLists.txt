
add_subdirectory(utils)
add_subdirectory(layer)

add_executable(main main.cpp)

target_link_libraries(main PUBLIC layer)
target_link_libraries(main PRIVATE utils)
target_compile_options(main PRIVATE -Wall -Wextra)