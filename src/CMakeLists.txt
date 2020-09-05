
add_subdirectory(neuron)

add_executable(main main.cpp)

target_link_libraries(main PUBLIC neuron)