add_subdirectory(catch2)
add_executable(tests test.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2 layer mlp mnist utils)
