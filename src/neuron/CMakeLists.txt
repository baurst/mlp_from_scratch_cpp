add_library(neuron SHARED neuron.cpp)
target_include_directories(neuron PUBLIC include)
target_link_libraries(neuron utils)