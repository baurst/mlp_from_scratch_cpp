#include <iostream>
#include "neuron.h"

int main() {
    auto neu = Neuron(3,3);
    std::vector<float> foo = {1.0, 2.0, 3.0};
    const auto output = neu.call(foo);
    std::cout << "Output: " << output << std::endl;
    return 0;
}