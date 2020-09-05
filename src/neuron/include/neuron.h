#include<vector>
#include<numeric>

class Neuron {
    public:
    Neuron(size_t number_of_inputs, size_t number_of_outputs);
    float call(const std::vector<float> input);
   private:
      std::vector<float> weights;
      float bias;
};

template <typename T>
T dot(const std::vector<T> x, const std::vector<T> y) {
    const T dp = std::inner_product(x.begin(), x.end(), y.begin(), 0);
    return dp;
};
