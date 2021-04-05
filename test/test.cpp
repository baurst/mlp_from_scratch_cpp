#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

int factorial(int foo) {
  int result = 1;
  for (int i = 1; i <= foo; ++i) {
    result *= i;
  }
  return result;
}

TEST_CASE("factorials are computed", "[factorial]") {
  REQUIRE(factorial(1) == 1);
  REQUIRE(factorial(2) == 2);
  REQUIRE(factorial(3) == 6);
  REQUIRE(factorial(10) == 3628800);
}
