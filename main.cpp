#include "src/audi.hpp"

using namespace audi;

int main() {
	gdual x(0., "y", 3);

	auto f = erf(x);
	std::cout << f << std::endl;
}