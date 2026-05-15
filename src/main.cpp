#include "tensor.h"
#include <iostream>
#include <vector>

int main() {
   // std::vector<float> test_v{{1.000, 2.000, 3.000},{1.000, 2.000, 3.000},{1.000, 2.000, 3.000}};
    Tensor test_tensor({{1.000, 2.000, 3.000},{1.000, 2.000, 3.000},{1.000, 2.000, 3.000}});

    std::cout << test_tensor << std::endl;
    return 0;
}

//cmake --build build --config Debug --target ml_app./build/Debug/ml_app.exe
