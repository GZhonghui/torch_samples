#include <cstddef>

// extern "C" disables C++ name mangling so ctypes can find stable symbols.
extern "C" {

int add_ints(int left, int right) {
    return left + right;
}

void scale_in_place(float* values, std::size_t length, float factor) {
    for (std::size_t index = 0; index < length; ++index) {
        values[index] *= factor;
    }
}

}
