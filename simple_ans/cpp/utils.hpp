#pragma once

#include <cstddef>
#include <memory>
#include <numeric>
#include <vector>

namespace simple_ans::utils
{

constexpr size_t VECTOR_WIDTH = []() constexpr
{
#if defined(__AVX512F__) || defined(_M_AVX512)
    return 512 / 8;
#elif defined(__AVX__) || defined(_M_AVX)
    return 256 / 8;
#elif defined(__SSE2__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    return 128 / 8;
#elif defined(__ARM_NEON) || defined(__aarch64__)  // ARM NEON support
    return 128 / 8;
#elif defined(__VSX__)                             // PowerPC VSX (Vector Scalar eXtension)
    return 128 / 8;
#elif defined(__ALTIVEC__)                         // PowerPC AltiVec
    return 128 / 8;
#else
    return 64 / 8;  // Default scalar width
#endif
}();

/**
 * An aligned allocator is a custom memory allocator that ensures the allocated memory is aligned to
 * a specified boundary. This is useful for optimizing performance on certain hardware architectures
 * that benefit from aligned memory access. The aligned allocator provides methods for allocating
 * and deallocating memory, and it can be used with standard containers like `std::vector` to create
 * aligned vectors.
 *
 * Alignment in computer memory refers to arranging data in memory according to certain boundaries.
 * This is important for performance and correctness on many hardware architectures. Here are the
 * key points:
 *
 * 1. **Performance**: Many processors can access aligned memory faster than unaligned memory. For
 * example, accessing a 4-byte integer that is aligned to a 4-byte boundary can be faster than
 * accessing one that is not. It also helps vectorization and SIMD operations on modern CPUs.
 *
 * 2. **Correctness**: Some hardware architectures require data to be aligned. Accessing unaligned
 * data can cause hardware exceptions or crashes.
 *
 * 3. **Memory Allocation**: When allocating memory, you can specify an alignment requirement. This
 * ensures that the starting address of the allocated memory block is a multiple of the specified
 * alignment.
 */
template <typename T, std::size_t Alignment>
struct aligned_allocator
{
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <class U>
    struct rebind
    {
        using other = aligned_allocator<U, Alignment>;
    };

    aligned_allocator() noexcept = default;

    template <typename U>
    explicit aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept
    {
    }

    // Allocates memory for n objects of type T with the specified alignment.
    static T* allocate(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();

        std::size_t bytes = n * sizeof(T);

        // std::aligned_alloc requires that 'bytes' is a multiple of Alignment.
        if (bytes % Alignment != 0)
        {
            bytes = ((bytes / Alignment) + 1) * Alignment;
        }

        void* ptr = std::aligned_alloc(Alignment, bytes);
        if (!ptr)
            throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    // Deallocates the memory pointed to by p.
    static void deallocate(T* p, std::size_t /*n*/) noexcept
    {
        std::free(p);
    }
};

// Two allocators are always considered equal.
template <typename T, typename U, std::size_t Alignment>
bool operator==(const aligned_allocator<T, Alignment>&,
                const aligned_allocator<U, Alignment>&) noexcept
{
    return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const aligned_allocator<T, Alignment>&,
                const aligned_allocator<U, Alignment>&) noexcept
{
    return false;
}

// Create an aligned vector with custom allocator
template <typename T, std::size_t Alignment = VECTOR_WIDTH>
using AlignedVector = std::vector<T, aligned_allocator<T, Alignment>>;

}  // namespace simple_ans::utils