#pragma once

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <memory>
#include <numeric>

#include <ankerl/unordered_dense.h>

#include <iostream>

#if defined(__AVX512F__) || defined(_M_AVX512)
    #define VECTOR_WIDTH 512
#elif defined(__AVX__) || defined(_M_AVX)
    #define VECTOR_WIDTH 256
#elif defined(__SSE2__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    #define VECTOR_WIDTH 128
#elif defined(__ARM_NEON) || defined(__aarch64__)  // ARM NEON support
    #define VECTOR_WIDTH 128
#elif defined(__VSX__)  // PowerPC VSX (Vector Scalar eXtension)
    #define VECTOR_WIDTH 128
#elif defined(__ALTIVEC__)  // PowerPC AltiVec
    #define VECTOR_WIDTH 128
#else
    #define VECTOR_WIDTH 64  // Default scalar width
#endif

namespace simple_ans
{

struct EncodedData
{
    uint32_t state;
    std::vector<uint64_t>
        bitstream;    // Each uint64_t contains 64 bits, with padding in last word if needed
    size_t num_bits;  // Actual number of bits used (may be less than bitstream.size() * 64)
};

// Helper function to verify if a number is a power of 2
inline bool is_power_of_2(uint32_t x)
{
    return x && !(x & (x - 1));
}


template<typename T, std::size_t Alignment = alignof(T)>
struct aligned_allocator {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <class U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };

    aligned_allocator() noexcept = default;

    template <typename U>
    explicit aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

    // Allocates memory for n objects of type T with the specified alignment.
    static T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();

        std::size_t bytes = n * sizeof(T);

        // std::aligned_alloc requires that 'bytes' is a multiple of Alignment.
        if (bytes % Alignment != 0) {
            bytes = ((bytes / Alignment) + 1) * Alignment;
        }

        void* ptr = std::aligned_alloc(Alignment, bytes);
        if (!ptr)
            throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    // Deallocates the memory pointed to by p.
    static void deallocate(T* p, std::size_t /*n*/) noexcept {
        std::free(p);
    }
};

// Two allocators are always considered equal.
template<typename T, typename U, std::size_t Alignment>
bool operator==(const aligned_allocator<T, Alignment>&,
                const aligned_allocator<U, Alignment>&) noexcept {
    return true;
}

template<typename T, typename U, std::size_t Alignment>
bool operator!=(const aligned_allocator<T, Alignment>&,
                const aligned_allocator<U, Alignment>&) noexcept {
    return false;
}

// Create an aligned vector with custom allocator
template <typename T, std::size_t Alignment=64>
using AlignedVector = std::vector<T, aligned_allocator<T, Alignment>>;

// require the type to be numeric
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
T vector_accumulate(const T* data, const size_t n, const T init = T(0))
{
    constexpr static size_t block_size = VECTOR_WIDTH/sizeof(T);
    alignas(64) std::array<T, block_size> sums{T(0)};
    const auto pow2_size = n & -block_size;
    for (size_t i = 0; i < pow2_size; i += block_size)
    {
        for (size_t j = 0; j < block_size; ++j)
        {
            sums[j] += data[i + j];
        }
    }
    auto sum = std::accumulate(sums.begin(), sums.end(), init);
    for (size_t i = pow2_size; i < n; ++i)
    {
        sum += data[i];
    }
    return sum;
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
void vector_inclusive_scan(const T* input, T* aligned_output, const size_t num_symbols) {
    // assume aligned output
#if defined(__GNUC__) || defined(__clang__)
    aligned_output = static_cast<T*>(__builtin_assume_aligned(aligned_output, 64));
#endif
    static constexpr size_t block_size = VECTOR_WIDTH*2/sizeof(T);
    // Set the first element to 0
    aligned_output[0] = 0;
    // Let n be the number of elements in the input array.
    size_t n = num_symbols - 1;

    // Process blocks of size N.
    size_t pow2_size = n & ~(block_size - 1); // largest multiple of N less than or equal to n

    T s = 0; // running sum from previous blocks
    size_t i = 0;
    for (; i < pow2_size; i += block_size) {
        constexpr size_t Half = block_size / 2;
        // --- Process first lane (elements i ... i+Half-1) ---
        // Write the first prefix element: output[i+1] = input[i] + running sum.
        aligned_output[i + 1] = input[i] + s;
        for (size_t j = 1; j < Half; ++j) {
            // Each output element is the sum of the previous output and the current input.
            aligned_output[i + 1 + j] = aligned_output[i + j] + input[i + j];
        }
        // Save the last value of lane 0 as the local block sum.
        T lane0_sum = aligned_output[i + Half];

        // --- Process second lane (elements i+Half ... i+N-1) ---
        T temp[Half]; // temporary storage for the local prefix of lane 1
        temp[0] = input[i + Half];
        for (size_t j = 1; j < Half; ++j) {
            temp[j] = temp[j - 1] + input[i + Half + j];
        }
        // Add the lane0_sum to the local prefix values and write them out.
        for (size_t j = 0; j < Half; ++j) {
            aligned_output[i + 1 + Half + j] = temp[j] + lane0_sum;
        }
        // Update the running sum to be the last element computed in this block.
        s = aligned_output[i + block_size];
    }

    // --- Process any remaining elements with a simple scalar loop ---
    for (; i < n; i++) {
        aligned_output[i + 1] = aligned_output[i] + input[i];
    }
}



template <typename T>
EncodedData ans_encode_t(const T* signal,
                         size_t signal_size,
                         const uint32_t* symbol_counts,
                         const T* symbol_values,
                         size_t num_symbols);

template <typename T>
void ans_decode_t(T* output,
                  size_t n,
                  uint32_t state,
                  const uint64_t* bitstream,
                  size_t num_bits,
                  const uint32_t* symbol_counts,
                  const T* symbol_values,
                  size_t num_symbols);
}  // namespace simple_ans

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

namespace simple_ans
{
constexpr int unique_array_threshold = static_cast<int>(std::numeric_limits<uint16_t>::max()) + 1;
constexpr int lookup_array_threshold = unique_array_threshold;

template <typename T>
std::tuple<std::vector<T>, std::vector<uint64_t>> unique_with_counts(const T* values, size_t n)
{
    // WARNING: This is ONLY a helper function. It doesn't support arrays with a large domain, and
    // will instead fail return empty vectors. It is up to the caller to handle this case
    // separately. numpy.unique() is quite fast, with improvements to use vectorized sorts (in 2.x,
    // at least), so I didn't bother to implement a more efficient version here.
    std::vector<T> unique_values;
    std::vector<uint64_t> counts;
    if (!n)
    {
        return {unique_values, counts};
    }

    int64_t min_value = values[0];
    int64_t max_value = values[0];
    // Check if the range of values is small enough to use a lookup array
    for (size_t i = 1; i < n; ++i)
    {
        min_value = std::min(min_value, static_cast<int64_t>(values[i]));
        max_value = std::max(max_value, static_cast<int64_t>(values[i]));
    }

    if ((max_value - min_value + 1) <= unique_array_threshold)
    {
        std::vector<uint64_t> raw_counts(max_value - min_value + 1);
        for (size_t i = 0; i < n; ++i)
        {
            raw_counts[values[i] - min_value]+=1;
        }

        for (size_t i = 0; i < raw_counts.size(); ++i)
        {
            if (raw_counts[i])
            {
                unique_values.push_back(static_cast<T>(i + min_value));
                counts.push_back(raw_counts[i]);
            }
        }
    }

    return {std::move(unique_values), std::move(counts)};
}

inline void read_bits_from_end_of_bitstream(const uint64_t* bitstream,
                                            const int64_t& source_bit_position,
                                            uint32_t& dest,
                                            uint32_t dest_start_bit,
                                            uint32_t dest_end_bit)
{
    uint32_t d = dest_end_bit - dest_start_bit;
    if ((source_bit_position & 63) >= (d - 1))
    {
        // in this case we can grab all the bits we need at once from the current word
        uint32_t word_idx = source_bit_position >> 6;  // Divide by 64
        uint32_t bit_idx = source_bit_position & 63;   // Modulo 64
        // get bits from bit_idx - d + 1 to bit_idx
        auto bits =
            static_cast<uint32_t>((bitstream[word_idx] >> (bit_idx - d + 1)) & ((1 << d) - 1));
        dest |= (bits << dest_start_bit);
    }
    else
    {
        // this is possibly the slower case, but should be less common
        for (uint32_t j = 0; j < d; ++j)
        {
            uint32_t word_idx = (source_bit_position - j) >> 6;  // Divide by 64
            uint32_t bit_idx = (source_bit_position - j) & 63;   // Modulo 64
            dest |= (static_cast<uint32_t>((bitstream[word_idx] >> bit_idx) & 1)
                     << (d - 1 - j + dest_start_bit));
        }
    }
}

template <typename T>
EncodedData ans_encode_t(const T* signal,
                         size_t signal_size,
                         const uint32_t* symbol_counts,
                         const T* symbol_values,
                         size_t num_symbols)
{
    static_assert(sizeof(T) < sizeof(int64_t),
                  "Value range of T must fit in int64_t for table lookup");

    // Calculate L and verify it's a power of 2
    const auto L = vector_accumulate(symbol_counts, num_symbols);
    if (!is_power_of_2(L))
    {
        throw std::invalid_argument("L must be a power of 2");
    }

    // Pre-compute cumulative sums
    AlignedVector<uint32_t> C(num_symbols);
    vector_inclusive_scan(symbol_counts, C.data(), num_symbols);
    // Create symbol index lookup (for fallback)

    const auto [symbol_index_lookup, min_symbol, max_symbol] = [symbol_values, num_symbols]
    {
        ankerl::unordered_dense::map<T, size_t>&& symbol_index_lookup{};
        int64_t min_symbol = symbol_values[0];
        int64_t max_symbol = symbol_values[0];
        for (size_t i = 0; i < num_symbols; ++i)
        {
            symbol_index_lookup[symbol_values[i]] = i;
            min_symbol = std::min(min_symbol, static_cast<int64_t>(symbol_values[i]));
            max_symbol = std::max(max_symbol, static_cast<int64_t>(symbol_values[i]));
        }
        return std::tuple{symbol_index_lookup, min_symbol, max_symbol};
    }();

    // Decide whether to use a lookup array
    const bool use_lookup_array = (max_symbol - min_symbol + 1) <= lookup_array_threshold;
    std::vector<size_t> symbol_index_lookup_array(0);
    if (use_lookup_array)
    {
        symbol_index_lookup_array.resize(max_symbol - min_symbol + 1,
                                         std::numeric_limits<size_t>::max());

        for (size_t i = 0; i < num_symbols; ++i)
        {
            symbol_index_lookup_array[symbol_values[i] - min_symbol] = i;
        }
    }

    // Initialize state and packed bitstream
    uint32_t state = L;
    std::vector<uint64_t> bitstream((signal_size * 32 + 63) / 64, 0);  // Preallocate worst case
    size_t num_bits = 0;

    const auto encode_symbol = [&](const size_t s_ind)
    {
        uint32_t state_normalized = state;
        const uint32_t L_s = symbol_counts[s_ind];
        auto iterations =0;
        // Otherwise, perform normalization.
        while (state_normalized >= 2 * L_s)
        {
            const size_t word_idx = num_bits >> 6;  // Divide by 64
            const size_t bit_idx  = num_bits & 63;   // Modulo 64
            bitstream[word_idx] |= static_cast<uint64_t>(state_normalized & 1) << bit_idx;
            ++num_bits;
            state_normalized >>= 1;
            iterations ++;
        }
        // Update state after normalization.
        state = L + C[s_ind] + state_normalized - L_s;
    };

    constexpr auto block_size = 512/sizeof(T);
    // Fixed-size storage for each symbol in the block.
    std::array<int, block_size> k{};             // Number of normalization iterations per symbol.
    std::array<uint32_t, block_size> normalized{}; // Final normalized state for each symbol.

    // Use a vector for each symbol to hold normalization bits (to avoid a fixed MAX_K).
    std::array<std::vector<uint64_t>, block_size> norm_bits{};
    auto encode_block = [&](const std::array<size_t, block_size>& symbol_indices) {

        // Carry dependency: state_block holds the state as it is updated symbol by symbol.
        uint32_t state_block = state;

        // First pass: Process each symbol in the block.
        for (size_t i = 0; i < block_size; ++i) {
            size_t s_ind = symbol_indices[i];
            const uint32_t L_s = symbol_counts[s_ind];
            uint32_t current = state_block;
            int j = 0;

            // While the state is too high, extract bits and update 'current'.
            while (current >= 2 * L_s) {
                norm_bits[i].push_back(current & 1); // Save the least-significant bit.
                current >>= 1;
                ++j;
            }
            k[i] = j;
            normalized[i] = current;

            // Update the state for the next symbol:
            // ANS update: state = L + C[s_ind] + current - L_s.
            state_block = L + C[s_ind] + current - L_s;
        }

        // Update the global state after processing the block.
        state = state_block;

        // Second pass: Write out the precomputed normalization bits into the bitstream.
        for (size_t i = 0; i < block_size; ++i) {
            for (size_t j = 0; j < norm_bits[i].size(); ++j) {
                size_t word_idx = num_bits >> 6;  // Divide by 64.
                size_t bit_idx  = num_bits & 63;   // Modulo 64.
                bitstream[word_idx] |= norm_bits[i][j] << bit_idx;
                ++num_bits;
            }
        }

        for (auto& v : norm_bits) {
            v.clear();
        }
    };

    // Encode each symbol using the appropriate lookup method
    if (use_lookup_array)
    {
        for (size_t i = 0; i < signal_size; ++i)
        {
            const int64_t lookup_ind = signal[i] - min_symbol;
            if (lookup_ind < 0 || lookup_ind >= static_cast<int64_t>(symbol_index_lookup_array.size()))
            {
                throw std::invalid_argument("Signal value not found in symbol_values");
            }
            const size_t s_ind = symbol_index_lookup_array[lookup_ind];
            if (s_ind == std::numeric_limits<size_t>::max())
            {
                throw std::invalid_argument("Signal value not found in symbol_values");
            }
            // Optionally, you can assert that s_ind matches the map lookup:
            // assert(s_ind == symbol_index_lookup[signal[i]]);
            encode_symbol(s_ind);
        }
    }
    else
    {
        const auto pow2_size = signal_size & -block_size;
        std::array<std::size_t, block_size> symbol_indices{};
        for (size_t i = 0; i < pow2_size; i += block_size)
        {
            for (size_t j = 0; j < block_size; ++j)
            {
                const auto it = symbol_index_lookup.find(signal[i+j]);
                if (it == symbol_index_lookup.end())
                {
                    throw std::invalid_argument("Signal value not found in symbol_values");
                }
                symbol_indices[j] = it->second;
            }
            encode_block(symbol_indices);
        }
        for (size_t i = 0; i < signal_size; ++i)
        {
            auto it = symbol_index_lookup.find(signal[i]);
            if (it == symbol_index_lookup.end())
            {
            throw std::invalid_argument("Signal value not found in symbol_values");
            }
            encode_symbol(it->second);
        }
    }

    // Truncate bitstream to the actual size used
    size_t final_words = (num_bits + 63) / 64;
    bitstream.resize(final_words);

    return {state, std::move(bitstream), num_bits};
}

template <typename T>
void ans_decode_t(T* output,
                  size_t n,
                  uint32_t state,
                  const uint64_t* bitstream,
                  size_t num_bits,
                  const uint32_t* symbol_counts,
                  const T* symbol_values,
                  size_t num_symbols)
{
    // Calculate L and verify it's a power of 2
    uint32_t L = 0;
    for (size_t i = 0; i < num_symbols; ++i)
    {
        L += symbol_counts[i];
    }
    if (!is_power_of_2(L))
    {
        throw std::invalid_argument("L must be a power of 2");
    }

    // Pre-compute cumulative sums
    std::vector<uint32_t> C(num_symbols);
    C[0] = 0;
    for (size_t i = 1; i < num_symbols; ++i)
    {
        C[i] = C[i - 1] + symbol_counts[i - 1];
    }

    // Create symbol lookup table
    std::vector<uint32_t> symbol_lookup(L);
    for (size_t s = 0; s < num_symbols; ++s)
    {
        for (uint32_t j = 0; j < symbol_counts[s]; ++j)
        {
            symbol_lookup[C[s] + j] = s;
        }
    }

    // Create state update table
    std::vector<uint32_t> state_update(L);
    for (uint32_t i = 0; i < L; ++i)
    {
        uint32_t s = symbol_lookup[i];
        uint32_t f_s = symbol_counts[s];
        state_update[i] = f_s + i - C[s];
    }

    // Create bit count table
    uint32_t max_f_s = 0;
    for (size_t s = 0; s < num_symbols; ++s)
    {
        max_f_s = std::max(max_f_s, symbol_counts[s]);
    }
    std::vector<uint32_t> bit_count_table(2 * max_f_s);
    for (uint32_t i = 1; i < 2 * max_f_s; ++i)
    {
        uint32_t d = 0;
        while ((i << d) < L)
        {
            d++;
        }
        bit_count_table[i] = d;
    }

    // Prepare bit reading
    auto bit_pos = static_cast<int64_t>(num_bits - 1);

    // Decode symbols in reverse order
    for (size_t i = 0; i < n; ++i)
    {
        uint32_t s_ind = symbol_lookup[state - L];
        output[n - 1 - i] = symbol_values[s_ind];

        uint32_t state_2 = state_update[state - L];
        uint32_t d = bit_count_table[state_2];
        uint32_t new_state = state_2 << d;

        // Read d bits from bitstream
        if (d > 0)
        {
            read_bits_from_end_of_bitstream(bitstream, bit_pos, new_state, 0, d);
        }
        bit_pos -= d;
        state = new_state;
    }
}

}  // namespace simple_ans


#undef VECTOR_WIDTH