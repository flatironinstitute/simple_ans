#pragma once

#include "utils.hpp"

#include <bit>
#include <cassert>
#include <limits>
#include <stdexcept>

#include <hash_table7.hpp>

namespace simple_ans
{

struct EncodedData
{
    uint32_t state;
    utils::AlignedVector<uint64_t>
        bitstream;    // Each uint64_t contains 64 bits, with padding in last word if needed
    size_t num_bits;  // Actual number of bits used (may be less than bitstream.size() * 64)
};

// Helper function to verify if a number is a power of 2
inline bool is_power_of_2(uint32_t x)
{
    return std::has_single_bit(x);
}

// require the type to be numeric
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
T vector_accumulate(const T* data, const size_t n, const T init = T(0))
{
    constexpr static auto block_size = utils::VECTOR_WIDTH / sizeof(T);
    alignas(utils::VECTOR_WIDTH) std::array<T, block_size> sums{T(0)};
    const auto pow2_size = n & -block_size;
    size_t i = 0;
    for (; i < pow2_size; i += block_size)
    {
#pragma GCC ivdep
#pragma clang loop vectorize(enable)
#pragma omp simd
        for (uint_fast8_t j = 0; j < block_size; ++j)
        {
            sums[j] += data[i + j];
        }
    }
    return std::accumulate(sums.begin(), sums.end(), std::accumulate(data + i, data + n, init));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
void vector_inclusive_scan(const T* input, T* aligned_output, const size_t num_symbols)
{
    // assume aligned output
#if defined(__GNUC__) || defined(__clang__)
    aligned_output = static_cast<T*>(__builtin_assume_aligned(aligned_output, utils::VECTOR_WIDTH));
#endif
    static constexpr auto block_size = utils::VECTOR_WIDTH * 2 / sizeof(T);
    // Set the first element to 0
    aligned_output[0] = 0;
    // Let n be the number of elements in the input array.
    const auto n = num_symbols - 1;

    // Process blocks of size N.

    static constexpr auto Half = block_size / 2;
    alignas(utils::VECTOR_WIDTH) std::array<T, Half> temp{};

    const auto pow2_size = n & -block_size;  // largest multiple of N less than or equal to n

    size_t i = 0;
    for (T s = 0; i < pow2_size; i += block_size)  // s: running sum from previous blocks
    {
        // --- Process first lane (elements i ... i+Half-1) ---
        // Write the first prefix element: output[i+1] = input[i] + running sum.
        aligned_output[i + 1] = input[i] + s;
#pragma GCC ivdep
#pragma clang loop vectorize(enable)
#pragma omp simd
        for (uint_fast8_t j = 1; j < Half; ++j)
        {
            // Each output element is the sum of the previous output and the current input.
            aligned_output[i + 1 + j] = aligned_output[i + j] + input[i + j];
        }
        // Save the last value of lane 0 as the local block sum.
        const T lane0_sum = aligned_output[i + Half];

        // --- Process second lane (elements i+Half ... i+N-1) ---

        temp[0] = input[i + Half];
#pragma GCC ivdep
#pragma clang loop vectorize(enable)
#pragma omp simd
        for (uint_fast8_t j = 1; j < Half; ++j)
        {
            temp[j] = temp[j - 1] + input[i + Half + j];
        }
        // Add the lane0_sum to the local prefix values and write them out.
        for (size_t j = 0; j < Half; ++j)
        {
            aligned_output[i + 1 + Half + j] = temp[j] + lane0_sum;
        }
        // Update the running sum to be the last element computed in this block.
        s = aligned_output[i + block_size];
    }

    // --- Process any remaining elements with a simple scalar loop ---
    for (; i < n; i++)
    {
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
constexpr auto unique_array_threshold = static_cast<int>(std::numeric_limits<uint16_t>::max()) + 1;
constexpr auto lookup_array_threshold = unique_array_threshold;

template <typename T>
std::tuple<std::vector<T>, std::vector<uint64_t>> unique_with_counts(const T* values,
                                                                     const size_t n)
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
            raw_counts[values[i] - min_value] += 1;
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
    const auto d = dest_end_bit - dest_start_bit;
    if ((source_bit_position & 63) >= (d - 1)) [[likely]]  // Branch prediction hint
    {
        // Compute word and bit positions efficiently
        const auto word_idx = source_bit_position >> 6;  // Divide by 64
        const auto bit_idx = source_bit_position & 63;   // Modulo 64

        // Compute mask and extract bits in one step
        const auto mask = (1U << d) - 1;  // Precompute mask
        const auto bits = static_cast<uint32_t>((bitstream[word_idx] >> (bit_idx - d + 1)) & mask);

        // Store result
        dest |= (bits << dest_start_bit);
    }
    else
    {
        // this is possibly the slower case, but should be less common
        for (uint_fast32_t j = 0; j < d; ++j)
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
    utils::AlignedVector<uint32_t> C(num_symbols);
    vector_inclusive_scan(symbol_counts, C.data(), num_symbols);

    // Create symbol index lookup (for fallback)
    const auto [symbol_index_lookup, min_symbol, max_symbol] = [symbol_values, num_symbols]
    {
        emhash7::HashMap<T, size_t> symbol_index_lookup{};
        int64_t min_symbol = symbol_values[0];
        int64_t max_symbol = symbol_values[0];
#pragma GCC ivdep
#pragma clang loop vectorize(enable)
#pragma omp simd
        for (size_t i = 0; i < num_symbols; ++i)
        {
            symbol_index_lookup[symbol_values[i]] = i;
            min_symbol = std::min(min_symbol, static_cast<int64_t>(symbol_values[i]));
            max_symbol = std::max(max_symbol, static_cast<int64_t>(symbol_values[i]));
        }
        return std::tuple{
            std::move(symbol_index_lookup), std::move(min_symbol), std::move(max_symbol)};
    }();

    // Decide whether to use a lookup array
    const bool use_lookup_array = (max_symbol - min_symbol + 1) <= lookup_array_threshold;
    const auto symbol_index_lookup_array = [&]()
    {
        if (use_lookup_array)
        {
            utils::AlignedVector<size_t> lookup_array(max_symbol - min_symbol + 1,
                                                      std::numeric_limits<size_t>::max());
            for (size_t i = 0; i < num_symbols; ++i)
            {
                lookup_array[symbol_values[i] - min_symbol] = i;
            }
            return lookup_array;
        }
        return utils::AlignedVector<size_t>(0);
    }();

    // Initialize state and packed bitstream
    utils::AlignedVector<uint64_t> bitstream((signal_size * 32 + 63) / 64,
                                             0);  // Preallocate worst case
    auto state = L;
    size_t num_bits = 0;

    const auto encode_symbol = [&](const size_t s_ind) constexpr noexcept
    {
        auto state_normalized = state;
        const auto L_s = symbol_counts[s_ind];
        // Otherwise, perform normalization.
        while (state_normalized >= 2 * L_s)
        {
            const auto word_idx = num_bits >> 6;  // Divide by 64
            const auto bit_idx = num_bits & 63;   // Modulo 64
            bitstream[word_idx] |= static_cast<uint64_t>(state_normalized & 1) << bit_idx;
            ++num_bits;
            state_normalized >>= 1;
        }
        // Update state after normalization.
        state = L + C[s_ind] + state_normalized - L_s;
    };

    // Encode each symbol using the appropriate lookup method
    if (use_lookup_array)
    {
        for (size_t i = 0; i < signal_size; ++i)
        {

            const int64_t lookup_ind = signal[i] - min_symbol;
            if (lookup_ind < 0 ||
                lookup_ind >= static_cast<int64_t>(symbol_index_lookup_array.size())) [[unlikely]]
            {
                throw std::invalid_argument("Signal value not found in symbol_values");
            }

            const size_t s_ind = symbol_index_lookup_array[lookup_ind];
            if (s_ind == std::numeric_limits<size_t>::max()) [[unlikely]]
            {
                throw std::invalid_argument("Signal value not found in symbol_values");
            }

            encode_symbol(s_ind);
        }
    }
    else
    {
        for (size_t i = 0; i < signal_size; ++i)
        {
            const auto it = symbol_index_lookup.find(signal[i]);
            if (it == symbol_index_lookup.end()) [[unlikely]]
            {
                throw std::invalid_argument("Signal value not found in symbol_values");
            }
            encode_symbol(it->second);
        }
    }

    // Truncate bitstream to the actual size used
    const auto final_words = (num_bits + 63) / 64;
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
    const auto L = vector_accumulate(symbol_counts, num_symbols);
    if (!is_power_of_2(L))
    {
        throw std::invalid_argument("L must be a power of 2");
    }

    // Pre-compute cumulative sums
    utils::AlignedVector<uint32_t> C(num_symbols);
    vector_inclusive_scan(symbol_counts, C.data(), num_symbols);

    // Create symbol lookup table
    utils::AlignedVector<uint32_t> symbol_lookup(L);
    for (size_t s = 0; s < num_symbols; ++s)
    {
        for (uint32_t j = 0; j < symbol_counts[s]; ++j)
        {
            symbol_lookup[C[s] + j] = s;
        }
    }

    // Create state update table
    utils::AlignedVector<uint32_t> state_update(L);
    for (uint_fast32_t i = 0; i < L; ++i)
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
    utils::AlignedVector<uint32_t> bit_count_table(2 * max_f_s);

    for (uint_fast32_t i = 1; i < 2 * max_f_s; ++i)
    {
        bit_count_table[i] = std::bit_width((L - 1) / i);
    }

    // Prepare bit reading
    auto bit_pos = static_cast<int64_t>(num_bits - 1);

    // Decode symbols in reverse order
    for (size_t i = 0; i < n; ++i)
    {
        const auto s_ind = symbol_lookup[state - L];
        output[n - 1 - i] = symbol_values[s_ind];

        const auto state_2 = state_update[state - L];
        const auto d = bit_count_table[state_2];
        auto new_state = state_2 << d;

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
