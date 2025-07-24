#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>
#include <unordered_map>

#include "libdivide.h"

namespace simple_ans
{

struct EncodedData
{
    uint64_t state;
    std::vector<uint32_t> words;
};

// Helper function to verify if a number is a power of 2
inline bool is_power_of_2(uint32_t x)
{
    return x && !(x & (x - 1));
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
                  uint64_t state,
                  const uint32_t* words,
                  size_t num_words,
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

constexpr int STATE_BITS = 64;
constexpr int WORD_BITS = 32;
constexpr uint64_t THRESHOLD = 1ULL << (STATE_BITS - WORD_BITS);
constexpr uint64_t MASK_WORD = (1ULL << WORD_BITS) - 1;

template <typename T>
std::tuple<std::vector<T>, std::vector<uint64_t>> unique_with_counts(const T* values, size_t n)
{
    // WARNING: This is ONLY a helper function. It doesn't support arrays with a large domain, and will instead fail
    // return empty vectors. It is up to the caller to handle this case separately. numpy.unique() is quite fast, with
    // improvements to use vectorized sorts (in 2.x, at least), so I didn't bother to implement a more efficient version here.
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
            raw_counts[values[i] - min_value]++;
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
    uint32_t index_size = 0;
    for (size_t i = 0; i < num_symbols; ++i)
    {
        index_size += symbol_counts[i];
    }
    if (!is_power_of_2(index_size))
    {
        throw std::invalid_argument("L must be a power of 2");
    }

    int PRECISION_BITS = 0;
    while ((1U << PRECISION_BITS) < index_size)
    {
        PRECISION_BITS++;
    }

    // Pre-compute cumulative sums
    std::vector<uint32_t> C(num_symbols);
    C[0] = 0;
    for (size_t i = 1; i < num_symbols; ++i)
    {
        C[i] = C[i - 1] + symbol_counts[i - 1];
    }

    // Precompute libdivide dividers for each symbol count
    std::vector<libdivide::divider<uint64_t>> fast_dividers(num_symbols);
    for (size_t i = 0; i < num_symbols; ++i)
    {
        fast_dividers[i] = libdivide::divider<uint64_t>(symbol_counts[i]);
    }

    // Create symbol index lookup
    std::unordered_map<T, size_t> symbol_index_lookup;
    int64_t min_symbol = symbol_values[0];
    int64_t max_symbol = symbol_values[0];
    for (size_t i = 0; i < num_symbols; ++i)
    {
        symbol_index_lookup[symbol_values[i]] = i;
        min_symbol = std::min(min_symbol, static_cast<int64_t>(symbol_values[i]));
        max_symbol = std::max(max_symbol, static_cast<int64_t>(symbol_values[i]));
    }

    // Map lookups can be a bottleneck, so we use a lookup array if the number of symbols is "small"
    const bool use_lookup_array = (max_symbol - min_symbol + 1) <= lookup_array_threshold;
    std::vector<size_t> symbol_index_lookup_array;
    if (use_lookup_array)
    {
        symbol_index_lookup_array.resize(max_symbol - min_symbol + 1);

        std::fill(symbol_index_lookup_array.begin(),
                  symbol_index_lookup_array.end(),
                  std::numeric_limits<size_t>::max());

        for (size_t i = 0; i < num_symbols; ++i)
        {
            symbol_index_lookup_array[symbol_values[i] - min_symbol] = i;
        }
    }

    // Initialize state and words
    uint64_t state = 0;
    std::vector<uint32_t> words; // Use dynamic allocation instead of preallocating
    words.reserve(signal_size / 8); // Reserve a reasonable estimate to avoid frequent reallocations

    // Encode each symbol
    auto SHIFT = STATE_BITS - PRECISION_BITS;

    for (size_t i = 0; i < signal_size; ++i)
    {
        // Symbol index lookup
        size_t s_ind;
        if (use_lookup_array)
        {
            const int64_t lookup_ind = signal[i] - min_symbol;
            if (lookup_ind < 0 || lookup_ind >= lookup_array_threshold)
            {
                throw std::invalid_argument("Signal value not found in symbol_values");
            }
            s_ind = symbol_index_lookup_array[lookup_ind];
            if (s_ind == std::numeric_limits<size_t>::max())
            {
                throw std::invalid_argument("Signal value not found in symbol_values");
            }
            assert(s_ind == symbol_index_lookup[signal[i]]);
        }
        else
        {
            auto it = symbol_index_lookup.find(signal[i]);
            if (it == symbol_index_lookup.end())
            {
                throw std::invalid_argument("Signal value not found in symbol_values");
            }
            s_ind = it->second;
        }

        // Cache frequently accessed symbol data to avoid repeated array lookups
        const uint32_t F_s = symbol_counts[s_ind];
        const uint32_t C_s = C[s_ind];
        const auto& divider = fast_dividers[s_ind];

        // Check if we need to normalize
        if ((state >> SHIFT) >= F_s)
        {
            words.push_back(state & MASK_WORD);
            state >>= WORD_BITS;
        }

        // Update state using libdivide for faster division
        const uint64_t prefix = state / divider;
        const uint64_t remainder = state - prefix * F_s;
        state = (prefix << PRECISION_BITS) | (C_s + remainder);
    }

    return {state, std::move(words)};
}

template <typename T>
void ans_decode_t(T* output,
                  size_t n,
                  uint64_t state,
                  const uint32_t* words,
                  size_t num_words,
                  const uint32_t* symbol_counts,
                  const T* symbol_values,
                  size_t num_symbols)
{
    // very important that this is signed, because it becomes -1
    int32_t word_idx = num_words - 1;
    // Calculate index size and verify it's a power of 2
    uint32_t index_size = 0;
    for (size_t i = 0; i < num_symbols; ++i)
    {
        index_size += symbol_counts[i];
    }
    if (!is_power_of_2(index_size))
    {
        throw std::invalid_argument("L must be a power of 2");
    }

    int PRECISION_BITS = 0;
    while ((1U << PRECISION_BITS) < index_size)
    {
        PRECISION_BITS++;
    }

    // Pre-compute cumulative sums
    std::vector<uint32_t> C(num_symbols);
    C[0] = 0;
    for (size_t i = 1; i < num_symbols; ++i)
    {
        C[i] = C[i - 1] + symbol_counts[i - 1];
    }

    // Create symbol lookup table
    std::vector<uint32_t> symbol_lookup(index_size);
    for (size_t s = 0; s < num_symbols; ++s)
    {
        for (uint32_t j = 0; j < symbol_counts[s]; ++j)
        {
            symbol_lookup[C[s] + j] = s;
        }
    }

    // Decode symbols in reverse order
    for (size_t i = 0; i < n; ++i)
    {
        const uint64_t prefix = state >> PRECISION_BITS;
        const uint64_t quantile = state & ((1U << PRECISION_BITS) - 1);
        const uint32_t s_ind = symbol_lookup[quantile];

        // Cache frequently accessed symbol data to avoid repeated array lookups
        const uint32_t F_s = symbol_counts[s_ind];
        const uint32_t C_s = C[s_ind];
        const T symbol_value = symbol_values[s_ind];

        uint64_t previous_state = prefix * F_s + quantile - C_s;

        if (previous_state < THRESHOLD && word_idx >= 0)
        {
            const uint32_t emit_word = words[word_idx];
            word_idx--;
            previous_state = (previous_state << WORD_BITS) | emit_word;
        }

        state = previous_state;
        output[n - 1 - i] = symbol_value;
    }
}

}  // namespace simple_ans
