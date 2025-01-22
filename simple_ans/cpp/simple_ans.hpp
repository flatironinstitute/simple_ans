#pragma once

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

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

inline void read_bits_from_end_of_bitstream(const uint64_t* bitstream,
                                            int64_t& source_bit_position,
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
        uint32_t bits =
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

    // Create symbol index lookup
    std::unordered_map<T, size_t> symbol_index_lookup;
    for (size_t i = 0; i < num_symbols; ++i)
    {
        symbol_index_lookup[symbol_values[i]] = i;
    }

    // Initialize state and packed bitstream
    uint32_t state = L;
    std::vector<uint64_t> bitstream(
        (signal_size * 32 + 63) / 64,
        0);  // Preallocate worst case (todo: is this the correct worst case?)
    size_t num_bits = 0;

    // Encode each symbol
    for (size_t i = 0; i < signal_size; ++i)
    {
        auto it = symbol_index_lookup.find(signal[i]);
        if (it == symbol_index_lookup.end())
        {
            throw std::invalid_argument("Signal value not found in symbol_values");
        }
        size_t s_ind = it->second;

        uint32_t state_normalized = state;
        const uint32_t L_s = symbol_counts[s_ind];

        // Normalize state
        // we need state_normalized to be in the range [L_s, 2*L_s)
        while (state_normalized >= 2 * L_s)
        {
            // Add bit to packed format
            size_t word_idx = num_bits >> 6;  // Divide by 64
            size_t bit_idx = num_bits & 63;   // Modulo 64
            bitstream[word_idx] |= static_cast<uint64_t>(state_normalized & 1) << bit_idx;
            num_bits++;
            state_normalized >>= 1;
        }

        // Update state
        state = L + C[s_ind] + state_normalized - L_s;
    }

    // Truncate bitstream to actual size used
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
    int64_t bit_pos = num_bits - 1;

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
