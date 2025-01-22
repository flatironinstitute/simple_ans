#pragma once

#include <chrono>
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

// Template version for different integer types
template <typename T>
EncodedData encode_t(const T* signal,
                     size_t signal_size,
                     const uint32_t* symbol_counts,
                     const T* symbol_values,
                     size_t num_symbols);

template <typename T>
EncodedData ans_encode_t(
    const T* signal,
    size_t signal_size,
    const uint32_t* symbol_counts,
    const T* symbol_values,
    size_t num_symbols);

template <typename T>
void decode_t(T* output,
              size_t n,
              uint32_t state,
              const uint64_t* bitstream,
              size_t num_bits,
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

// Legacy int32_t versions for backward compatibility
inline EncodedData encode(const int32_t* signal,
                          size_t signal_size,
                          const uint32_t* symbol_counts,
                          const int32_t* symbol_values,
                          size_t num_symbols)
{
    return encode_t(signal, signal_size, symbol_counts, symbol_values, num_symbols);
}

inline void decode(int32_t* output,
                   size_t n,
                   uint32_t state,
                   const uint64_t* bitstream,
                   size_t num_bits,
                   const uint32_t* symbol_counts,
                   const int32_t* symbol_values,
                   size_t num_symbols)
{
    decode_t(output, n, state, bitstream, num_bits, symbol_counts, symbol_values, num_symbols);
}

// Convert real-valued proportions into integer counts summing to L
// proportions: array of real-valued proportions
// num_proportions: length of proportions array
// counts_out: pre-allocated buffer to store resulting counts (must be size num_proportions)
void choose_symbol_counts(uint32_t* counts_out,
                          const double* proportions,
                          size_t num_proportions,
                          uint32_t L);

}  // namespace simple_ans

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Template Implementation
////////////////////////////////////////////////////////////////////////////////

namespace simple_ans
{

template <typename T>
EncodedData encode_t(const T* signal,
                     size_t signal_size,
                     const uint32_t* symbol_counts,
                     const T* symbol_values,
                     size_t num_symbols)
{
    // Calculate L (sum of symbol counts) and verify it's a power of 2
    uint32_t L = 0;
    for (size_t i = 0; i < num_symbols; ++i)
    {
        L += symbol_counts[i];
    }

    if (!is_power_of_2(L))
    {
        throw std::invalid_argument("L must be a power of 2");
    }

    // Pre-compute cumulative sums for efficiency
    std::vector<uint32_t> cumsum(num_symbols + 1, 0);
    for (size_t i = 0; i < num_symbols; ++i)
    {
        cumsum[i + 1] = cumsum[i] + symbol_counts[i];
    }

    // Create mapping from values to indices
    std::unordered_map<T, size_t> symbol_index_for_value;
    for (size_t i = 0; i < num_symbols; ++i)
    {
        symbol_index_for_value[symbol_values[i]] = i;
    }

    // Initialize state and packed bitstream
    uint32_t state = L;
    // Preallocate bitstream - in worst case, each symbol needs log2(2L) bits
    // Convert to 64-bit words by dividing by 64 and rounding up
    size_t max_bits = signal_size * (32 - __builtin_clz(2 * L - 1));
    size_t num_words = (max_bits + 63) / 64;
    std::vector<uint64_t> bitstream(num_words, 0);
    size_t num_bits = 0;

    // Encode each symbol
    for (size_t i = 0; i < signal_size; ++i)
    {
        auto it = symbol_index_for_value.find(signal[i]);
        if (it == symbol_index_for_value.end())
        {
            throw std::invalid_argument("Signal value not found in symbol_values");
        }
        const auto s = it->second;
        uint32_t state_normalized = state;
        const uint32_t L_s = symbol_counts[s];

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
        state = L + cumsum[s] + state_normalized - L_s;
    }

    // Truncate bitstream to actual size used
    size_t final_words = (num_bits + 63) / 64;
    bitstream.resize(final_words);
    return {state, std::move(bitstream), num_bits};
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
    for (size_t i = 0; i < num_symbols; ++i) {
        L += symbol_counts[i];
    }
    if (!is_power_of_2(L)) {
        throw std::invalid_argument("L must be a power of 2");
    }

    // Pre-compute cumulative sums
    std::vector<uint32_t> C(num_symbols);
    C[0] = 0;
    for (size_t i = 1; i < num_symbols; ++i) {
        C[i] = C[i-1] + symbol_counts[i-1];
    }

    // Create bit shift table
    std::vector<std::vector<uint32_t>> bit_shift_table(L, std::vector<uint32_t>(num_symbols));
    for (uint32_t i = 0; i < L; ++i) {
        for (size_t s = 0; s < num_symbols; ++s) {
            uint32_t f_s = symbol_counts[s];
            uint32_t d = 0;
            while ((L + i) / (1u << d) >= 2 * f_s) {
                d++;
            }
            bit_shift_table[i][s] = d;
        }
    }

    // Create state transition table
    std::vector<std::vector<uint32_t>> state_transition_table(L, std::vector<uint32_t>(num_symbols));
    for (uint32_t i = 0; i < L; ++i) {
        for (size_t s = 0; s < num_symbols; ++s) {
            uint32_t f_s = symbol_counts[s];
            uint32_t d = bit_shift_table[i][s];
            state_transition_table[i][s] = L + C[s] + ((L + i) / (1u << d) - f_s);
        }
    }

    // Create symbol index lookup
    std::unordered_map<T, size_t> symbol_index_lookup;
    for (size_t i = 0; i < num_symbols; ++i) {
        symbol_index_lookup[symbol_values[i]] = i;
    }

    // Initialize state and packed bitstream
    uint32_t state = L;
    std::vector<uint64_t> bitstream((signal_size * 32 + 63) / 64, 0); // Preallocate worst case
    size_t num_bits = 0;

    // Encode each symbol
    for (size_t i = 0; i < signal_size; ++i) {
        auto it = symbol_index_lookup.find(signal[i]);
        if (it == symbol_index_lookup.end()) {
            throw std::invalid_argument("Signal value not found in symbol_values");
        }
        size_t s_ind = it->second;

        // Get number of bits to shift
        uint32_t d = bit_shift_table[state - L][s_ind];

        // Add bits to bitstream
        // if ((num_bits & 63) >= (d - 1)) {
        if (false) {
            // can't figure this out!
        } else {
            // this is possibly the slower case, but should be less common
            for (uint32_t j = 0; j < d; ++j) {
                size_t word_idx = (num_bits + j) >> 6;  // Divide by 64
                size_t bit_idx = (num_bits + j) & 63;   // Modulo 64
                bitstream[word_idx] |= static_cast<uint64_t>((state >> (j)) & 1) << bit_idx;
            }
        }
        num_bits += d;

        // Update state
        state = state_transition_table[state - L][s_ind];
    }

    // Truncate bitstream to actual size used
    size_t final_words = (num_bits + 63) / 64;
    bitstream.resize(final_words);
    return {state, std::move(bitstream), num_bits};
}

template <typename T>
void decode_t(T* output,
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

    if (state < L || state >= 2 * L)
    {
        throw std::invalid_argument("Initial state is invalid");
    }

    // Pre-compute lookup table for symbol finding
    std::vector<uint32_t> symbol_lookup(L);
    std::vector<uint32_t> cumsum(num_symbols + 1, 0);

    // Build cumulative sums and lookup table
    for (size_t i = 0; i < num_symbols; ++i)
    {
        cumsum[i + 1] = cumsum[i] + symbol_counts[i];
        for (uint32_t j = cumsum[i]; j < cumsum[i + 1]; ++j)
        {
            symbol_lookup[j] = static_cast<uint32_t>(i);
        }
    }

    // Prepare num_bytes_needed_lookup
    // For each i, it's the smallest number n such that i * 2^n >= L
    std::vector<uint32_t> num_bits_needed_lookup(L);
    uint32_t X = 32 - __builtin_clz(L);  // log2(L)
    for (uint32_t n = 1; n <= X; ++n)
    {
        uint32_t L_0 = L >> n;
        uint32_t L_1 = L >> (n - 1);
        for (uint32_t i = L_0; i < L_1; ++i)
        {
            num_bits_needed_lookup[i] = n;
        }
    }

    // Prepare bit reading
    int64_t bit_pos = num_bits - 1;
    const uint32_t L_mask = L - 1;  // For fast modulo since L is power of 2

    // Decode symbols in reverse order
    for (size_t i = 0; i < n; ++i)
    {
        // Find symbol using lookup table instead of binary search
        uint32_t remainder = state & L_mask;  // Fast modulo for power of 2
        uint32_t s = symbol_lookup[remainder];

        // Calculate normalized state which will be in the range [L_s, 2*L_s)
        uint32_t L_s = symbol_counts[s];
        state = L_s + state - L - cumsum[s];

        // read from the bit stream and get us back to the range [L, 2L)
        // determine how many bits need to be read
        uint32_t num_bits_needed = num_bits_needed_lookup[state];
        if (num_bits_needed > 0) {
            if ((bit_pos & 63) >= (num_bits_needed - 1)) {
                // in this case we can grab all the bits we need at once from the current word
                // note: it wasn't clear that this trick was making a difference in performance
                uint32_t word_idx = bit_pos >> 6;  // Divide by 64
                uint32_t bit_idx = bit_pos & 63;   // Modulo 64
                // get bits from bit_idx - num_bits_needed + 1 to bit_idx
                uint32_t bits = static_cast<uint32_t>((bitstream[word_idx] >> (bit_idx - num_bits_needed + 1)) & ((1 << num_bits_needed) - 1));
                state = (state << num_bits_needed) | bits;
                bit_pos -= num_bits_needed;
            }
            else {
                // this is possibly the slower case, but should be less common
                for (uint32_t j = 0; j < num_bits_needed; ++j)
                {
                    uint32_t word_idx = bit_pos >> 6;  // Divide by 64
                    uint32_t bit_idx = bit_pos & 63;   // Modulo 64
                    state = (state << 1) | ((bitstream[word_idx] >> bit_idx) & 1);
                    bit_pos--;
                }
            }
        }

        output[n - 1 - i] = symbol_values[s];
    }
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
    auto start_total = std::chrono::high_resolution_clock::now();
    auto start_setup = start_total;
    // Calculate L and verify it's a power of 2
    uint32_t L = 0;
    for (size_t i = 0; i < num_symbols; ++i) {
        L += symbol_counts[i];
    }
    if (!is_power_of_2(L)) {
        throw std::invalid_argument("L must be a power of 2");
    }

    // Pre-compute cumulative sums
    std::vector<uint32_t> C(num_symbols);
    C[0] = 0;
    for (size_t i = 1; i < num_symbols; ++i) {
        C[i] = C[i-1] + symbol_counts[i-1];
    }

    auto end_setup = std::chrono::high_resolution_clock::now();
    auto start_tables = end_setup;

    // Create symbol lookup table
    std::vector<uint32_t> symbol_lookup(L);
    for (size_t s = 0; s < num_symbols; ++s) {
        for (uint32_t j = 0; j < symbol_counts[s]; ++j) {
            symbol_lookup[C[s] + j] = s;
        }
    }

    // Create state update table
    std::vector<uint32_t> state_update(L);
    for (uint32_t i = 0; i < L; ++i) {
        uint32_t s = symbol_lookup[i];
        uint32_t f_s = symbol_counts[s];
        state_update[i] = f_s + i - C[s];
    }

    // Create bit count table
    uint32_t max_f_s = 0;
    for (size_t s = 0; s < num_symbols; ++s) {
        max_f_s = std::max(max_f_s, symbol_counts[s]);
    }
    std::vector<uint32_t> bit_count_table(2 * max_f_s);
    for (uint32_t i = 1; i < 2 * max_f_s; ++i) {
        uint32_t d = 0;
        while (i * (1u << d) < L) {
            d++;
        }
        bit_count_table[i] = d;
    }

    // Prepare bit reading
    int64_t bit_pos = num_bits - 1;

    auto end_tables = std::chrono::high_resolution_clock::now();
    auto start_decode = end_tables;

    // Decode symbols in reverse order
    size_t fast_path = 0;
    size_t slow_path = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t s_ind = symbol_lookup[state - L];
        output[n - 1 - i] = symbol_values[s_ind];

        uint32_t state_2 = state_update[state - L];
        uint32_t d = bit_count_table[state_2];
        uint32_t new_state = state_2 << d;

        // Read d bits from bitstream
        if ((bit_pos & 63) >= (d - 1)) {
            // in this case we can grab all the bits we need at once from the current word
            uint32_t word_idx = bit_pos >> 6;  // Divide by 64
            uint32_t bit_idx = bit_pos & 63;   // Modulo 64
            // get bits from bit_idx - d + 1 to bit_idx
            uint32_t bits = static_cast<uint32_t>((bitstream[word_idx] >> (bit_idx - d + 1)) & ((1 << d) - 1));
            new_state |= bits;
            bit_pos -= d;
            fast_path++;
        } else {
            // this is possibly the slower case, but should be less common
            for (uint32_t j = 0; j < d; ++j)
            {
                uint32_t word_idx = (bit_pos - j) >> 6;  // Divide by 64
                uint32_t bit_idx = (bit_pos - j) & 63;   // Modulo 64
                new_state |= (static_cast<uint32_t>((bitstream[word_idx] >> bit_idx) & 1) << (d - 1 - j));
            }
            bit_pos -= d;
            slow_path++;
        }
        state = new_state;
    }

    auto end_decode = std::chrono::high_resolution_clock::now();
    auto end_total = end_decode;

    // Calculate durations in microseconds
    auto setup_time = std::chrono::duration_cast<std::chrono::microseconds>(end_setup - start_setup).count();
    auto tables_time = std::chrono::duration_cast<std::chrono::microseconds>(end_tables - start_tables).count();
    auto decode_time = std::chrono::duration_cast<std::chrono::microseconds>(end_decode - start_decode).count();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count();

    std::cout << "\nANS Decode Profiling:\n"
              << "Setup time: " << setup_time << " µs\n"
              << "Tables creation time: " << tables_time << " µs\n"
              << "Main decode loop time: " << decode_time << " µs\n"
              << "Total time: " << total_time << " µs\n"
              << "Fast path hits: " << fast_path << "\n"
              << "Slow path hits: " << slow_path << "\n"
              << std::endl;
}

}  // namespace simple_ans
