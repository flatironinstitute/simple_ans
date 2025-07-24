from typing import Union
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simple_ans.EncodedSignal import EncodedSignal
from simple_ans.choose_symbol_counts import choose_symbol_counts

STATE_BITS = 64
WORD_BITS = 32
THRESHOLD = 1 << (STATE_BITS - WORD_BITS)


def py_ans_encode(signal: np.ndarray, *, precision: Union[int, None]=None) -> EncodedSignal:
    """Encode a signal using Asymmetric Numeral Systems (ANS).

    Args:
        signal: Input signal to encode as a 1D numpy array. Must be int32, int16, uint32, or uint16.
        index_size: Size of the index table. (default: 2**16).
        Must be a power of 2.
        Must be at least as large as the number of unique symbols in the input signal.

    Returns:
        An EncodedSignal object containing the encoded data.
    """
    if signal.dtype not in [np.int32, np.int16, np.uint32, np.uint16]:
        raise TypeError("Input signal must be int32, int16, uint32, or uint16")
    assert signal.ndim == 1, "Input signal must be a 1D array"

    # Get symbol counts and values using the main implementation
    signal_length = len(signal)
    vals, counts = np.unique(signal, return_counts=True)
    vals = np.array(vals, dtype=signal.dtype)
    probs = counts / np.sum(counts)
    S = len(vals)

    if precision is None:
        precision = 2
        entropy_target = -np.sum(probs * np.log2(probs))
        while precision < 24:
            L = 2 ** precision
            if L >= len(vals):
                symbol_counts_0 = choose_symbol_counts(probs, L)
                probs_0 = symbol_counts_0 / L
                entropy_target = -np.sum(probs * np.log2(probs))
                entropy_0 = -np.sum(probs * np.log2(probs_0))
                if entropy_0 <= entropy_target / 0.98 or L >= 2**20:
                    print(f'Using precision {precision} with index size {L} (entropy ratio: {(entropy_0 / entropy_target if entropy_target else 1):.2f})')
                    index_size = L
                    break
            precision += 1
    assert precision is not None

    index_size = 2 ** precision
    if S > index_size:
        raise ValueError(f"Number of unique symbols cannot be greater than L, got {S} unique symbols and index size = {index_size}")

    symbol_counts = choose_symbol_counts(probs, index_size)
    symbol_values = vals

    # Calculate L and verify it's a power of 2
    L = np.sum(symbol_counts)
    if L & (L - 1) != 0:
        raise ValueError("L must be a power of 2")

    # Pre-compute cumulative sums
    C = np.zeros(len(symbol_counts), dtype=np.uint32)
    for i in range(1, len(symbol_counts)):
        C[i] = C[i - 1] + symbol_counts[i - 1]

    # Create symbol index lookup
    symbol_index_lookup = {symbol_values[i]: i for i in range(len(symbol_values))}

    # Initialize state and stack
    state = np.uint64(0)  # Use uint64 for state
    words = []
    PRECISION_BITS = precision

    MASK_WORD = (1 << WORD_BITS) - 1  # Mask for the last WORD_BITS bits

    # Encode each symbol in reverse order
    for i in range(signal_length):
        symbol = signal[i]
        s_ind = symbol_index_lookup[symbol]

        F_s = symbol_counts[s_ind]  # Frequency of the symbol
        C_s = C[s_ind]  # Cumulative frequency up to the symbol

        # Check if we need to normalize
        # if (state >> (STATE_BITS - PRECISION_BITS)) >= F_s
        if (state >> (STATE_BITS - PRECISION_BITS)) >= F_s:
            emit_word = np.uint32(state & MASK_WORD)
            state = state >> WORD_BITS
            words.append(emit_word)

        remainder = state % F_s
        prefix = state // F_s
        quantile = C_s + remainder
        state = (prefix << PRECISION_BITS) | quantile
        if len(words) > 0 and state < THRESHOLD:
            raise ValueError(f"Unexpected: State is too small during encoding.")
        # print(f'(PY) State after encoding symbol {symbol} (index {s_ind}): {state}')

    return EncodedSignal(
        state=int(state),
        words=np.array(words, dtype=np.uint32),
        symbol_counts=symbol_counts,
        symbol_values=symbol_values,
        signal_length=signal_length
    )


def py_ans_decode(E: EncodedSignal) -> np.ndarray:
    """Decode an ANS-encoded signal.

    Args:
        E: EncodedSignal object containing the encoded data.

    Returns:
        Decoded signal as a numpy array.
    """
    # Calculate index size and verify it's a power of 2
    index_size = np.uint64(np.sum(E.symbol_counts))
    if index_size & (index_size - 1) != 0:
        raise ValueError("L must be a power of 2")
    precision = 1
    while (1 << precision) < index_size:
        precision += 1
    if (1 << precision) != index_size:
        raise ValueError(f"Index size {index_size} is not a power of 2, got precision {precision}")

    # Pre-compute cumulative sums
    C = np.zeros(len(E.symbol_counts), dtype=np.uint64)
    for i in range(1, len(E.symbol_counts)):
        C[i] = C[i - 1] + E.symbol_counts[i - 1]

    # Create symbol lookup table
    symbol_lookup = np.zeros(index_size, dtype=np.uint64)
    for s_ind in range(len(E.symbol_counts)):
        for j in range(E.symbol_counts[s_ind]):
            symbol_lookup[C[s_ind] + j] = s_ind

    words = E.words

    stack_index = len(words) - 1

    # Initialize output array
    output = np.zeros(E.signal_length, dtype=E.symbol_values.dtype)
    state = np.uint64(E.state)

    PRECISION_BITS = precision

    state = E.state

    # Decode symbols in reverse order
    for i in range(E.signal_length):
        if stack_index >= 0:
            # verify that we are in the correct range
            if state < THRESHOLD:
                raise ValueError("State is too small, likely due to an error in encoding or decoding")
        prefix = state >> PRECISION_BITS
        quantile = state & ((1 << PRECISION_BITS) - 1)
        # Find s such that C_s <= quantile < C_s + f_s
        s_ind = 0
        while s_ind < len(E.symbol_counts) and quantile >= C[s_ind] + E.symbol_counts[s_ind]:
            s_ind += 1
        if s_ind >= len(E.symbol_counts):
            raise ValueError(f"Quantile {quantile} out of bounds for cumulative counts {C}")
        F_s = E.symbol_counts[s_ind]  # Frequency of the symbol
        previous_state = np.uint64(prefix) * np.uint64(F_s) + quantile - C[s_ind]

        if previous_state < THRESHOLD and stack_index >= 0:
            word = words[stack_index]
            stack_index -= 1
            previous_state = (previous_state << WORD_BITS) | word
            if previous_state < THRESHOLD:
                raise ValueError("Unexpected: State is too small after correcting for normalization.")

        state = previous_state
        output[E.signal_length - i - 1] = E.symbol_values[s_ind]

    return output


if __name__ == '__main__':
    from simple_ans import ans_encode, ans_decode

    signals = []

    # test 1 - Basic test with small array
    proportions = [1, 2, 3]
    probs = np.array(proportions) / np.sum(proportions)
    signal_length = 20
    signal = np.random.choice(len(proportions), signal_length, p=probs).astype(np.uint16)
    signals.append(signal)

    # test 2 - Large uniform
    num_symbols = 10
    signal_length = 100_000
    signal = np.random.randint(num_symbols, size=signal_length).astype(np.int32)
    signals.append(signal)

    # test 3 - Skewed - mostly zeros, and some other values
    signal_length = 100_000
    proportions = [1000, 1, 2, 5, 10]
    probs = np.array(proportions) / np.sum(proportions)
    signal = np.random.choice(len(proportions), signal_length, p=probs).astype(np.int16)
    signals.append(signal)

    # test 4 - Negative numbers
    signal = np.random.randint(-10, 10, size=100_000).astype(np.int32)
    signals.append(signal)

    # test 5 - Binary signal
    signal = np.random.choice([0, 1], size=50000, p=[0.3, 0.7]).astype(np.uint16)
    signals.append(signal)

    # test 6 - Constant signal
    signal = np.full(1000, 5).astype(np.int16)  # Array of 1000 fives
    signals.append(signal)

    for i, signal in enumerate(signals):
        print(f'Test {i + 1}')
        encoded = py_ans_encode(signal)
        encoded2 = ans_encode(signal)
        assert encoded.state == encoded2.state, f"States do not match for test {i + 1}. {encoded.state} != {encoded2.state}"
        words1 = encoded.words
        words2 = encoded2.words
        assert np.all(words1 == words2), f"Words do not match for test {i + 1}"
        decoded = py_ans_decode(encoded)
        assert np.all(signal == decoded), f"Test {i + 1} failed"
        decoded2 = py_ans_decode(encoded2)
        assert np.all(signal == decoded2), f"Test {i + 1} failed"

    print("All tests passed!")
