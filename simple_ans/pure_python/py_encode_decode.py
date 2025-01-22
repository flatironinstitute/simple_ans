import numpy as np
from ..EncodedSignal import EncodedSignal
from ..choose_symbol_counts import choose_symbol_counts


def py_ans_encode(signal: np.ndarray, *, index_size: int = 2**16) -> EncodedSignal:
    """Encode a signal using Asymmetric Numeral Systems (ANS).

    Args:
        signal: Input signal to encode as a 1D numpy array. Must be int32, int16, uint32, or uint16.
        index_size: Size of the index table. (default: 2**16).
        Must be a power of 2.
        Must be at least as large as the number of unique symbols in the input signal.

    Returns:
        An EncodedSignal object containing the encoded data.
    """
    assert signal.dtype in [np.int32, np.int16, np.uint32, np.uint16], "Input signal must be int32, int16, uint32, or uint16"
    assert signal.ndim == 1, "Input signal must be a 1D array"

    # index_size must be a power of 2
    if index_size & (index_size - 1) != 0:
        raise ValueError("index_size must be a power of 2")

    signal_length = len(signal)
    vals, counts = np.unique(signal, return_counts=True)
    vals = np.array(vals, dtype=signal.dtype)
    probs = counts / np.sum(counts)
    S = len(vals)
    if S > index_size:
        raise ValueError(f"Number of unique symbols cannot be greater than L, got {S} unique symbols and L = {index_size}")

    symbol_counts = choose_symbol_counts(probs, index_size)
    symbol_values = vals

    assert np.sum(symbol_counts) == index_size

    C = np.cumsum(symbol_counts) - symbol_counts  # More efficient than list comprehension

    # bit_shift_table[i, s] is the number of bits d to shift such that (L + i) // 2**d is between [f_s, 2 * f_s))
    bit_shift_table = np.zeros((index_size, S), dtype=np.uint32)
    for i in range(index_size):
        for s in range(S):
            f_s = symbol_counts[s]
            d = 0
            while (index_size + i) // (2 ** d) >= 2 * f_s:
                d += 1
            bit_shift_table[i, s] = d

    # State transition table maps current state and symbol to next state
    # state_transition_table[i, s] = L + C[s_ind] + ((L + i) // (2 ** d) - f_s
    state_transition_table = np.zeros((index_size, S), dtype=np.uint32)
    for i in range(index_size):
        for s in range(S):
            f_s = symbol_counts[s]
            d = bit_shift_table[i, s]
            assert f_s <= (index_size + i) // (2 ** d) < 2 * f_s
            state_transition_table[i, s] = index_size + C[s] + ((index_size + i) // (2 ** d) - f_s)
            assert C[s] < index_size
            assert index_size <= state_transition_table[i, s] < 2 * index_size

    symbol_index_lookup = {}
    for i in range(len(symbol_values)):
        symbol_index_lookup[symbol_values[i]] = i

    bits = []
    state = index_size
    for s_val in signal:
        s_ind = symbol_index_lookup[s_val]
        # choose d such that state // 2^d is in [f_s, 2 * f_s)
        d = bit_shift_table[state - index_size, s_ind]
        for j in range(d):
            bits.append((state >> (d - 1 - j)) % 2)
        state = int(state_transition_table[state - index_size, s_ind])
        # check that it is between L and 2L
        assert index_size <= state < 2 * index_size

    num_bits = len(bits)
    bitstream = pack_bitstream(bits)

    return EncodedSignal(
        state=state,
        bitstream=bitstream,
        num_bits=num_bits,
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
    bits = unpack_bitstream(E.bitstream, E.num_bits)
    x = np.zeros((E.signal_length), dtype=E.symbol_values.dtype)
    L = np.sum(E.symbol_counts)
    # Must be a power of 2
    assert L & (L - 1) == 0, "L must be a power of 2"
    C = np.cumsum(E.symbol_counts) - E.symbol_counts  # More efficient than list comprehension
    S = len(E.symbol_values)

    # symbol_lookup[i] is the decoded symbol index for state L + i
    symbol_lookup = np.zeros((L), dtype=np.int32)
    for s_ind in range(S):
        for j in range(E.symbol_counts[s_ind]):
            symbol_lookup[C[s_ind] + j] = s_ind

    # state_update[i] = f_s + i - C[s] computes intermediate state value
    state_update = np.zeros((L), dtype=np.uint32)
    for i in range(L):
        s = symbol_lookup[i]
        f_s = E.symbol_counts[s]
        state_update[i] = f_s + i - C[s]

    # bit_count_table[i] is the number of bits d such that i * 2**d is in [L, 2L)
    max_f_s = np.max(E.symbol_counts)
    bit_count_table = np.zeros((2 * max_f_s), dtype=np.uint32)
    for i in range(2 * max_f_s):
        if i > 0:
            d = 0
            while i * (2 ** d) < L:
                d += 1
            bit_count_table[i] = d

    state = E.state
    for i in range(E.signal_length):
        s_ind = symbol_lookup[state - L]
        x[i] = E.symbol_values[s_ind]

        state_2 = state_update[state - L]

        # find d such that state_2 * 2**d >= L
        d = bit_count_table[state_2]
        new_state = state_2 * (2 ** d)
        for j in range(d):
            bit = bits.pop()
            if bit:
                new_state += 2 ** j
        state = new_state
    return x[::-1]


def pack_bitstream(bits: list) -> bytes:
    """Pack a list of bits into a bytes object.

    Args:
        bits: List of bits (0s and 1s) to pack.

    Returns:
        Packed bits as a bytes object.
    """
    num_bits = len(bits)
    num_bytes = (num_bits + 7) // 8
    x = np.zeros((num_bytes), dtype=np.uint8)
    for i in range(len(bits)):
        if bits[i]:
            a = i // 8
            b = i % 8
            # Pack bits from MSB to LSB within each byte
            x[a] += 2 ** (7 - b)
    return x.tobytes()

def unpack_bitstream(bitstream: bytes, num_bits: int) -> list:
    """Unpack a bitstream back into a list of bits.

    Args:
        bitstream: Packed bits as a bytes object.
        num_bits: Number of bits to unpack.

    Returns:
        List of unpacked bits (0s and 1s).
    """
    x = np.frombuffer(bitstream, dtype=np.uint8)
    bits = []
    for i in range(num_bits):
        a = i // 8
        b = i % 8
        # Unpack bits from MSB to LSB within each byte
        bits.append((x[a] >> (7 - b)) % 2)
    return bits


if __name__ == '__main__':
    # Test 1: Basic test with small array
    print("Test 1: Basic test")
    proportions = [1, 2, 3]
    probs = np.array(proportions) / np.sum(proportions)
    signal_length = 10
    signal = np.random.choice(len(proportions), signal_length, p=probs).astype(np.uint16)
    encoded = py_ans_encode(signal)
    decoded = py_ans_decode(encoded)
    assert np.all(signal == decoded), "Basic test failed"

    # Test 2: Large uniform
    print("Test 2: Large uniform")
    num_symbols = 10
    signal_length = 10000
    signal = np.random.randint(num_symbols, size=signal_length).astype(np.int32)
    encoded = py_ans_encode(signal)
    decoded = py_ans_decode(encoded)
    assert np.all(signal == decoded), "Large uniform test failed"

    # Test 3: Skewed - mostly zeros, and some other values
    print("Test 3: Skewed")
    signal_length = 100000
    proportions = [1000, 1, 2, 5, 10]
    probs = np.array(proportions) / np.sum(proportions)
    signal = np.random.choice(len(proportions), signal_length, p=probs).astype(np.int16)
    encoded = py_ans_encode(signal)
    decoded = py_ans_decode(encoded)
    assert np.all(signal == decoded), "Skewed test failed"

    # Test 4: Negative numbers
    print("Test 4: Negative numbers")
    signal = np.random.randint(-10, 10, size=1000).astype(np.int32)
    encoded = py_ans_encode(signal)
    decoded = py_ans_decode(encoded)
    assert np.all(signal == decoded), "Negative numbers test failed"

    # Test 5: Binary signal
    print("Test 5: Binary signal")
    signal = np.random.choice([0, 1], size=50000, p=[0.3, 0.7]).astype(np.uint16)
    encoded = py_ans_encode(signal)
    decoded = py_ans_decode(encoded)
    assert np.all(signal == decoded), "Binary signal test failed"

    # Test 6: Constant signal
    print("Test 6: Constant signal")
    signal = np.full(1000, 5).astype(np.int16)  # Array of 1000 fives
    encoded = py_ans_encode(signal)
    decoded = py_ans_decode(encoded)
    assert np.all(signal == decoded), "Constant signal test failed"

    print("All tests passed!")
