import numpy as np
import pytest
from simple_ans import (
    ans_encode,
    ans_decode,
    choose_symbol_counts,
    determine_symbol_counts_and_values,
    EncodedSignal,
)


def test_encode_decode():
    # Test all supported types
    dtypes = [np.int32, np.int16, np.uint32, np.uint16]
    for dtype in dtypes:
        # Create a simple test signal
        signal = np.array([0, 1, 2, 1, 0], dtype=dtype)

        # Create symbol counts and values
        symbol_counts = np.array([3, 3, 2], dtype=np.uint32)  # For symbols 0,1,2
        symbol_values = np.array([0, 1, 2], dtype=dtype)  # Corresponding values

        # Encode
        encoded = ans_encode(signal, symbol_counts, symbol_values)
        assert isinstance(
            encoded, EncodedSignal
        ), "Result should be EncodedSignal object"
        assert isinstance(
            encoded.bitstream, bytes
        ), "Encoded bitstream should be bytes"
        assert len(encoded.bitstream) % 8 == 0, "Bitstream length should be multiple of 8"

        # Decode
        decoded = ans_decode(encoded)

        # Verify
        assert np.array_equal(
            signal, decoded
        ), f"Decoded signal does not match original for dtype {dtype}"
    print("Test passed: encode/decode works correctly for all types")


def test_choose_symbol_counts():
    # Test with some probabilities
    proportions = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    L = 1024  # Should be power of 2

    counts = choose_symbol_counts(proportions, L)
    total = sum(counts)

    assert total == L, f"Total counts should sum to {L}"
    print("Test passed: choose_symbol_counts works correctly")


def test_determine_symbol_counts_and_values():
    # Test with default index length
    signal = [0, 1, 2, 1, 0]
    counts, values = determine_symbol_counts_and_values(signal, dtype=np.dtype(np.int32))
    assert isinstance(counts, np.ndarray), "Counts should be a numpy array"
    assert isinstance(values, np.ndarray), "Values should be a numpy array"
    assert counts.dtype == np.uint32, "Counts should be uint32 type"
    assert (
        values.dtype == np.int32
    ), "Values should match requested dtype (default int32)"
    assert len(counts) == len(values), "Counts and values should have same length"
    assert np.array_equal(
        values, np.array([0, 1, 2], dtype=np.int32)
    ), "Values should match unique signal values"
    assert sum(counts) == 2**16, "Total counts should sum to default index length"

    # Test with custom index length
    counts, values = determine_symbol_counts_and_values(signal, index_length=1024, dtype=np.dtype(np.int32))
    assert sum(counts) == 1024, "Total counts should sum to specified index length"

    # Test error cases
    with pytest.raises(ValueError):
        determine_symbol_counts_and_values([])  # Empty signal
    with pytest.raises(ValueError):
        determine_symbol_counts_and_values(
            signal, index_length=1000
        )  # Non-power-of-2 index length
    with pytest.raises(ValueError):
        determine_symbol_counts_and_values(
            signal, index_length=-1
        )  # Negative index length


def test_auto_symbol_counts():
    print("Starting test_auto_symbol_counts")
    # Test all supported types
    dtypes = [np.int32, np.int16, np.uint32, np.uint16]
    for dtype in dtypes:
        # Test encoding with auto-determined symbol counts
        signal = np.array([0, 1, 2, 1, 0], dtype=dtype)
        print(f"Testing dtype {dtype}")
        encoded = ans_encode(signal, dtype=dtype)  # No symbol_counts provided
        print("Signal encoded")
        decoded = ans_decode(encoded)
        print("Signal decoded")
        assert np.array_equal(
            signal, decoded
        ), f"Decoded signal does not match original for dtype {dtype}"
    print("Test passed: auto symbol counts works correctly for all types")


def test_incorrect_data_types():
    # Test with incorrect signal dtype
    signal_float = np.array([0, 1, 2, 1, 0], dtype=np.float32)
    symbol_counts = np.array([3, 3, 2], dtype=np.uint32)
    symbol_values = np.array([0, 1, 2], dtype=np.int32)
    with pytest.raises((TypeError, ValueError)):
        ans_encode(signal_float, symbol_counts, symbol_values)

    # Test with incorrect symbol_counts dtype
    signal = np.array([0, 1, 2, 1, 0], dtype=np.int32)
    symbol_counts_float = np.array([3, 3, 2], dtype=np.float32)
    with pytest.raises((TypeError, ValueError)):
        ans_encode(signal, symbol_counts_float, symbol_values)

    # Test with mismatched symbol_values dtype
    signal = np.array([0, 1, 2, 1, 0], dtype=np.int32)
    symbol_values_wrong = np.array([0, 1, 2], dtype=np.uint32)
    with pytest.raises((TypeError, ValueError)):
        ans_encode(signal, symbol_counts, symbol_values_wrong)

    # Test with incorrect types in auto mode
    with pytest.raises((TypeError, ValueError)):
        ans_encode(signal_float)  # Should fail with float signal

    # Test with invalid dtype
    with pytest.raises(ValueError):
        ans_encode(signal, dtype=np.dtype(np.float32))  # Should fail with float dtype

    print("Test passed: incorrect data types handled correctly")


if __name__ == "__main__":
    test_encode_decode()
    test_choose_symbol_counts()
    test_determine_symbol_counts_and_values()
    test_auto_symbol_counts()
    test_incorrect_data_types()
