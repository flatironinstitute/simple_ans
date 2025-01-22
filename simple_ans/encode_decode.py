import numpy as np
from .EncodedSignal import EncodedSignal
from .choose_symbol_counts import choose_symbol_counts
from ._simple_ans import (
    ans_encode_int16 as _ans_encode_int16,
    ans_decode_int16 as _ans_decode_int16,
    ans_encode_int32 as _ans_encode_int32,
    ans_decode_int32 as _ans_decode_int32,
    ans_encode_uint16 as _ans_encode_uint16,
    ans_decode_uint16 as _ans_decode_uint16,
    ans_encode_uint32 as _ans_encode_uint32,
    ans_decode_uint32 as _ans_decode_uint32
)


def ans_encode(signal: np.ndarray, *, index_size: int = 2**16) -> EncodedSignal:
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

    dtype = signal.dtype
    if dtype == np.int32:
        encoded = _ans_encode_int32(signal, symbol_counts, symbol_values)
    elif dtype == np.int16:
        encoded = _ans_encode_int16(signal, symbol_counts, symbol_values)
    elif dtype == np.uint32:
        encoded = _ans_encode_uint32(signal, symbol_counts, symbol_values)
    else:  # dtype == np.uint16
        encoded = _ans_encode_uint16(signal, symbol_counts, symbol_values)

    return EncodedSignal(
        state=encoded.state,
        bitstream=encoded.bitstream,
        num_bits=encoded.num_bits,
        symbol_counts=symbol_counts,  # Already numpy array from above
        symbol_values=symbol_values,  # Already numpy array from above
        signal_length=signal_length
    )


def ans_decode(encoded: EncodedSignal) -> np.ndarray:
    """Decode an ANS-encoded signal.

    Args:
        E: EncodedSignal object containing the encoded data.

    Returns:
        Decoded signal as a numpy array.
    """
    if encoded.symbol_values.dtype == np.int32:
        return _ans_decode_int32(
            encoded.state,
            encoded.bitstream,
            encoded.num_bits,
            encoded.symbol_counts,
            encoded.symbol_values,
            encoded.signal_length,
        )
    elif encoded.symbol_values.dtype == np.int16:
        return _ans_decode_int16(
            encoded.state,
            encoded.bitstream,
            encoded.num_bits,
            encoded.symbol_counts,
            encoded.symbol_values,
            encoded.signal_length,
        )
    elif encoded.symbol_values.dtype == np.uint32:
        return _ans_decode_uint32(
            encoded.state,
            encoded.bitstream,
            encoded.num_bits,
            encoded.symbol_counts,
            encoded.symbol_values,
            encoded.signal_length,
        )
    else:  # dtype == np.uint16
        return _ans_decode_uint16(
            encoded.state,
            encoded.bitstream,
            encoded.num_bits,
            encoded.symbol_counts,
            encoded.symbol_values,
            encoded.signal_length,
        )
