from typing import Union
import numpy as np
from .EncodedSignal import EncodedSignal
from .choose_symbol_counts import choose_symbol_counts
from ._simple_ans import (
    ans_encode_int16 as _ans_encode_int16,
    ans_decode_int16 as _ans_decode_int16,
    ans_unique_int16 as _ans_unique_int16,
    ans_encode_int32 as _ans_encode_int32,
    ans_decode_int32 as _ans_decode_int32,
    ans_unique_int32 as _ans_unique_int32,
    ans_encode_uint16 as _ans_encode_uint16,
    ans_decode_uint16 as _ans_decode_uint16,
    ans_unique_uint16 as _ans_unique_uint16,
    ans_encode_uint32 as _ans_encode_uint32,
    ans_decode_uint32 as _ans_decode_uint32,
    ans_unique_uint32 as _ans_unique_uint32,
    ans_encode_uint8 as _ans_encode_uint8,
    ans_decode_uint8 as _ans_decode_uint8,
    ans_unique_uint8 as _ans_unique_uint8,
)


def _ans_unique(arr: np.ndarray):
    """Find unique elements and the number of times they appear.

    Args:
        arr: 1D numpy array. Must be int32, int16, uint32, uint16, or uint8.

    Returns:
        A tuple[ndarray, ndarray] where the first array contains the sorted unique elements,
        and the second is the respective counts.
    """
    dtype = arr.dtype
    if dtype == np.int32:
        vals, counts = _ans_unique_int32(arr)
    elif dtype == np.int16:
        vals, counts = _ans_unique_int16(arr)
    elif dtype == np.uint32:
        vals, counts = _ans_unique_uint32(arr)
    elif dtype == np.uint8:
        vals, counts = _ans_unique_uint8(arr)
    elif dtype == np.uint16:
        vals, counts = _ans_unique_uint16(arr)
    else:
        raise TypeError("Invalid numpy type")

    assert len(vals) == len(counts)

    if not len(vals):
        vals, counts = np.unique(arr, return_counts=True)

    return vals, counts


def ans_encode(signal: np.ndarray, *, precision: Union[int, None] = None, verbose=False) -> EncodedSignal:
    """Encode a signal using Asymmetric Numeral Systems (ANS).

    Args:
        signal: Input signal to encode as a 1D numpy array. Must be int32, int16, uint32, uint16, or uint8.
        precision: The size of the index table will be 2^precision.
            If None, the precision is chosen smartly to be the smallest value that is expected to preserve
            98% of the compressibility, but not more than 24.
        verbose: If True, print additional information such as the chosen index size.

    Returns:
        An EncodedSignal object containing the encoded data.
    """
    if signal.dtype not in [np.int32, np.int16, np.uint32, np.uint16, np.uint8]:
        raise TypeError("Input signal must be int32, int16, uint32, uint16, or uint8")
    assert signal.ndim == 1, "Input signal must be a 1D array"

    signal_length = len(signal)
    vals, counts = _ans_unique(signal)
    vals = np.array(vals, dtype=signal.dtype)
    probs = counts / np.sum(counts)

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
    S = len(vals)
    if S > index_size:
        raise ValueError(f"Number of unique symbols cannot be greater than index size, got {S} unique symbols and index size = {index_size}")

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
    elif dtype == np.uint16:
        encoded = _ans_encode_uint16(signal, symbol_counts, symbol_values)
    else:  # dtype == np.uint8
        encoded = _ans_encode_uint8(signal, symbol_counts, symbol_values)

    ret = EncodedSignal(
        state=encoded.state,
        words=encoded.words,
        symbol_counts=symbol_counts,  # Already numpy array from above
        symbol_values=symbol_values,  # Already numpy array from above
        signal_length=signal_length
    )
    return ret


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
            encoded.words,
            encoded.symbol_counts,
            encoded.symbol_values,
            encoded.signal_length,
        )
    elif encoded.symbol_values.dtype == np.int16:
        return _ans_decode_int16(
            encoded.state,
            encoded.words,
            encoded.symbol_counts,
            encoded.symbol_values,
            encoded.signal_length,
        )
    elif encoded.symbol_values.dtype == np.uint32:
        return _ans_decode_uint32(
            encoded.state,
            encoded.words,
            encoded.symbol_counts,
            encoded.symbol_values,
            encoded.signal_length,
        )
    elif encoded.symbol_values.dtype == np.uint16:
        return _ans_decode_uint16(
            encoded.state,
            encoded.words,
            encoded.symbol_counts,
            encoded.symbol_values,
            encoded.signal_length,
        )
    else:  # dtype == np.uint8
        return _ans_decode_uint8(
            encoded.state,
            encoded.words,
            encoded.symbol_counts,
            encoded.symbol_values,
            encoded.signal_length,
        )
