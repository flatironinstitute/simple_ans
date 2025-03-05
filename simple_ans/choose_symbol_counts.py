import numpy as np


def choose_symbol_counts(proportions: np.ndarray, L: int) -> np.ndarray:
    """Convert real-valued proportions into integer counts summing to L.

    We aim to minimize (approximately) the sum of squared differences
    between the ideal proportions pi and the final counts ci / L, i.e.:

        minimize   sum_i (pi - (ci / L))^2

    The procedure used is:
      1. Normalize proportions so they sum to 1 (if they don't already).
      2. Give each category 1 item (so counts[i] >= 1).
      3. Distribute the remaining items (L - k) by taking the "largest remainder"
         of L' * pi, where L' = L - k, in descending order.

    Args:
        proportions: The target proportions [p1, p2, ..., pk].
                     They need not sum to 1.0 exactly; we re-normalize internally.
        L: Total number of items to be distributed among k categories.

    Returns:
        A numpy array of integer counts that sum to L and reflect the proportions
        as closely as possible, with each count >= 1.

    Raises:
        ValueError: If there are more proportions (k) than items to distribute (L).
    """
    k = len(proportions)
    if k > L:
        raise ValueError("Number of proportions cannot exceed total items to distribute.")

    # Normalize proportions so that sum(proportions) == 1
    proportions = proportions / proportions.sum()

    # Initially allocate 1 item to each category
    counts = np.ones(k, dtype=int)

    # Distribute the remaining L - k items proportionally
    remainder = L - k
    if remainder > 0:
        # Calculate fractional counts for the remainder
        fractional_counts = remainder * proportions

        # Split into floor and fraction
        floors = np.floor(fractional_counts).astype(int)
        frac_parts = fractional_counts - floors

        # Add the floored allocations
        counts += floors

        # How many items are left after flooring?
        leftover = remainder - floors.sum()
        if leftover > 0:
            # Distribute leftover by largest fractional parts
            # (argsort(descending) on the fractional parts)
            indices_desc_by_frac = np.argsort(-frac_parts)
            for i in range(leftover):
                counts[indices_desc_by_frac[i]] += 1

    return counts.astype(np.uint32)
