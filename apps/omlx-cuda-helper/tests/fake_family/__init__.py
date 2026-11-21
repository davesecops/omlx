# SPDX-License-Identifier: Apache-2.0
"""A family with numpy arithmetic for layers, to test stages without a GPU."""

FIRST_STAGE_MIN = 2
PACKS_WEIGHTS = True


# Bytes of a packed layer whose codes keep their own widths; a test sets it
# below the widened size.
NATIVE_BYTES = 1 << 19


def plan(model_dir, packed_weights=False, widen=True):
    """(bytes outside the layers, bytes per layer) of the fake checkpoint."""
    if not packed_weights:
        return 1 << 20, [1 << 20] * 8
    return 1 << 20, [(1 << 19) if widen else NATIVE_BYTES] * 8
