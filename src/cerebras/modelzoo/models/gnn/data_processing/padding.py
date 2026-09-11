"""Count neighbor padding in consumed, host-side fixed-shape batches."""

import numpy as np


class NeighborPaddingStats:
    """Accumulate slot counts, including repeated occurrences of the same node."""

    def __init__(self):
        self.counts = []

    def update(self, payload):
        masks = payload["neighbor_masks"]
        if not self.counts:
            self.counts = [[0, 0, 0] for _ in masks]
        if len(masks) != len(self.counts):
            raise ValueError("Neighbor hop count changed during measurement")
        for depth, mask in enumerate(masks):
            # numpy shares CPU mask storage and counts without an int64 tensor copy.
            valid = int(np.count_nonzero(mask.numpy()))
            parents = int(np.count_nonzero(payload["node_masks"][depth].numpy()))
            counts = self.counts[depth]
            counts[0] += mask.numel()
            counts[1] += valid
            counts[2] += parents * mask.shape[-1]

    def merge(self, other):
        if not self.counts:
            self.counts = [row.copy() for row in other.counts]
            return
        if len(self.counts) != len(other.counts):
            raise ValueError("Neighbor hop count changed during measurement")
        for total, row in zip(self.counts, other.counts):
            for index, value in enumerate(row):
                total[index] += value

    def summary(self):
        def record(slots, valid, valid_parent_slots):
            padded = slots - valid
            missing = valid_parent_slots - valid
            return {
                "slots": slots,
                "valid_slots": valid,
                "padded_slots": padded,
                "padding_percent": 100 * padded / slots if slots else None,
                "slots_from_padded_parents": slots - valid_parent_slots,
                "slots_from_valid_parents": valid_parent_slots,
                "padded_slots_from_valid_parents": missing,
                "padding_percent_from_valid_parents": (
                    100 * missing / valid_parent_slots if valid_parent_slots else None
                ),
            }

        totals = [sum(row[index] for row in self.counts) for index in range(3)]
        return {
            "by_hop": [
                {"hop": depth + 1, **record(*counts)}
                for depth, counts in enumerate(self.counts)
            ],
            "overall": record(*totals),
        }
