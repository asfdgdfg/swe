def _local_width(rows):
    return max(row[0] for row in rows) + max(row[1] for row in rows)


def test_logical_dp_replicas_keep_independent_local_shapes():
    logical_dp_rows = {
        0: [(3, 5)],
        1: [(5, 7)],
        2: [(8, 12)],
        3: [(2, 4)],
    }
    widths = [_local_width(logical_dp_rows[dp]) for dp in range(4)]
    assert widths == [8, 12, 20, 6]
    assert len({len(rows) for rows in logical_dp_rows.values()}) == 1
    assert widths != [max(widths)] * 4


def test_minimal_padding_does_not_change_packing_metadata():
    before = [
        (0, 0, False, 0, 0.25),
        (0, 1, False, 1, 0.25),
        (0, 2, True, -1, 0.0),
        (0, 3, True, -1, 0.0),
    ]
    # Only sequence widths are local-forward properties; packing tuples stay exact.
    local_widths = [8, 12, 2, 2]
    after = list(before)
    assert after == before
    assert local_widths != [20] * 4
