import math

import pytest


@pytest.mark.parametrize("micro_size", [1, 2, 4])
def test_equal_target_rows_give_every_dp_the_same_microbatch_schedule(micro_size):
    target_rows = 6
    expected = [
        min(micro_size, target_rows - start)
        for start in range(0, target_rows, micro_size)
    ]
    schedules = [expected.copy() for _ in range(4)]
    assert all(schedule == schedules[0] for schedule in schedules)
    assert len(expected) == math.ceil(target_rows / micro_size)


def test_smaller_bucket_empty_dp_still_gets_dummy_schedule():
    real_rows = [3, 0, 0, 0]
    sp_size = 2
    target_rows = math.ceil(max(real_rows) / sp_size) * sp_size
    assert target_rows == 4
    assert [target_rows - rows for rows in real_rows] == [1, 4, 4, 4]
