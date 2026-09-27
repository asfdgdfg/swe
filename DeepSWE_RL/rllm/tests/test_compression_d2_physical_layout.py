from rllm.trainer.verl.agent_ppo_trainer import _physical_rank_major_row_indices


def _rows(real_counts, target_rows):
    rows = []
    for dp, real_count in enumerate(real_counts):
        for index in range(target_rows):
            rows.append({"optimizer_step_id": 0, "target_dp_replica": dp, "is_dummy": index >= real_count})
    return rows


def test_sp2_target_rows_rounds_five_up_to_six():
    real = [4, 2, 5, 3]
    target = 6
    rows = _rows(real, target)
    order = _physical_rank_major_row_indices(rows, effective_dp_degree=4, sp_size=2)
    chunks = [order[i * 3:(i + 1) * 3] for i in range(8)]
    assert [len(chunk) for chunk in chunks] == [3] * 8
    for dp in range(4):
        restored = chunks[2 * dp] + chunks[2 * dp + 1]
        assert {rows[index]["target_dp_replica"] for index in restored} == {dp}
    assert [target - value for value in real] == [2, 4, 1, 3]


def test_multiple_steps_remain_physical_rank_major_for_world_chunk():
    rows = []
    for step in (0, 1):
        for dp in range(4):
            for _ in range(4):
                rows.append({"optimizer_step_id": step, "target_dp_replica": dp})
    order = _physical_rank_major_row_indices(rows, effective_dp_degree=4, sp_size=2)
    chunks = [order[i * 4:(i + 1) * 4] for i in range(8)]
    for rank, chunk in enumerate(chunks):
        assert {rows[index]["target_dp_replica"] for index in chunk} == {rank // 2}
        assert [rows[index]["optimizer_step_id"] for index in chunk] == [0, 0, 1, 1]
