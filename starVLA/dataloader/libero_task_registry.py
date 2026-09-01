from __future__ import annotations


# ---------------------------------------------------------------------
# LIBERO suite-local task ID -> merged LeRobot dataset task_index
#
# These indices correspond to:
# dataset/libero_merged_no_noops_20hz/meta/tasks.parquet
# ---------------------------------------------------------------------

LIBERO_TASK_INDEX_MAP = {
    "libero_goal": {
        0: 19,
        1: 17,
        2: 14,
        3: 12,
        4: 18,
        5: 15,
        6: 13,
        7: 16,
        8: 10,
        9: 11,
    },
    
    "libero_object": {
        0: 24,
        1: 22,
        2: 26,
        3: 23,
        4: 21,
        5: 28,
        6: 27,
        7: 25,
        8: 29,
        9: 20,
    },
}


LIBERO_TASK_INSTRUCTION_MAP = {
    "libero_goal": {
        0: "open the middle drawer of the cabinet",
        1: "put the bowl on the stove",
        2: "put the wine bottle on top of the cabinet",
        3: "open the top drawer and put the bowl inside",
        4: "put the bowl on top of the cabinet",
        5: "push the plate to the front of the stove",
        6: "put the cream cheese in the bowl",
        7: "turn on the stove",
        8: "put the bowl on the plate",
        9: "put the wine bottle on the rack",
    },
    
    "libero_object": {
        0: "pick up the alphabet soup and place it in the basket",
        1: "pick up the cream cheese and place it in the basket",
        2: "pick up the salad dressing and place it in the basket",
        3: "pick up the bbq sauce and place it in the basket",
        4: "pick up the ketchup and place it in the basket",
        5: "pick up the tomato sauce and place it in the basket",
        6: "pick up the butter and place it in the basket",
        7: "pick up the milk and place it in the basket",
        8: "pick up the chocolate pudding and place it in the basket",
        9: "pick up the orange juice and place it in the basket",
    },
}


def resolve_libero_task_indices(
    suite: str,
    task_ids,
) -> list[int]:
    """
    Convert LIBERO suite-local task IDs into task_index values used by
    the merged LeRobot LIBERO dataset.

    Example:
        suite="libero_goal"
        task_ids=[0, 1, 2, 3, 4, 5]

    returns:
        [19, 17, 14, 12, 18, 15]
    """
    suite = str(suite)

    if suite not in LIBERO_TASK_INDEX_MAP:
        raise ValueError(
            f"Unsupported LIBERO suite: {suite}. "
            f"Available suites: {sorted(LIBERO_TASK_INDEX_MAP)}"
        )

    mapping = LIBERO_TASK_INDEX_MAP[suite]

    task_ids = [int(task_id) for task_id in task_ids]

    if len(task_ids) == 0:
        raise ValueError(
            "`cl_task_ids` must contain at least one task."
        )

    invalid = [
        task_id
        for task_id in task_ids
        if task_id not in mapping
    ]

    if invalid:
        raise ValueError(
            f"Invalid task IDs for {suite}: {invalid}. "
            f"Available task IDs: {sorted(mapping)}"
        )

    # Prevent silent duplicate tasks.
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(
            f"Duplicate task IDs are not allowed: {task_ids}"
        )

    return [
        mapping[task_id]
        for task_id in task_ids
    ]


def get_libero_task_instruction(
    suite: str,
    task_id: int,
) -> str:
    return LIBERO_TASK_INSTRUCTION_MAP[
        str(suite)
    ][int(task_id)]
