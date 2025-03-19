def compute_score_single(conclusion, ground_truth):
    last_line = conclusion.split("\n")
    if len(last_line) == 0:
        return 0.

    last_line = last_line[-1].strip()
    conclusion = last_line
    if not conclusion.startswith("ANSWER: "):
        return 0.
    else:
        if "ANSWER: " in conclusion:
            conclusion = conclusion[conclusion.index(
                "ANSWER: ")+len("ANSWER: "):].strip()
            return 1.0 if conclusion == ground_truth else 0.
        else:
            return 0.


def compute_score_nothink(batch_solution_str, batch_ground_truth):
    rewards = []
    for i, (solution_str, ground_truth) in enumerate(zip(batch_solution_str, batch_ground_truth)):
        rewards.append(
            compute_score_single(solution_str, ground_truth)
        )
    return rewards
