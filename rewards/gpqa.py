import re


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


def compute_score_nothink(batch_data_sources, batch_solution_str, batch_ground_truth):
    rewards = []
    for i, (solution_str, ground_truth) in enumerate(zip(batch_solution_str, batch_ground_truth)):
        raw_solution_str = solution_str
        solution_str = postprocess_solution(solution_str)
        rewards.append(
            compute_score_single(solution_str, ground_truth)
        )
        print(f"--------------------------------")
        print(f"Solution: {repr(raw_solution_str)}")
        print(f"Ground Truth: {repr(ground_truth)}")
        print(f"Reward: {rewards[-1]}")
    return rewards


def get_thought(solution_str: str):
    thought = re.findall(r'```xml.*```', solution_str, re.DOTALL)[0]
    return thought


def get_conclusion(solution_str: str):
    thought = get_thought(solution_str)
    return solution_str[solution_str.index(thought)+len(thought):].strip()


def postprocess_solution(solution_str):
    if "<|im_end|>" in solution_str:
        return solution_str[:solution_str.index("<|im_end|>")]
    return solution_str


def compute_score(batch_data_sources, batch_solution_str, batch_ground_truth):
    rewards = []
    for i, (solution_str, ground_truth) in enumerate(zip(batch_solution_str, batch_ground_truth)):
        raw_solution_str = solution_str
        try:
            solution_str = postprocess_solution(solution_str)
            conclusion = get_conclusion(solution_str)
            rewards.append(
                compute_score_single(conclusion, ground_truth)
            )
        except Exception as err:
            rewards.append(0.)
        print(f"--------------------------------")
        print(f"Solution: {repr(raw_solution_str)}")
        print(f"Ground Truth: {repr(ground_truth)}")
        print(f"Reward: {rewards[-1]}")
    return rewards
