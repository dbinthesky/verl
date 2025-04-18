import re
from functools import partial


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution


def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def compute_score_single(conclusion, ground_truth):
    if "ANSWER: " not in conclusion:
        return 0.
    conclusion = conclusion[conclusion.index(
        "ANSWER:")+len("ANSWER:"):].strip()
    count = 0
    for key in ("A", "B", "C", "D"):
        if key in conclusion:
            count += 1
    if count > 1:
        return 0.0

    if ground_truth in conclusion:
        return 1.0
    return 0.


def postprocess_solution(solution_str):
    if "<|im_end|>" in solution_str:
        return solution_str[:solution_str.index("<|im_end|>")]
    return solution_str


def compute_score(batch_data_sources, batch_solution_str, batch_ground_truth, postprocess_solution_fn=postprocess_solution):
    rewards = []

    def clip_str(s):
        if len(s) > 2000:
            return f'{s[:1000]}... ...{s[-1000:]}'
        return s

    for i, (solution_str, ground_truth) in enumerate(zip(batch_solution_str, batch_ground_truth)):
        ground_truth = ground_truth["ground_truth"]
        raw_solution_str = solution_str
        try:
            conclusion = postprocess_solution_fn(solution_str)
            if conclusion is None:
                conclusion = raw_solution_str
            rewards.append(
                compute_score_single(conclusion, ground_truth)
            )
            print(f"-----------------[MMLU TEST]---------------")
            print(f"【Solution】`{repr(clip_str(conclusion))}`")
            print(f"【Ground Truth】`{repr(ground_truth)}`")
            print(
                f'Reward={rewards[-1]}')
        except Exception as err:
            print(f'[EVAL ERROR] {err}')
            rewards.append(0.)
    return rewards


if __name__ == "__main__":
    s = "To solve this problem, we need to analyze the forces acting on the two objects and ensure that the equilibrium condition is maintained after the mass of Object A is halved.\n\n### Step 1: Initial Equilibrium Condition\nInitially, the gravitational force and the electrostatic force between the two objects are equal. The gravitational force \\( F_g \\) is given by:\n\\[ F_g = G \\frac{m_A m_B}{r^2} \\]\nwhere \\( G \\) is the gravitational constant, \\( m_A \\) and \\( m_B \\) are the masses of the objects, and \\( r \\) is the distance between them.\n\nThe electrostatic force \\( F_e \\) is given by:\n\\[ F_e = k \\frac{q_A q_B}{r^2} \\]\nwhere \\( k \\) is the Coulomb constant, and \\( q_A \\) and \\( q_B \\) are the charges of the objects.\n\nAt equilibrium, these forces are equal:\n\\[ G \\frac{m_A m_B}{r^2} = k \\frac{q_A q_B}{r^2} \\]\nWe can cancel out \\( \\frac{1}{r^2} \\) from both sides:\n\\[ G m_A m_B = k q_A q_B \\]\n\n### Step 2: New Condition After Halving the Mass of Object A\nNow, the mass of Object A is halved, so the new mass of Object A is \\( \\frac{m_A}{2} \\). The new equilibrium condition must still be:\n\\[ G \\frac{\\left(\\frac{m_A}{2}\\right) m_B}{r^2} = k \\frac{q_A q_B}{r^2} \\]\nAgain, we can cancel out \\( \\frac{1}{r^2} \\) from both sides:\n\\[ G \\frac{m_A}{2} m_B = k q_A q_B \\]\n\n### Step 3: Simplify the New Condition\nWe can simplify the new condition by dividing both sides by \\( G m_B \\):\n\\[ \\frac{m_A}{2} = \\frac{k q_A q_B}{G m_B} \\]\n\n### Step 4: Compare with the Original Condition\nFrom the original condition, we know:\n\\[ m_A = \\frac{k q_A q_B}{G m_B} \\]\n\nSubstituting this into the new condition, we get:\n\\[ \\frac{1}{2} m_A = \\frac{k q_A q_B}{G m_B} \\]\n\\[ \\frac{1}{2} \\left( \\frac{k q_A q_B}{G m_B} \\right) = \\frac{k q_A q_B}{G m_B} \\]\n\\[ \\frac{1}{2} m_A = m_A \\]\n\nThis simplifies to:\n\\[ \\frac{1}{2} m_A = m_A \\]\n\n### Step 5: Determine the Required Change in Charge\nTo maintain the equilibrium, the product \\( q_A q_B \\) must be halved. Therefore, the charge on Object B must be doubled.\n\n### Conclusion\nThe correct choice is:\n\\[ \\boxed{C} \\]\n\nANSWER: C"
    print(compute_score(None, [s], [{"ground_truth": "C"}]))
