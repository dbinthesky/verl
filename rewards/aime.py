import re
from functools import partial
from typing import Optional

# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  # implicit mults
    return step


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string


# Dan Hendrycks' code
def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd and ground_truth_normalized_mathd is not None:
        return True
    return False


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(
                    ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


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


def postprocess_solution(solution_str):
    if "<|im_end|>" in solution_str:
        return solution_str[:solution_str.index("<|im_end|>")]
    return solution_str


def qwq_longcot_postprocess_solution(solution_str):
    solution_str = postprocess_solution(solution_str)
    try:
        thought = re.findall(r'<think>.*</think>',
                             solution_str, re.DOTALL)[0]
    except Exception as err:
        return None

    conclusion = solution_str.replace(thought, "").strip()

    return conclusion


def compute_score_single(conclusion, ground_truth):
    conclusion = extract_answer(conclusion)
    if isinstance(ground_truth, (str, float, int)):
        ground_truths = [ground_truth]
    processed_ground_truths = []
    for truth in ground_truths:
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)

    for ground_truth in processed_ground_truths:
        if grade_answer_mathd(
            conclusion, ground_truth
        ) or grade_answer_sympy(conclusion, ground_truth):
            return 1.0
    return 0.0


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
                rewards.append(0.)
                continue

            compute_score_single(conclusion, ground_truth)
            rewards.append(
                compute_score_single(conclusion, ground_truth)
            )
            print(f"-----------------[AIME 2024 TEST]---------------")
            print(f"Solution: {repr(clip_str(conclusion))}")
            print(f"Ground Truth: {repr(ground_truth)}")
            print(
                f'Reward: {rewards[-1]}; Format Checker: <think> in response={"<think>" in solution_str}; </think> in response={"</think>" in solution_str}')
        except Exception as err:
            print(err)
            rewards.append(0.)
    return rewards


qwq_longcot_compute_score = partial(
    compute_score, postprocess_solution_fn=qwq_longcot_postprocess_solution)

if __name__ == "__main__":
    print(qwq_longcot_compute_score(
        [None],
        batch_solution_str=[
            "<think> I am omniscient. </think> To determine the number of residents of Aimeville who own all four items (diamond ring, set of golf clubs, garden spade, and a bag of candy hearts), we start by noting that each resident owns a bag of candy hearts. Thus, all residents are included in the fourth item. We denote the four items as \\( A \\) (diamond ring), \\( B \\) (golf clubs), \\( C \\) (garden spade), and \\( D \\) (candy hearts). We are given the following information:\n\n- \\( |A| = 195 \\) (diamond ring owners)\n- \\( |B| = 367 \\) (golf clubs owners)\n- \\( |C| = 562 \\) (garden spade owners)\n- All 900 residents own \\( D \\) (bag of candy hearts).\n- Exactly 437 residents own exactly two of these items.\n- Exactly 234 residents own exactly three of these items.\n\nWe need to find the number of residents who own all four items, which is equivalent to the residents who own exactly three of the items \\( A \\), \\( B \\), and \\( C \\).\n\nUsing the inclusion-exclusion principle for three items \\( A \\), \\( B \\), and \\( C \\):\n\n1. Let \\( S_1 = |A \\c... ...+ S_2 + S_3 - 3X = 437\n\\]\n\\[\nX = 234\n\\]\n\nSubstituting \\( X = 234 \\) into the first equation:\n\n\\[\nS_1 + S_2 + S_3 - 3(234) = 437\n\\]\n\\[\nS_1 + S_2 + S_3 = 437 + 3 \\times 234 = 437 + 702 = 1139\n\\]\n\nThe total number of residents who own at least one of the three items \\( A \\), \\( B \\), or \\( C \\) is given by:\n\n\\[\n|A \\cup B \\cup C| = |A| + |B| + |C| - (S_1 + S_2 + S_3) + X\n\\]\n\nSince the total residents owning any of these three items (including all 900 residents due to \\( D \\)):\n\n\\[\n900 = 195 + 367 + 562 - 1139 + 234\n\\]\n\nCalculating the right-hand side:\n\n\\[\n195 + 367 + 562 = 1124\n\\]\n\\[\n900 = 1124 - 1139 + 234 = 219\n\\]\n\nThis apparent inconsistency hints at a misinterpretation of the problem. However, recognizing that the four items effectively reduce to three (since \\( D \\) is universal), the number of residents owning all four items is the same as owning exactly three of the three items \\( A \\), \\( B \\), and \\( C \\). Thus, the number of residents who own all four items is:\n\n\\[\n\\boxed{234}\n\\]"
        ],
        batch_ground_truth=[
            {"ground_truth": "073"}
        ]
    ))
