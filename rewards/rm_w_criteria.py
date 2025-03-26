import re
import uuid
import time
import random
import requests
from tqdm import tqdm
import xml.etree.ElementTree as ET
from functools import partial
from collections import namedtuple, defaultdict


URLS = [
    "http://10.130.1.205:5001"
]

DEFAULT_PARSE_FAILURE_REWARD = -2.
DEFAULT_RM_REWARD_CLIP = 0.1
DEFAULT_RM_REWARD_CLIP_AMPLIFY = 1.0


def contain_chinese(string):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    if re.search(pattern, string):
        return True
    return False


def simple_tokenize(s):
    if contain_chinese(s):
        return list(s)
    else:
        return s.split(" ")


def batchify(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def is_subject_question(ground_truth):
    if "# JUDGE CRITERIA" in ground_truth or "# 评价标准" in ground_truth:
        return True
    else:
        return False


def post_with_retry(urls, data, max_retries=3, retry_delay=1):
    retries = 0
    while retries < max_retries:
        try:
            url = random.choice(urls)
            response = requests.post(f'{url}/reward', json=data, timeout=30)
            response.raise_for_status()  # 如果状态码不是 200，抛出异常
            return response.json()
        except requests.RequestException as e:
            print(f"请求(数据总量={len(data)})失败，错误信息: {e}，重试第 {retries + 1} 次...")
            retries += 1
            if retries < max_retries:
                time.sleep(retry_delay)
    print("达到最大重试次数，请求失败。")
    return None


def postprocess_solution(solution_str):
    if "<|im_end|>" in solution_str:
        return solution_str[:solution_str.index("<|im_end|>")]
    return solution_str


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# BASE
# ------------------------------------------------------------------------------------------------------------------------------------------------------


def compute_rm_score(
        batch_solution_str,
        batch_ground_truth,
        postprocess_solution_fn,
        parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD,
):
    input_datas = []
    rewards = {}

    for i, (solution_str, ground_truth) in enumerate(zip(batch_solution_str, batch_ground_truth)):
        solution_str = postprocess_solution_fn(solution_str)
        if solution_str is None:
            rewards[i] = parse_result_failure_score
            continue

        input_data = {
            "prompt": ground_truth["ground_truth"], "response": solution_str, "id": i
        }
        input_datas.append(input_data)

    if len(input_datas) > 0:
        for batch in tqdm(batchify(input_datas, n=32), desc='[RM] batchify inference'):
            output_datas = post_with_retry(URLS, batch)
            for _ in output_datas['reward']:
                _id = int(_["id"])
                rewards[_id] = _["rm_score"]

    final_results = []
    for i in range(len(batch_solution_str)):
        if i in rewards:
            final_results.append(rewards[i])
        else:
            final_results.append(0.)
    return final_results

    #         penalty_log = []
    #         # Subject Question
    #         if is_subject.get(i, False):
    #             _reward = rewards[i]
    #         else:
    #             if rewards[i] >= reward_clip:
    #                 _reward = reward_clip_amplify
    #             else:
    #                 _reward = rewards[i]

    #         for name, _penalty in penalty.items():
    #             if i in _penalty:
    #                 _reward += _penalty[i]
    #                 penalty_log.append(f'{name} Penalty={_penalty[i]:.2f}')

    #         final_results.append(_reward)

    #         if split == "valid" and i in logs:
    #             print(f"----------------[VALID]----------------")
    #             print(f"【Solution】 `{repr(clip_str(logs[i][0]))}`")
    #             print(f"【Ground Truth】 `{repr(logs[i][1])}`")
    #             print(f'Reward={_reward};{";".join(penalty_log)}\n')
    #         elif split == "train" and i in logs and random.random() < 0.2:
    #             print(f"----------------[TRAIN]----------------")
    #             print(f"【Solution】 `{repr(clip_str(logs[i][0]))}`")
    #             print(
    #                 f"【Ground Truth】 (is_subject={is_subject_question(logs[i][1])})`{repr(logs[i][1])}`")
    #             print(f'Reward={_reward};{";".join(penalty_log)}\n')
    #     else:
    #         final_results.append(0.)


def compute_score_base(
        batch_data_sources,
        batch_solution_str,
        batch_ground_truth,
        postprocess_solution_fn,
        format_failure_reward=DEFAULT_PARSE_FAILURE_REWARD,
        reward_clip=DEFAULT_RM_REWARD_CLIP,
        reward_clip_amplify=DEFAULT_RM_REWARD_CLIP_AMPLIFY,
        penalty_fn=None,
        norm_ground_truth_fn=None,
        split="train"
):

    input_datas = []
    rewards, logs, is_subject = {}, {}, {}
    penalty = defaultdict(dict)

    for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
        print(ground_truth)
        raise NotImplementedError
        if is_subject_question(ground_truth):
            is_subject[i] = True

        raw_solution_str = solution_str
        solution_str = postprocess_solution_fn(solution_str)
        if solution_str is None:
            rewards[i] = format_failure_reward
            logs[i] = (raw_solution_str, ground_truth)
            continue
        else:
            # Normalize Ground Truth
            # For sake of penalty correctness.
            if norm_ground_truth_fn is not None:
                norm_ground_truth = norm_ground_truth_fn(ground_truth)
            else:
                norm_ground_truth = ground_truth

            if penalty_fn is not None:
                for name, fn in penalty_fn.items():
                    penalty[name][i] = fn(solution_str, norm_ground_truth)
            logs[i] = (solution_str, norm_ground_truth)

        input_data = {
            "prompt": ground_truth, "response": solution_str, "id": i
        }
        input_datas.append(input_data)

    if len(input_datas) > 0:
        for batch in tqdm(batchify(input_datas, n=32), desc='[RM] batchify inference'):
            output_datas = post_with_retry(URLS, batch)
            for _ in output_datas['reward']:
                _id = int(_["id"])
                rewards[_id] = _["rm_score"]

    final_results = []

    def clip_str(s):
        if len(s) > 2000:
            return f'{s[:1000]}... ...{s[-1000:]}'
        return s

    for i in range(len(batch_solution_str)):
        if i in rewards:
            penalty_log = []
            # Subject Question
            if is_subject.get(i, False):
                _reward = rewards[i]
            else:
                if rewards[i] >= reward_clip:
                    _reward = reward_clip_amplify
                else:
                    _reward = rewards[i]

            for name, _penalty in penalty.items():
                if i in _penalty:
                    _reward += _penalty[i]
                    penalty_log.append(f'{name} Penalty={_penalty[i]:.2f}')

            final_results.append(_reward)

            if split == "valid" and i in logs:
                print(f"----------------[VALID]----------------")
                print(f"【Solution】 `{repr(clip_str(logs[i][0]))}`")
                print(f"【Ground Truth】 `{repr(logs[i][1])}`")
                print(f'Reward={_reward};{";".join(penalty_log)}\n')
            elif split == "train" and i in logs and random.random() < 0.2:
                print(f"----------------[TRAIN]----------------")
                print(f"【Solution】 `{repr(clip_str(logs[i][0]))}`")
                print(
                    f"【Ground Truth】 (is_subject={is_subject_question(logs[i][1])})`{repr(logs[i][1])}`")
                print(f'Reward={_reward};{";".join(penalty_log)}\n')
        else:
            final_results.append(0.)

    return final_results


compute_score_nothink = partial(
    compute_score_base, postprocess_solution_fn=postprocess_solution)

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# QwQ LongCoT Reward
# ------------------------------------------------------------------------------------------------------------------------------------------------------


def qwq_longcot_postprocess_solution(solution_str):
    solution_str = postprocess_solution(solution_str)
    # return solution_str
    try:
        thought = re.findall(r'<think>.*</think>',
                             solution_str, re.DOTALL)[0]
    except Exception as err:
        return None

    conclusion = solution_str.replace(thought, "").strip()

    return conclusion


def qwq_longcot_length_penalty(solution_str, ground_truth, length_limit=600):
    if is_subject_question(ground_truth):
        return 0.
    return -0.05 * min(max(len(simple_tokenize(solution_str))-length_limit, 0) / length_limit, 5.)


qwq_longcot_compute_score = partial(
    compute_score_base, postprocess_solution_fn=qwq_longcot_postprocess_solution, penalty_fn={
        "LENGTH": qwq_longcot_length_penalty
    })

qwq_longcot_compute_score_train = partial(
    qwq_longcot_compute_score, split="train")
qwq_longcot_compute_score_valid = partial(
    qwq_longcot_compute_score, split="valid")


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# XML CoT Reward
# ------------------------------------------------------------------------------------------------------------------------------------------------------


def get_thought(solution_str: str):
    thought = re.findall(r'```xml.*```', solution_str, re.DOTALL)[0]
    return thought


def get_conclusion(solution_str: str):
    thought = get_thought(solution_str)
    return solution_str[solution_str.index(thought)+len(thought):].strip()


def parse_xml_cot_solution_score(solution_str):
    try:
        thought = get_thought(solution_str)
    except Exception as err:
        return (-0.1, None, None)

    try:
        conclusion = get_conclusion(solution_str).strip()
        if len(conclusion) == 0:
            return (-0.1, None, None)
    except Exception as err:
        return (-0.1, None, None)

    try:
        thought_content = re.findall(r'```xml(.*)```', thought, re.DOTALL)[0]

    except Exception as err:
        return (-0.1, None, None)

    thought_content = f'<doc> {thought_content} </doc>'
    try:
        root = ET.fromstring(thought_content)
        min_threshold = -0.05
    except Exception as err:
        min_threshold = -0.1

    return (None, min_threshold, conclusion)


def xml_cot_compute_score(batch_data_sources, batch_solution_str, batch_ground_truth):
    input_datas = []
    rewards, reward_min_threshold = {}, {}
    for i, (solution_str, ground_truth) in enumerate(zip(batch_solution_str, batch_ground_truth)):
        solution_str = postprocess_solution(solution_str)
        reward, min_reward_threshold, conclusion = parse_xml_cot_solution_score(
            solution_str)
        if reward is not None:
            rewards[i] = reward
        else:
            reward_min_threshold[i] = min_reward_threshold
            input_data = {
                "prompt": ground_truth, "response": conclusion, "id": i
            }
            input_datas.append(input_data)

    if len(input_datas) > 0:
        for batch in tqdm(batchify(input_datas, n=32), desc='[RM] batchify inference'):
            output_datas = post_with_retry(URLS, batch)
            for _ in output_datas['reward']:
                _id = int(_["id"])
                rewards[_id] = _["rm_score"]
    final_results = []
    for i in range(len(batch_solution_str)):
        if i in rewards:
            if i in reward_min_threshold:
                min_threshold = reward_min_threshold[i]
                final_results.append(max(min_threshold, rewards[i]))
            else:
                final_results.append(rewards[i])
            rewards[i]
        else:
            final_results.append(0.)

    return final_results


# ------------------------------
# Fabricate QA Reward
# ------------------------------

def length_penalty(solution_str, ground_truth):
    return -0.05 * min(abs(len(simple_tokenize(solution_str))-len(simple_tokenize(ground_truth))) / len(simple_tokenize(ground_truth)), 5.)


def fabricate_qa_format_penalty(solution_str):
    if solution_str.startswith('"') and solution_str.endswith('"'):
        return solution_str[1:-1].strip(), 0.
    else:
        return solution_str, -0.1


def fabricate_qa_task_postprocess(solution_str):
    if "[CONCLUSION BEGIN]" not in solution_str or "[CONCLUSION END]" not in solution_str:
        return None
    solution_str = solution_str[solution_str.index(
        "[CONCLUSION BEGIN]")+len("[CONCLUSION BEGIN]"):solution_str.index("[CONCLUSION END]")]

    if "The constructed question is: " in solution_str:
        solution_str = solution_str.replace(
            "The constructed question is: ", "").strip()

    solution_str = solution_str.strip()
    if not solution_str.startswith("**Question:**"):
        return None
    solution_str = solution_str.replace("**Question:**", "").strip()

    return solution_str


def fabricate_qa_compute_score_nothink(batch_data_sources, batch_solution_str, batch_ground_truth, split="train"):
    input_datas = []
    rewards = {}
    len_penalty, format_penalty = {}, {}

    logs = {}

    for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
        solution_str = postprocess_solution(solution_str)
        if data_source == "fabricate_qa":
            raw_solution_str = solution_str
            solution_str = fabricate_qa_task_postprocess(solution_str)

            show_ground_truth = ground_truth
            flag = "# Final Anwer (Authentic Exam)"
            if flag in show_ground_truth:
                show_ground_truth = show_ground_truth[show_ground_truth.index(
                    flag)+len(flag):].strip()
            flag = "## Note"
            if flag in show_ground_truth:
                show_ground_truth = show_ground_truth[:show_ground_truth.index(
                    "## Note")].strip()

            if solution_str is None:
                rewards[i] = -1.
                logs[i] = (raw_solution_str, show_ground_truth)
                continue
            else:
                _len_penalty = length_penalty(solution_str, show_ground_truth)
                len_penalty[i] = _len_penalty
                logs[i] = (solution_str, show_ground_truth)
                solution_str, _format_penalty = fabricate_qa_format_penalty(
                    solution_str)
                format_penalty[i] = _format_penalty

        input_data = {
            "prompt": ground_truth, "response": solution_str, "id": i
        }
        input_datas.append(input_data)

    if len(input_datas) > 0:
        for batch in tqdm(batchify(input_datas, n=32), desc='[RM] batchify inference'):
            output_datas = post_with_retry(URLS, batch)
            for _ in output_datas['reward']:
                _id = int(_["id"])
                rewards[_id] = _["rm_score"]
    final_results = []
    for i in range(len(batch_solution_str)):
        if i in rewards:
            _reward = rewards[i]
            if i in len_penalty:
                _reward += len_penalty[i]
                _reward += format_penalty[i]
            final_results.append(_reward)
            if split == "valid" and batch_data_sources[i] == "fabricate_qa" and i in logs:
                print(f"--------------------------------")
                print(f"【Solution】 `{repr(logs[i][0])}`")
                print(f"【Ground Truth】 `{repr(logs[i][1])}`")
                print(
                    f'Reward={_reward};Length Penalty={len_penalty.get(i, 0.)};Format Penalty={format_penalty.get(i, 0.)}')
        else:
            final_results.append(0.)

    return final_results


fabricate_qa_compute_score_nothink_train = partial(
    fabricate_qa_compute_score_nothink, split="train")
fabricate_qa_compute_score_nothink_valid = partial(
    fabricate_qa_compute_score_nothink, split="train")


if __name__ == "__main__":
    # s = '[STEP1 BEGIN]\nTo begin, I need to ensure that the question I construct will comprehensively cover all the provided key points: "Special relativity," "Velocity addition formula," "Applying the velocity addition formula," and "Simplifying algebraic expressions." The required difficulty level is "Novice," which implies the question should be straightforward and accessible, suitable for someone with a solid understanding of the topic at an undergraduate level. Given the responder\'s background and education level, the question should not delve into complex derivations or advanced problem-solving but should still require the application of the velocity addition formula and basic algebraic manipulation.\n\nThe question must be close-ended, meaning it should have a definitive answer. This ensures that there is a clear solution path and outcome, aligning with the examination\'s need for definitive assessment. The question should be designed to test the responder\'s ability to apply the velocity addition formula in a simple context, while also requiring them to perform basic algebraic simplification, thus covering all the specified skills.\n[STEP1 END]\n\n[STEP2 BEGIN]\nConsidering the "Novice" difficulty level, the question should be designed to be easily understandable and solvable by someone with a foundational grasp of special relativity and the velocity addition formula. It should not involve intricate calculations or deep theoretical insights but should still require the responder to demonstrate their understanding of these concepts in a practical scenario.\n\nThe question should be structured in a way that it naturally incorporates all the key points: it must involve the application of the velocity addition formula, necessitating the responder to perform algebraic simplification to arrive at a solution. This ensures that the question is comprehensive and tests the required skills effectively.\n[STEP2 END]\n\n[CONCLUSION BEGIN]\n**Question:**\nIn a thought experiment, two spaceships, A and B, are moving in the same direction relative to an observer on Earth. Spaceship A is moving at a velocity of \\(0.6c\\) (where \\(c\\) is the speed of light) relative to Earth, and spaceship B is moving at a velocity of \\(0.4c\\) relative to spaceship A. Using the velocity addition formula from special relativity, which is given by \\(u = \\frac{u\' + v}{1 + \\frac{u\'v}{c^2}}\\), where \\(u\'\\) is the velocity of the second object relative to the first, and \\(v\\) is the velocity of the first object relative to the observer, calculate the velocity of spaceship B relative to Earth. Express your answer as a fraction of \\(c\\).\n\n**Knowledge and Skills Covered:**\n- **Special relativity**: The question directly applies the principles of special relativity by using the velocity addition formula.\n- **Velocity addition formula**: The question explicitly requires the application of this formula to calculate the relative velocity.\n- **Applying the velocity addition formula**: The responder must use the formula to perform the necessary calculations.\n- **Simplifying algebraic expressions**: The responder needs to simplify the resulting algebraic expression to find the final velocity.\n\n**Difficulty Level: Novice**\n- The question is designed to be straightforward, requiring basic application of a known formula and simple algebraic manipulation, suitable for someone with a solid understanding of the topic at an undergraduate level.\n\n**Solutions:**\n- The velocity addition formula is applied as \\(u = \\frac{0.4c + 0.6c}{1 + \\frac{(0.4c)(0.6c)}{c^2}} = \\frac{1.0c}{1 + 0.24} = \\frac{1.0c}{1.24} = \\frac{100}{124}c = \\frac{25}{31}c\\).\n- The solution involves direct substitution into the formula, basic algebraic simplification, and fraction reduction, ensuring the question is solvable and the process is clear and accessible for a Novice level responder.\n[CONCLUSION END]'
    # print(fabricate_qa_task_postprocess(s))
    # print(length_penalty(s, s))

    # print(qwq_longcot_compute_score_valid(
    #     [None] * len(batch_solution_str), batch_solution_str, batch_ground_truth))
    import json

    def grid_search_rm_threshold(num=1000):
        TEST_CASE = "/cpfs01/shared/llm_ddd/tongjian/ddm/thought_xml/verify_enhance/xml_verify_enhance_v2.jsonl"

        batch_solution_str, batch_ground_truth = [], []
        correct_indices, wrong_indices = [], []
        with open(TEST_CASE, "rt") as f:
            for i, line in enumerate(f):
                example = json.loads(line)
                try:
                    prompt = example["self_improvement"]["prompt"]
                    corrects = example["self_improvement"]["responses"]
                    wrongs = example["self_improvement"]["wrong_responses"]
                    if len(corrects) > 0 and len(wrongs) > 0:
                        correct = random.choice(corrects)
                        wrong = random.choice(wrongs)

                        batch_solution_str.append(correct["conclusion"])
                        batch_ground_truth.append({
                            "ground_truth": f'{prompt}\n\n\n\nJudge only by determining whether the final answer is correct.\n** Final Answer: {example["self_improvement"]["reference_meta"]["final_answer"]}',
                            "extra_info": {
                                "question_type": "object"
                            }
                        })
                        correct_indices.append(len(batch_ground_truth)-1)

                        batch_solution_str.append(wrong["conclusion"])
                        batch_ground_truth.append({
                            "ground_truth": f'{prompt}\n\n\n\nJudge only by determining whether the final answer is correct.\n** Final Answer: {example["self_improvement"]["reference_meta"]["final_answer"]}',
                            "extra_info": {
                                "question_type": "object"
                            }
                        })
                        wrong_indices.append(len(batch_ground_truth)-1)

                except Exception as err:
                    continue
                if i > num:
                    break

        for i, score in enumerate(compute_rm_score(batch_solution_str, batch_ground_truth, postprocess_solution)):
            print(score, i)
            # "ground_truth": example["self_improvement"]["prompt"] + criteria,
            # "extra_info": {
            #     "uuid": example["uuid"],
            #     "question_type": question_type
            # }

    grid_search_rm_threshold()
