import re
import uuid
import time
import random
import requests
from tqdm import tqdm
from collections import namedtuple
import xml.etree.ElementTree as ET


URLS = [
    # [he]
    # "http://10.130.1.180:5004",
    # [hc]
    # "http://10.130.1.54:5001"
    # [ddd]
    "http://10.130.1.205:5001"
]


def batchify(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def get_thought(solution_str: str):
    thought = re.findall(r'```xml.*```', solution_str, re.DOTALL)[0]
    return thought


def get_conclusion(solution_str: str):
    thought = get_thought(solution_str)
    return solution_str[solution_str.index(thought)+len(thought):].strip()


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


def compute_score(batch_data_sources, batch_solution_str, batch_ground_truth):
    input_datas = []
    rewards, reward_min_threshold = {}, {}
    for i, (solution_str, ground_truth) in enumerate(zip(batch_solution_str, batch_ground_truth)):
        solution_str = postprocess_solution(solution_str)
        reward, min_reward_threshold, conclusion = parse_solution_score(
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


def fabricate_qa_task_postprocess(solution_str):
    if "[CONCLUSION BEGIN]" not in solution_str or "[CONCLUSION END]" not in solution_str:
        return None
    solution_str = solution_str[solution_str.index(
        "[CONCLUSION BEGIN]")+len("[CONCLUSION BEGIN]"):solution_str.index("[CONCLUSION END]")]

    if "The constructed question is: " in solution_str:
        solution_str = solution_str.replace(
            "The constructed question is: ", "").strip()
    if solution_str.startswith('\"') and solution_str.endswith('"'):
        solution_str = solution_str[1:-1].strip()
    return solution_str.strip()


def compute_score_nothink(batch_data_sources, batch_solution_str, batch_ground_truth):
    input_datas = []
    rewards = {}
    for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
        solution_str = postprocess_solution(solution_str)
        if data_source == "fabricate_qa":
            solution_str = fabricate_qa_task_postprocess(solution_str)
            if solution_str is None:
                rewards[i] = -1.
                continue
            else:
                show_ground_truth = ground_truth
                flag = "# Final Anwer (Authentic Exam)"
                if flag in show_ground_truth:
                    show_ground_truth = show_ground_truth[show_ground_truth.index(flag)+len(flag):].strip()
                
                print(f"--------------------------------")
                print(f"Solution: {repr(solution_str)}")
                print(f"Ground Truth: {repr(show_ground_truth)}")

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
            final_results.append(rewards[i])
        else:
            final_results.append(0.)

    return final_results


def parse_solution_score(solution_str):
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


if __name__ == "__main__":
    import json

    TEST_CASE = "/cpfs01/shared/llm_ddd/tongjian/ddm/thought_xml/verify_enhance/xml_verify_enhance_v2.jsonl"

    batch_solution_str, batch_ground_truth = [], []
    with open(TEST_CASE, "rt") as f:
        for i, line in enumerate(f):
            example = json.loads(line)
            if "top_response" in example["self_improvement"]:
                prompt = example["self_improvement"]["prompt"]
                response = example["self_improvement"]["top_response"]["response"]
                batch_solution_str.append(response)
                batch_ground_truth.append(prompt)
            if i > 100:
                break

    print(compute_score(batch_solution_str, batch_ground_truth))
