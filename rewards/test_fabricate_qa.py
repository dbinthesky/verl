import os
import re
import time
import json
import random
import string
import unittest
import aiohttp
import pandas as pd
import asyncio as aio
from tqdm import tqdm
from collections import defaultdict
from fabricate_qa import (
    agent,
    criteria_parse_solution_fn,
    get_total_score,
    fabricate_parse_solution_fn,
    question_constraint,
    decode_to_question,
    criteria_get_score,
    question_similarity,
    QwQLongCoTCreateCriteriaComputeScore,
    qwq_longcot_create_criteria_compute_score_valid,
    FabricateQATooLongPenalty,
    BleuSimilarity,
    fabricate_parse_solution_fn,
    QwQLongCoTFabricateQAComputeScore,
    qwq_longcot_fabricate_qa_compute_score_valid,
    doc2query_parse_solution_fn,
    Doc2QueryFormatReward,
    QuestionSimilarity,
    RuleBasedOptionMatch,
    QwQLongCoTDoc2QueryComputeScore,
    qwq_longcot_doc2query_compute_score_valid,
    batchify
)


def generate_random_string(n):
    all_characters = string.ascii_letters + string.digits + " "
    return ''.join(random.choice(all_characters) for _ in range(n))


def load_criteria():
    filename = "/cpfs01/shared/llm_ddd/tongjian/verl/rewards/fabricate_qa_criteria.json"
    batch_solution_str, batch_ground_truth = [], []

    with open(filename, "rt") as f:
        data = json.load(f)
    batch_solution_str, batch_ground_truth = data["batch_solution_str"], data["batch_ground_truth"]
    return batch_solution_str, batch_ground_truth


def load_qwq_fabricate_qa_data(num=100):
    filename = "/cpfs01/shared/llm_ddd/tongjian/rl/fabricate_qa/super_gpqa_aio_noneasy_test_0517.parquet"
    batch_solution_str, batch_ground_truth = [], []

    df = pd.read_parquet(filename)
    for _, row in df.iterrows():
        row = row.to_dict()
        batch_ground_truth.append(row["reward_model"])
        gt = row["reward_model"]["authentic_question"]

        batch_solution_str.append(
            f'<think>\n{generate_random_string(100)}\n</think>\n\n<question>{gt}\n{generate_random_string(2000)}</question>')
    return batch_solution_str, batch_ground_truth


def load_doc2query(num=40):
    # path = "/cpfs01/shared/llm_ddd/tongjian/rl/doc2query/super_gpqa_iscalc_high_equation_mix"
    path = "/cpfs01/shared/llm_ddd/tongjian/rl/doc2query/super_gpqa_test"
    batch_solution_str, batch_ground_truth = [], []

    df = pd.read_parquet(path)
    for i, row in df.iterrows():
        row = row.to_dict()
        batch_ground_truth.append(row["reward_model"])
        gt = row["reward_model"]

        if i > num-1:
            break
        try:
            options = []
            for x, y in zip(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"], gt["options"]):
                options.append(f'{x}) {y}')
            options = "\n".join(options)
            ans_letter = gt["options"].tolist().index(gt["answer"])
            ans_letter = ["A", "B", "C", "D", "E", "F", "G", "H",
                          "I", "J", "K", "L", "M", "N", "O", "P"][ans_letter]
            batch_solution_str.append(
                f'<think>***</think><question>\nQuestion: {gt["question"]}\n\nOptions:\n{options}\n\nAnswer: {ans_letter}\n</question><｜end▁of▁sentence｜>')
            # batch_solution_str.append(
            #     f'<think>***</think><question>\nQuestion: {gt["question"]}\n\nOptions:\n\nAnswer: {ans_letter}\n</question>')
        except Exception as err:
            batch_solution_str.append(
                f'<think>***</think><question>\nQuestion: {gt["question"]}\n\nOptions:\nA) {gt["answer"]}\nAnswer: A\n</question><｜end▁of▁sentence｜>')
            # batch_solution_str.append(
            #     f'<think>***</think><question>\nQuestion: Using a 0.1000 mol/L NaOH solution to titrate a 0.1000 mol/L formic acid solution, what is the pH at the stoichiometric point? \n\nOptions:\nA) 5.67\nB) 8.23\nC) 9.88\nD) 12.46\nE) 10.11\nF) 11.07\nG) 7.22\nH) 6.35\nI) 3.47\nJ) 4.55\n\nAnswer: A\n</question><｜end▁of▁sentence｜>')
    return batch_solution_str, batch_ground_truth


async def create_mock_data():
    df = pd.read_parquet(
        "/cpfs01/shared/llm_ddd/tongjian/rl/fabricate_qa_criteria/super_gpqa_aio_noneasy_train_0513_criteria_test_0514.parquet")

    prompt_mapper = {}
    for _, row in df.iterrows():
        row = row.to_dict()
        prompt = f'{row["prompt"][0]["content"]}\n\n\n{row["prompt"][1]["content"]}'
        prompt_mapper[prompt] = row

    prompts = prompt_mapper.keys()

    batch_solution_str, batch_ground_truth = [], []

    results = await agent.run(prompts, 32, desc="[MOCK DATASET]", postprocess_fns=[lambda x: x]*len(prompts))
    for prompt, response in results:
        example = prompt_mapper[prompt]
        rm = example["reward_model"]
        rm = {
            "positive": rm["positive"],
            "negatives": rm["negatives"].tolist(),
        }
        batch_ground_truth.append(rm)
        batch_solution_str.append(response)
    with open("fabricate_qa_criteria.json", "wt") as f:
        json.dump({"batch_ground_truth": batch_ground_truth,
                  "batch_solution_str": batch_solution_str}, f, ensure_ascii=False)


class TestFabricateQA(unittest.TestCase):
    def test_fabricate_qa_too_long_penalty(self):
        penalty_fn = FabricateQATooLongPenalty(
            postprocess_solution_fn=fabricate_parse_solution_fn,
        )
        batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data()
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty_fn.get_penalty_or_reward(solution_str, ground_truth))

    def test_bleu_similarity(self):
        penalty_fn = BleuSimilarity(
            postprocess_solution_fn=fabricate_parse_solution_fn,
        )
        batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data()
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty_fn.get_penalty_or_reward(solution_str, ground_truth))

    def test_rm_similarity(self):
        async def main():
            batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data(
                num=100)
            task = QwQLongCoTFabricateQAComputeScore(split="valid")
            results = await task.rm_similarity(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth
            )
            print(results)
        aio.run(main())

    def test_rm_criteria_checklist(self):
        async def main():
            batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data(
                num=100)
            task = QwQLongCoTFabricateQAComputeScore(split="valid")
            results = await task.rm_criteria_checklist(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth
            )
            print(results)
        aio.run(main())

    def test_llm_as_judge_criteria_checklist(self):
        async def main():
            batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data(
                num=100)
            task = QwQLongCoTFabricateQAComputeScore(split="valid")
            results = await task.llm_as_judge_criteria_checklist(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth
            )
            print(results)
        aio.run(main())

    def test_llm_as_judge_similarity(self):
        async def main():
            batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data(
                num=100)
            task = QwQLongCoTFabricateQAComputeScore(split="valid")
            results = await task.llm_as_judge_similarity(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth
            )
            print(results)
        aio.run(main())

    def test_question_constraint(self):
        async def main():
            batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data(
                num=100)

            fabricates = []
            for _ in batch_solution_str:
                fabricates.append(fabricate_parse_solution_fn(_))

            task = QwQLongCoTFabricateQAComputeScore(split="valid")
            results = await task.question_constraint(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth
            )
            print(results)
        aio.run(main())

    def test_compute_score(self):
        # async def main():
        #     batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data(
        #         num=100)
        #     task = QwQLongCoTFabricateQAComputeScore(split="valid")
        #     results = await task._compute_score(
        #         [None] *
        #         len(batch_solution_str), batch_solution_str, batch_ground_truth
        #     )
        #     print(results)
        # aio.run(main())
        bg = time.time()
        batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data(
            num=100)
        results = qwq_longcot_fabricate_qa_compute_score_valid(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth
        )
        print(results)
        print(f'Finish {time.time()-bg}')


class TestCriteria(unittest.TestCase):
    def test_criteria_get_score(self):
        batch_solution_str, batch_ground_truth = load_criteria()

        x, y = [], []
        for solution_str, gt in zip(batch_solution_str, batch_ground_truth):
            criteria = criteria_parse_solution_fn(solution_str)
            if criteria is not None:
                x.append(random.choice(gt["negatives"]))
                y.append(criteria)

        async def main():
            print(await criteria_get_score(x, y))
        aio.run(main())

    def test_decode_to_question(self):
        batch_solution_str, batch_ground_truth = load_criteria()

        x, y = [], []
        for solution_str, gt in zip(batch_solution_str, batch_ground_truth):
            criteria = criteria_parse_solution_fn(solution_str)
            if criteria is not None:
                x.append(random.choice(gt["negatives"]))
                y.append(criteria)

        y = y[:10]

        async def main():
            print(await decode_to_question(y))
        aio.run(main())

    def test_question_similarity(self):
        batch_solution_str, batch_ground_truth = load_criteria()

        x, y = [], []
        for solution_str, gt in zip(batch_solution_str, batch_ground_truth):
            criteria = criteria_parse_solution_fn(solution_str)
            if criteria is not None:
                x.append(gt["positive"])
                y.append(random.choice(gt["negatives"]))

        async def main():
            print(await question_similarity(x, y))
        aio.run(main())

    def test_calc_compression_ratio_reward(self):
        task = QwQLongCoTCreateCriteriaComputeScore()
        batch_solution_str, batch_ground_truth = load_criteria()

        async def main():
            print(await task.calc_compression_ratio_reward(
                [None]*len(batch_solution_str),
                batch_solution_str,
                batch_ground_truth))
        aio.run(main())

    def test_calc_classify_acc_reward(self):
        task = QwQLongCoTCreateCriteriaComputeScore(split="valid")
        batch_solution_str, batch_ground_truth = load_criteria()
        batch_solution_str = batch_solution_str[:10]
        batch_ground_truth = batch_ground_truth[:10]

        # async def main():
        #     print(await qwq_longcot_create_criteria_compute_score_valid(
        #         [None]*len(batch_solution_str),
        #         batch_solution_str,
        #         batch_ground_truth))
        # aio.run(main())
        qwq_longcot_create_criteria_compute_score_valid(
            [None]*len(batch_solution_str),
            batch_solution_str,
            batch_ground_truth)


def load_criteria_infer():
    filename = "/cpfs01/shared/llm_ddd/tongjian/rl/hard_case_mixed/gpqa/super_gpqa_aio_noneasy_train_0513_criteria_output.jsonl"

    def preprocess(batch):
        batch_solution_str, batch_ground_truth = [], []
        for elem in batch:
            batch_solution_str.append(
                elem["self_improvement"]["responses"][0]["response"]["text"])
            batch_ground_truth.append(
                {
                    "positive": elem["self_improvement"]["question"],
                    "negatives": elem["self_improvement"]["fabricate_questions"]
                }
            )
        return batch_solution_str, batch_ground_truth, batch

    batch = []
    with open(filename, "rt") as f:
        for line in f:
            example = json.loads(line)
            batch.append(example)

            if len(batch) == 1024:
                yield preprocess(batch)
                batch = []
    if len(batch):
        yield preprocess(batch)


async def offline_compute_score():
    with open("/cpfs01/shared/llm_ddd/tongjian/rl/hard_case_mixed/gpqa/_super_gpqa_aio_noneasy_train_0513_criteria_output2.jsonl", "wt") as g:
        max_concurrent_requests = 256
        task = QwQLongCoTCreateCriteriaComputeScore(split="train")
        for batch_solution_str, batch_ground_truth, batch_examples in load_criteria_infer():
            data_sources = [None] * len(batch_solution_str)
            score1, score2 = await task.calc_classify_acc_reward(
                data_sources,
                batch_solution_str, batch_ground_truth,
                max_concurrent_requests=max_concurrent_requests,
                return_single_score=False
            )
            calc_compression_ratio_reward = await task.calc_compression_ratio_reward(
                data_sources,
                batch_solution_str, batch_ground_truth,
                max_concurrent_requests=max_concurrent_requests
            )
            for example, _score1, _score2, _score3 in zip(batch_examples, score1, score2, calc_compression_ratio_reward):
                example["calc_compression_ratio_reward"] = _score3
                example["classify_acc_reward"] = _score1
                example["gt_match_score"] = _score2
                g.write(f'{json.dumps(example, ensure_ascii=False)}\n')


class TestDoc2Query(unittest.TestCase):

    def test_get_difficulty_reward(self):
        async def main():
            batch_solution_str, batch_ground_truth = load_doc2query()
            task = QwQLongCoTDoc2QueryComputeScore(split="valid")
            results = await task.get_difficulty_reward(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth
            )
            print(results)
        aio.run(main())

    def test_chat_completion_with_retry(self):
        async def main():
            batch_solution_str, batch_ground_truth = load_doc2query(32*8)
            task = QwQLongCoTDoc2QueryComputeScore(split="valid")
            prompts = []

            instruct = 'Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format "Answer: $LETTER" (without quotes), where LETTER is one of the option letters. You must first think step by step with very detail thinking process.'

            for _ in batch_solution_str:
                result = doc2query_parse_solution_fn(_)
                if result is None:
                    continue
                question, options, answer = result
                prompts.append(task.format_question(
                    question, options, None) + f"\n\n{instruct}")

            # results = await task.chat_completion_with_retry(
            #     "http://10.130.0.245:5002", prompts
            # )
            prompts = prompts * 16
            s1 = time.time()
            results = await task.generate_responses(
                prompts
            )
            # for p, r in zip(prompts, results):
            #     print(p)
            #     print(r)
            #     print("="*80)
            # print(f'Finish: {time.time()-s1}s')
        aio.run(main())

    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_doc2query(32)

        task = QwQLongCoTDoc2QueryComputeScore(split="valid")
        print(qwq_longcot_doc2query_compute_score_valid([None]*len(batch_solution_str),
                                                        batch_solution_str, batch_ground_truth))


def doc2query_format_filter(path):
    outputs = []
    with open(path, "rt") as f:
        for line in f:
            example = json.loads(line)
            prompt = example["self_improvement"]["chat_completion"]
            option_num = int(re.findall(
                r'- Number of options: (\d+)', prompt)[0].strip())
            difficulty = re.findall(
                r'- Difficulty level: (\w+)', prompt)[0].strip()

            response = example["self_improvement"]["responses"][0]["response"]["text"]
            result = doc2query_parse_solution_fn(response)
            if result is None:
                continue
            question, options, answer = result
            if len(options) != option_num:
                continue
            try:
                ans_index = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
                             'H', 'I', 'J', 'K', 'L', 'M'].index(answer)
            except Exception as err:
                continue
            if not (ans_index <= (len(options)-1)):
                continue
            prompt = prompt[prompt.index("<|im_start|>user") +
                            len("<|im_start|>user"):prompt.index("<|im_end|>\n<|im_start|>assistant")].strip()

            output = {
                "system": "You are a helpful assistant and an expert in creating a question. Now your task is to create a question. You first thinks about the reasoning process in the mind and then provides the user with the answer. Show your reasoning process in <think> </think> tags, and your final question creation in <question> </question> tags. Remember the part in <question> </question> tags MUST only be the question you created ** WITHOUT ** any explanation or solution.",
                "instruction": prompt,
                "input": "",
                "output": response
            }
            outputs.append(output)

    with open("/cpfs01/shared/llm_ddd/tongjian/synthetic/sft/fabricate_qa_v2.json", "rt") as f:
        mix = json.load(f)
        print(len(mix))

    mix.extend(outputs)
    random.shuffle(mix)

    with open("/cpfs01/shared/llm_ddd/tongjian/synthetic/sft/fabricate_qa_v3.json", "wt") as f:
        json.dump(mix, f, ensure_ascii=False)
    print(len(mix))


def doc2query_bon_merge(path, output):
    outputs = []
    scorer = RuleBasedOptionMatch()
    group = defaultdict(list)

    pbar = tqdm(total=736928)

    with open(path, "rt") as f:
        for line in f:
            example = json.loads(line)
            pbar.update(1)
            prompt = example["self_improvement"]["chat_completion"]
            response = example["self_improvement"]["responses"][0]["response"]["text"]
            result = doc2query_parse_solution_fn(response)
            if result is None:
                continue
            question, options, answer = result
            try:
                ans_index = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
                             'H', 'I', 'J', 'K', 'L', 'M'].index(answer)
            except Exception as err:
                continue
            if not (ans_index <= (len(options)-1)):
                continue

            score = scorer.get_penalty_or_reward(response, {
                "question": example["self_improvement"]["question"],
                "answer": example["self_improvement"]["answer"],
                "options": example["self_improvement"]["options"],
                "difficulty": example["self_improvement"]["difficulty"],
            })
            group[example["uuid"]].append({
                "response": response,
                "score": score
            })

    done = {}
    with open(output, 'wt') as g:
        with open(path, "rt") as f:
            for line in f:
                example = json.loads(line)
                if example["uuid"] not in done:
                    done[example["uuid"]] = True
                    del example["self_improvement"]["responses"]
                    example["self_improvement"]["responses"] = group[example["uuid"]]
                    g.write(f'{json.dumps(example, ensure_ascii=False)}\n')


def doc2query_difficulty_filter(path, output):
    pbar = tqdm(total=736928)

    with open(output, "wt") as g:
        with open(path, "rt") as f:
            for line in f:
                example = json.loads(line)
                pbar.update(1)
                responses = example["self_improvement"]["responses"]

                threshold = 0.5
                filtered = [_ for _ in responses if _["score"] > threshold]
                if len(filtered) < 7 and len(filtered):
                    del example["self_improvement"]["responses"]
                    del example["self_improvement"]["chat_completion"]
                    g.write(f'{json.dumps(example, ensure_ascii=False)}\n')


def doc2query_fabricate_qa_difficulty_reward(path, output):

    for filename in tqdm(os.listdir(path)):
        basename = os.path.basename(filename)
        outputs = []
        filename = os.path.join(path, filename)
        with open(filename, "rt") as f:
            for line in f:
                example = json.loads(line)
                outputs.append(
                    (
                        example, example["self_improvement"]["responses"][0]["response"]["text"],
                        {
                            "document": example["content"],
                        }
                    )
                )
        output_path = f'{output}_{basename}'
        if os.path.exists(output_path):
            continue

        async def main():
            task = QwQLongCoTDoc2QueryComputeScore(
                split="valid", difficulty_bon=4)
            with open(output_path, "wt") as f:
                for batch in batchify(outputs, n=1024):
                    examples = [_[0] for _ in batch]
                    batch_solution_str, batch_ground_truth = [
                        _[1] for _ in batch], [_[2] for _ in batch]
                    results, pass_rates = await task.get_difficulty_reward(
                        [None] *
                        len(batch_solution_str), batch_solution_str, batch_ground_truth, repeat=4
                    )
                    for example, result, pass_rate in zip(examples, results, pass_rates):
                        example["self_improvement"]["difficulty_reward"] = result
                        example["self_improvement"]["pass_rate"] = pass_rate
                        f.write(f'{json.dumps(example, ensure_ascii=False)}\n')
        aio.run(main())


def doc2query_postprocess(path, output):
    task = QwQLongCoTDoc2QueryComputeScore(split="valid")

    with open(output, "wt") as g:
        for filename in tqdm(os.listdir(path)):
            filename = os.path.join(path, filename)
            with open(filename, "rt") as f:
                for line in f:
                    example = json.loads(line)
                    response = example["self_improvement"]["responses"][0]["response"]["text"]
                    result = doc2query_parse_solution_fn(
                        response, remove_option_letter=False)
                    if result is None:
                        continue
                    question, options, answer = result
                    try:
                        ans_index = task.MULTICHOICE_LETTER.index(answer)
                    except Exception as err:
                        continue
                    if len(options) < 40:
                        continue
                    if ans_index > len(options) - 1 or not options[ans_index].startswith(f'{answer})'):
                        continue
                    result = doc2query_parse_solution_fn(
                        response, remove_option_letter=True)
                    question, options, answer = result
                    example["self_improvement"]["synthetic_qa_prompt"] = example["self_improvement"]["chat_completion"]
                    del example["self_improvement"]["chat_completion"]
                    example["self_improvement"]["synthetic_qa_response"] = response
                    example["self_improvement"]["question"] = question
                    answer_content = options[ans_index]

                    random.shuffle(options)

                    try:
                        example["self_improvement"]["options"] = options
                        example["self_improvement"]["answer"] = answer_content
                        example["self_improvement"]["answer_letter"] = task.MULTICHOICE_LETTER[options.index(
                            answer_content)]
                        example["self_improvement"]["prompt"] = 'Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format "Answer: $LETTER" (without quotes), where LETTER is one of the option letters.\n\n' + task.format_question(
                            question=question, options=options, answer=None
                        )

                        g.write(f'{json.dumps(example, ensure_ascii=False)}\n')
                    except Exception as err:
                        continue


if __name__ == '__main__':
    # async def main():
    #     await create_mock_data()
    # aio.run(main())
    # unittest.main()

    # async def main():
    #     await offline_compute_score()
    # aio.run(main())

    # doc2query_difficulty_filter(
    #     path="/cpfs01/shared/llm_ddd/tongjian/rl/hard_case_mixed/gpqa/super_gpqa_train_bo32.jsonl",
    #     output="/cpfs01/shared/llm_ddd/tongjian/rl/hard_case_mixed/gpqa/super_gpqa_train_bo32_results.jsonl",
    # )

    # doc2query_difficulty_filter(
    #     path="/cpfs01/shared/llm_ddd/tongjian/rl/hard_case_mixed/gpqa/super_gpqa_train_bo32.jsonl",
    #     output="/cpfs01/shared/llm_ddd/tongjian/rl/hard_case_mixed/gpqa/super_gpqa_train_pass6@32.jsonl",
    # )

    # doc2query_postprocess(
    #     path="/cpfs01/shared/llm_ddd/tongjian/pretrain_archive/doc2query_supergpqa_recall_0520.output",
    #     output="/cpfs01/shared/llm_ddd/tongjian/doc2query/doc2query_supergpqa_recall_0520_qa_0526.jsonl"
    # )

    doc2query_fabricate_qa_difficulty_reward(
        path="/cpfs01/shared/llm_ddd/tongjian/sft/self_improvement/high_equation_20k_rft_input_r8.output",
        output="shit"
    )
