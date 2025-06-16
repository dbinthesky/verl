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
    # criteria_parse_solution_fn,
    # get_total_score,
    # decode_to_question,
    # criteria_get_score,
    # QwQLongCoTCreateCriteriaComputeScore,
    # qwq_longcot_create_criteria_compute_score_valid,
    # QwQLongCoTFabricateQAComputeScore,
    # qwq_longcot_fabricate_qa_compute_score_valid,
    # doc2query_parse_solution_fn,
    QuestionSimilarity,
    CalculationAnswerFormatVerify,
    # QwQLongCoTDoc2QueryComputeScore,
    # QwQLongCoTDoc2QueryV2ComputeScore,
    # qwq_longcot_doc2query_compute_score_valid,
    custom_qa_parse_solution_fn,
    batchify,
    WithUnitSymbol,
    NumericalAnswer
)


def generate_random_string(n):
    all_characters = string.ascii_letters + string.digits + " "
    return ''.join(random.choice(all_characters) for _ in range(n))


def load_fabricate_aio_data(num=100, format="wrong_question"):
    filename = "/cpfs01/shared/llm_ddd/tongjian/rl/fabricate_aio/fabricate_aio_train_0616.parquet"
    batch_solution_str, batch_ground_truth = [], []

    df = pd.read_parquet(filename)
    for _, row in df.iterrows():
        row = row.to_dict()
        if format == "wrong_question":
            gt = row["reward_model"]["question"]
            if gt is not None:
                batch_solution_str.append(
                    f'<think>\n{generate_random_string(100)}\n</think>\n\n<question>\nQuestion: {gt}\n\nAnswer: \\boxed{{78}}\n\nAnswer Type: NumericalAnswer\n</question>')
                batch_ground_truth.append(row["reward_model"])
            else:
                continue
        if _ > num-1:
            break
    return batch_solution_str, batch_ground_truth


# def load_doc2query_v2(num=40):
#     path = "/cpfs01/shared/llm_ddd/tongjian/rl/doc2query_v2/iscalc_numeric_high_equation_mix_0608"
#     batch_solution_str, batch_ground_truth = [], []

#     df = pd.read_parquet(path)
#     for i, row in df.iterrows():
#         row = row.to_dict()
#         batch_ground_truth.append(row["reward_model"])
#         gt = row["reward_model"]

#         if i > num-1:
#             break
#         try:
#             assert gt["question"] is not None
#             batch_solution_str.append(
#                 f'<think>***</think><question>\nQuestion: {gt["question"]}\\n\nAnswer: {gt["answer"]}\n\nAnswer Type:{gt["answer_type"]}\n</question><｜end▁of▁sentence｜>')
#         except Exception as err:
#             question = random.choice(gt["fabricate_questions"])
#             ans = "\\boxed{18}"
#             batch_solution_str.append(
#                 f'<think>***</think><question>\nQuestion: {question}\n\nAnswer: {ans}\n\nAnswer Type: NumericalAnswer\n</question><｜end▁of▁sentence｜>')
#     return batch_solution_str, batch_ground_truth


# class TestFabricateQA(unittest.TestCase):
#     def test_question_similarity(self):
#         batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data(
#             num=100)
#         task = QwQLongCoTFabricateQAComputeScore(split="valid")
#         for solution_str, gt in zip(batch_solution_str, batch_ground_truth):
#             solution_str = f'<think>***</think><question>\nQuestion: {gt["authentic_question"]}\nAnswer: \\boxed{{-9}}\nAnswer Type: NumericalAnswer\n</question>'
#             score = task.get_penalties()["QSim"](solution_str, gt)
#             print(solution_str)
#             print(score)
#             print("="*80)
#             break

#     def test_llm_as_judge_similarity(self):
#         async def main():
#             batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data(
#                 num=100)
#             task = QwQLongCoTFabricateQAComputeScore(split="valid")
#             results = await task.llm_as_judge_similarity(
#                 [None] *
#                 len(batch_solution_str), batch_solution_str, batch_ground_truth
#             )
#             print(results)
#         aio.run(main())

#     def test_get_difficulty_reward(self):
#         async def main():
#             batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data(
#                 8)
#             task = QwQLongCoTFabricateQAComputeScore(split="valid")
#             results = await task.get_difficulty_reward(
#                 [None] *
#                 len(batch_solution_str), batch_solution_str, batch_ground_truth
#             )
#             print(results)
#         aio.run(main())

#     def test_compute_score(self):
#         bg = time.time()
#         batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data(
#             num=32)
#         results = qwq_longcot_fabricate_qa_compute_score_valid(
#             [None] *
#             len(batch_solution_str), batch_solution_str, batch_ground_truth
#         )
#         print(results)
#         print(f'Finish {time.time()-bg}')


class TestFabricate(unittest.TestCase):
    def test_with_unit_symbol(self):
        verifier = WithUnitSymbol()
        self.assertEqual(verifier.verify("1.275 mol"), True)
        self.assertEqual(verifier.verify("1.256 × 10^-67 J"), True)
        self.assertEqual(verifier.verify("9.42 pc"), True)
        self.assertEqual(verifier.verify("50.1 g"), True)
        self.assertEqual(verifier.verify("5.27×10^5 Pa"), True)
        self.assertEqual(verifier.verify("1.812 meV"), True)
        self.assertEqual(verifier.verify("5.27×10^5 Pa"), True)
        self.assertEqual(verifier.verify("6.100×10^11 Hz"), True)
        self.assertEqual(verifier.verify("0.010 °"), True)
        self.assertEqual(verifier.verify("10.938 kg/m²"), True)
        self.assertEqual(verifier.verify("12288 KB"), True)
        self.assertEqual(verifier.verify("1.256 × 10^-67 J"), True)
        self.assertEqual(verifier.verify("1.720 ppm"), True)
        self.assertEqual(verifier.verify("12.000 μm"), True)
        self.assertEqual(verifier.verify("10.870 kJ/g"), True)
        self.assertEqual(verifier.verify("127.500 MeV"), True)
        self.assertEqual(verifier.verify("-0.854 kJ"), True)
        self.assertEqual(verifier.verify("2560 m²"), True)
        self.assertEqual(verifier.verify("\\boxed{3.970e13}"), True)
        self.assertEqual(verifier.verify("\\boxed{82.6%}"), True)
        self.assertEqual(verifier.verify("\\boxed{-45.3%}"), True)
        self.assertEqual(verifier.verify("2.006 Å"), True)
        self.assertEqual(verifier.verify("1.256 × 10^{-67} J"), False)
        self.assertEqual(verifier.verify("5.27×10^{5}"), False)
        self.assertEqual(verifier.verify("2.73 × 10⁻²³²³ J/(K·m²)"), False)
        self.assertEqual(verifier.verify("342 lb_f"), False)

        verifier = NumericalAnswer()
        self.assertEqual(verifier.verify("\\boxed{0.00000225}"), True)
        self.assertEqual(verifier.verify("\\boxed{0.000186}"), True)
        self.assertEqual(verifier.verify("\\boxed{426}"), True)
        self.assertEqual(verifier.verify("\\boxed{-854}"), True)
        self.assertEqual(verifier.verify("\\boxed{0.00474}"), True)
        self.assertEqual(verifier.verify("\\boxed{0.000180}"), True)
        self.assertEqual(verifier.verify("\\boxed{0.0213}"), True)
        self.assertEqual(verifier.verify("\\boxed{46.8}"), True)
        self.assertEqual(verifier.verify("\\boxed{1.8}"), False)
        self.assertEqual(verifier.verify("\\boxed{56.35}"), True)
        self.assertEqual(verifier.verify("\\boxed{1.77e15}"), True)
        self.assertEqual(verifier.verify("\\boxed{-0.8}"), False)
        self.assertEqual(verifier.verify("\\boxed{1.667e-06}"), True)
        self.assertEqual(verifier.verify("\\boxed{202.5}"), True)
        self.assertEqual(verifier.verify("\\boxed{2.470e16}"), True)
        self.assertEqual(verifier.verify("\\boxed{7.800e10}"), True)
        self.assertEqual(verifier.verify("\\boxed{-32.4}"), True)
        self.assertEqual(verifier.verify("\\boxed{8.95}"), True)
        self.assertEqual(verifier.verify("\\boxed{26.32}"), True)
        self.assertEqual(verifier.verify("\\boxed{78}"), True)
        self.assertEqual(verifier.verify("\\boxed{-78}"), True)
        self.assertEqual(verifier.verify("\\boxed{5}"), True)

    def test_custom_qa_parse_solution_fn(self):
        self.assertFalse(custom_qa_parse_solution_fn(
            '<think>\nssssss\n</think>\n\n<question>\n某公司生产密码设备，其产品批次密码t为两位正整数。密码设置规则要求：将t乘以生产效率系数K后，结果最后两位必须是36。效率系数K通过如下方式计算：去年设备计划运行250天，实际因维护停机30天，另5%时间用于年度升级。K值等于[(计划运行天数 - 停机天数 - 升级天数) × 0.05] + 1。同时，公司年度维护成本为80,000元，但这不影响K的计算。求满足条件的密码t值。  \nAnswer: \\boxed{76}  \nAnswer Type: NumericalAnswer  \n</question><|im_end|><｜end▁of▁sentence｜>'
        ))
        self.assertTrue(custom_qa_parse_solution_fn(
            "<think>\nssssss\n</think>\n\n<question>\nQuestion: A green energy factory optimizes its production parameters via a reaction governed by the equation:  \n\\[ a^2 + 3b^2 + \\frac{c^2 + 3d^2}{2} = a + b + c + d - 1 \\]  \nwhere \\( a \\), \\( b \\), \\( c \\), and \\( d \\) are operational parameters (unitless in the equation). The process involves two steps with yields of 82% and 76%, respectively. Given that the initial reagent InCl₃ has a purity of 85%, and the target is to produce 0.0350 moles of the final product, calculate the operational efficiency index \\( E = 1000a + 100b + 10c + d \\). Ignore distractor information: annual maintenance costs of ¥90,000 and equipment depreciation period of 7 years.  \nAnswer: \\boxed{527}  \nAnswer Type: NumericalAnswer  \n</question><|im_end|>"
        ))
        self.assertTrue(custom_qa_parse_solution_fn(
            "<think>\nOkay, let me try to create a question based on the given requirements. The respondent is an undergraduate with knowledge in calculus, vector operations, central forces, work done by a force, position and velocity vectors. The skills include integrating velocity to find position, calculating dot products, applying the work-energy theorem, and understanding central force properties. The difficulty is advanced, so I need to incorporate multiple steps and potential distractors.\n\nFirst, I need to think of a scenario that combines these elements. Maybe a physics problem involving a particle moving under a central force, where they have to calculate work done or perhaps find a position vect... [省略] ... places  correct.\n\nTherefore, the final question is as constructed above, with the answer being 479.366 kJ. Let me adjust the question\'s answer in the output.\n</think>\n\n<question>\nQuestion: A researcher compresses an ideal gas in a laboratory. Initially, the gas has a pressure of 3 bar, a volume of 2.5 m³, and a temperature of 25°C. After isobaric compression, the volume is reduced to 1.2 m³, and the temperature is measured as 158°F. The molar specific heat at constant pressure (Cp) for the gas is 35.2 J/mol·K. During the process, 500 J of non-volume work is done on the gas. Calculate the change in enthalpy (ΔH) of the gas in kJ, rounded to three decimal places. (Note: The ideal gas constant R = 8.314 J/mol·K.)\nAnswer: 479.366 kJ\nAnswer Type: WithUnitSymbol\n</question><|im_end|>"))

    def test_question_similarity(self):
        batch_solution_str, batch_ground_truth = load_fabricate_aio_data(
            format="wrong_question", num=100)

        scorer = QuestionSimilarity(custom_qa_parse_solution_fn)
        for response, gt in zip(batch_solution_str, batch_ground_truth):
            score = scorer.get_penalty_or_reward(response, gt)
            self.assertTrue(score > 0.5)

    def test_calc_answer_verify(self):
        batch_solution_str, batch_ground_truth = load_fabricate_aio_data(
            format="wrong_question", num=100)

        scorer = CalculationAnswerFormatVerify(custom_qa_parse_solution_fn)
        for response, gt in zip(batch_solution_str, batch_ground_truth):
            score = scorer.get_penalty_or_reward(response, gt)
            self.assertTrue(score >= 0)


#         x, y = [], []
#         task = QwQLongCoTDoc2QueryV2ComputeScore(split="valid")
#         for solution_str, gt in zip(batch_solution_str, batch_ground_truth):
#             # QSim\Format\AnsFeature

#             solution_str = '<think>***</think><question>\nQuestion: 在范畴Set中，设函子G为Option，对于任何集合A，Option(A)是{None} ∪ A。给定符号函子H由H(X) = X·G定义。设函子Z和自然变换e满足：对于任意集合A，Z(A) = A × {0,1}，且e_{Option(A)}将Option(A)中的每个元素映射为：None → (None, 0)，而a ∈ A → (Some a, 1)。考虑集合A = {1, 2}，计算配分律δ_{(Z,e)}在Option(Z(A)) → Z(Option(A))的映射中，元素Some((1,0))的像在Z(Option(A))中的第二个分量的值。注意：Option(Z(A))中的元素Some((1,0))表示Some包裹着Z(A)中的元素(1,0)。\nAnswer: \\boxed{-1}\nAnswer Type: NumericalAnswer\n</question>'
#             score = task.get_penalties()["Format"](solution_str, gt)
#             print(solution_str)
#             print(score)
#             print("="*80)
#             break

#     def test_compute_score(self):
#         batch_solution_str, batch_ground_truth = load_doc2query_v2(32)

#         task = QwQLongCoTDoc2QueryV2ComputeScore(split="valid")
#         print(task.compute_score([None]*len(batch_solution_str),
#                                  batch_solution_str, batch_ground_truth))


# class TestDoc2Query(unittest.TestCase):
#     def test_get_difficulty_reward(self):
#         async def main():
#             batch_solution_str, batch_ground_truth = load_doc2query()
#             task = QwQLongCoTDoc2QueryComputeScore(split="valid")
#             results = await task.get_difficulty_reward(
#                 [None] *
#                 len(batch_solution_str), batch_solution_str, batch_ground_truth
#             )
#             print(results)
#         aio.run(main())

#     def test_chat_completion_with_retry(self):
#         async def main():
#             batch_solution_str, batch_ground_truth = load_doc2query(32*8)
#             task = QwQLongCoTDoc2QueryComputeScore(split="valid")
#             prompts = []

#             instruct = 'Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format "Answer: $LETTER" (without quotes), where LETTER is one of the option letters. You must first think step by step with very detail thinking process.'

#             for _ in batch_solution_str:
#                 result = doc2query_parse_solution_fn(_)
#                 if result is None:
#                     continue
#                 question, options, answer = result
#                 prompts.append(task.format_question(
#                     question, options, None) + f"\n\n{instruct}")

#             # results = await task.chat_completion_with_retry(
#             #     "http://10.130.0.245:5002", prompts
#             # )
#             prompts = prompts * 16
#             s1 = time.time()
#             results = await task.generate_responses(
#                 prompts
#             )
#             # for p, r in zip(prompts, results):
#             #     print(p)
#             #     print(r)
#             #     print("="*80)
#             # print(f'Finish: {time.time()-s1}s')
#         aio.run(main())

#     def test_compute_score(self):
#         # batch_solution_str, batch_ground_truth = load_doc2query(32)

#         task = QwQLongCoTDoc2QueryComputeScore(split="valid")
#         print(qwq_longcot_doc2query_compute_score_valid([None]*len(batch_solution_str),
#                                                         batch_solution_str, batch_ground_truth))


# def doc2query_format_filter(path):
#     outputs = []
#     with open(path, "rt") as f:
#         for line in f:
#             example = json.loads(line)
#             prompt = example["self_improvement"]["chat_completion"]
#             option_num = int(re.findall(
#                 r'- Number of options: (\d+)', prompt)[0].strip())
#             difficulty = re.findall(
#                 r'- Difficulty level: (\w+)', prompt)[0].strip()

#             response = example["self_improvement"]["responses"][0]["response"]["text"]
#             result = doc2query_parse_solution_fn(response)
#             if result is None:
#                 continue
#             question, options, answer = result
#             if len(options) != option_num:
#                 continue
#             try:
#                 ans_index = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
#                              'H', 'I', 'J', 'K', 'L', 'M'].index(answer)
#             except Exception as err:
#                 continue
#             if not (ans_index <= (len(options)-1)):
#                 continue
#             prompt = prompt[prompt.index("<|im_start|>user") +
#                             len("<|im_start|>user"):prompt.index("<|im_end|>\n<|im_start|>assistant")].strip()

#             output = {
#                 "system": "You are a helpful assistant and an expert in creating a question. Now your task is to create a question. You first thinks about the reasoning process in the mind and then provides the user with the answer. Show your reasoning process in <think> </think> tags, and your final question creation in <question> </question> tags. Remember the part in <question> </question> tags MUST only be the question you created ** WITHOUT ** any explanation or solution.",
#                 "instruction": prompt,
#                 "input": "",
#                 "output": response
#             }
#             outputs.append(output)

#     with open("/cpfs01/shared/llm_ddd/tongjian/synthetic/sft/fabricate_qa_v2.json", "rt") as f:
#         mix = json.load(f)
#         print(len(mix))

#     mix.extend(outputs)
#     random.shuffle(mix)

#     with open("/cpfs01/shared/llm_ddd/tongjian/synthetic/sft/fabricate_qa_v3.json", "wt") as f:
#         json.dump(mix, f, ensure_ascii=False)
#     print(len(mix))


# def doc2query_bon_merge(path, output):
#     outputs = []
#     scorer = RuleBasedOptionMatch()
#     group = defaultdict(list)

#     pbar = tqdm(total=736928)

#     with open(path, "rt") as f:
#         for line in f:
#             example = json.loads(line)
#             pbar.update(1)
#             prompt = example["self_improvement"]["chat_completion"]
#             response = example["self_improvement"]["responses"][0]["response"]["text"]
#             result = doc2query_parse_solution_fn(response)
#             if result is None:
#                 continue
#             question, options, answer = result
#             try:
#                 ans_index = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
#                              'H', 'I', 'J', 'K', 'L', 'M'].index(answer)
#             except Exception as err:
#                 continue
#             if not (ans_index <= (len(options)-1)):
#                 continue

#             score = scorer.get_penalty_or_reward(response, {
#                 "question": example["self_improvement"]["question"],
#                 "answer": example["self_improvement"]["answer"],
#                 "options": example["self_improvement"]["options"],
#                 "difficulty": example["self_improvement"]["difficulty"],
#             })
#             group[example["uuid"]].append({
#                 "response": response,
#                 "score": score
#             })

#     done = {}
#     with open(output, 'wt') as g:
#         with open(path, "rt") as f:
#             for line in f:
#                 example = json.loads(line)
#                 if example["uuid"] not in done:
#                     done[example["uuid"]] = True
#                     del example["self_improvement"]["responses"]
#                     example["self_improvement"]["responses"] = group[example["uuid"]]
#                     g.write(f'{json.dumps(example, ensure_ascii=False)}\n')


# def doc2query_difficulty_filter(path, output):
#     pbar = tqdm(total=736928)

#     with open(output, "wt") as g:
#         with open(path, "rt") as f:
#             for line in f:
#                 example = json.loads(line)
#                 pbar.update(1)
#                 responses = example["self_improvement"]["responses"]

#                 threshold = 0.5
#                 filtered = [_ for _ in responses if _["score"] > threshold]
#                 if len(filtered) < 7 and len(filtered):
#                     del example["self_improvement"]["responses"]
#                     del example["self_improvement"]["chat_completion"]
#                     g.write(f'{json.dumps(example, ensure_ascii=False)}\n')


# def doc2query_fabricate_qa_difficulty_reward(path, output):

#     for filename in tqdm(os.listdir(path)):
#         basename = os.path.basename(filename)
#         outputs = []
#         filename = os.path.join(path, filename)
#         with open(filename, "rt") as f:
#             for line in f:
#                 example = json.loads(line)
#                 outputs.append(
#                     (
#                         example, example["self_improvement"]["responses"][0]["response"]["text"],
#                         {
#                             "document": example["content"],
#                         }
#                     )
#                 )
#         output_path = f'{output}_{basename}'
#         if os.path.exists(output_path):
#             continue

#         async def main():
#             task = QwQLongCoTDoc2QueryComputeScore(
#                 split="valid", difficulty_bon=4)
#             with open(output_path, "wt") as f:
#                 for batch in batchify(outputs, n=1024):
#                     examples = [_[0] for _ in batch]
#                     batch_solution_str, batch_ground_truth = [
#                         _[1] for _ in batch], [_[2] for _ in batch]
#                     results, pass_rates = await task.get_difficulty_reward(
#                         [None] *
#                         len(batch_solution_str), batch_solution_str, batch_ground_truth, repeat=4
#                     )
#                     for example, result, pass_rate in zip(examples, results, pass_rates):
#                         example["self_improvement"]["difficulty_reward"] = result
#                         example["self_improvement"]["pass_rate"] = pass_rate
#                         f.write(f'{json.dumps(example, ensure_ascii=False)}\n')
#         aio.run(main())


# def doc2query_postprocess(path, output):
#     task = QwQLongCoTDoc2QueryComputeScore(split="valid")

#     with open(output, "wt") as g:
#         for filename in tqdm(os.listdir(path)):
#             filename = os.path.join(path, filename)
#             with open(filename, "rt") as f:
#                 for line in f:
#                     example = json.loads(line)
#                     response = example["self_improvement"]["responses"][0]["response"]["text"]
#                     result = doc2query_parse_solution_fn(
#                         response, remove_option_letter=False)
#                     if result is None:
#                         continue
#                     question, options, answer = result
#                     try:
#                         ans_index = task.MULTICHOICE_LETTER.index(answer)
#                     except Exception as err:
#                         continue
#                     if len(options) < 40:
#                         continue
#                     if ans_index > len(options) - 1 or not options[ans_index].startswith(f'{answer})'):
#                         continue
#                     result = doc2query_parse_solution_fn(
#                         response, remove_option_letter=True)
#                     question, options, answer = result
#                     example["self_improvement"]["synthetic_qa_prompt"] = example["self_improvement"]["chat_completion"]
#                     del example["self_improvement"]["chat_completion"]
#                     example["self_improvement"]["synthetic_qa_response"] = response
#                     example["self_improvement"]["question"] = question
#                     answer_content = options[ans_index]

#                     random.shuffle(options)

#                     try:
#                         example["self_improvement"]["options"] = options
#                         example["self_improvement"]["answer"] = answer_content
#                         example["self_improvement"]["answer_letter"] = task.MULTICHOICE_LETTER[options.index(
#                             answer_content)]
#                         example["self_improvement"]["prompt"] = 'Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format "Answer: $LETTER" (without quotes), where LETTER is one of the option letters.\n\n' + task.format_question(
#                             question=question, options=options, answer=None
#                         )

#                         g.write(f'{json.dumps(example, ensure_ascii=False)}\n')
#                     except Exception as err:
#                         continue


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
