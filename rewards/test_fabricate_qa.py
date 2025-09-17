import os
import re
import uuid
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
    Agent,
    RewardModelAgent,
    JudgeTwoQuestionSimilarity,
    Doc2QueryV3LooseQuestionEval,
    QuestionRefineHack,
    QuestionDifficultyEval,
    LRUCache,
    WithUnitSymbol,
    NumericalAnswer,
    doc2query_v2_parse_solution_fn,
    doc2query_v3_parse_solution_fn,
    salt_parse_solution_fn,
    Doc2QueryV2FormatVerify,
    LanguageConsistency,
    BadQuestionDetection,
    Doc2QueryV2ComputeScore,
    SALTFormatVerify,
    SALTBadQuestionDetection,
    QuestionSimilarityPenalty,
    SALTComputeScore,
    DOC2QUERY_V2_DEFAULT_PARAMS,
    DOC2QUERY_V2_DEV_PARAMS,
    DOC2QUERY_V3_DEFAULT_PARAMS,
    KG2QUERY_V1_DEFAULT_PARAMS,
    SALT_DEFAULT_PARAMS,
    SALT_DEV_PARAMS,
    RLVR_DEFAULT_PARAMS,
    Doc2QueryV3FormatVerify,
    Doc2QueryV3ComputeScore,
    Doc2QueryV3FanOutComputeScore,
    RLVRVerify,
    RLVRComputeScore,
    fabricate_aio_compute_score_valid,
    doc2query_v3_fanout_parse_solution_fn,
    rlvr_shortcot_compute_score_valid,
    KG2QueryV1ComputeScore,
    LEARNABLE_COT_DEFAULT_PARAMS,
    LearnableCoTComputeScore,
    rlvr_shortcot_parse_solution_fn,
    _multiturn_doc2query_v3_compute_score_train,
    multiturn_parse_question_solution_fn,
    DOC2QUERY_V4_DEFAULT_PARAMS,
    Doc2QueryV4ComputeScore,
    doc2query_v4_parse_solution_fn
)

UNITTEST_AGENT = Agent(**{
    "model": "qwen25_32B_instruct",
    "base_url": "http://10.130.142.223:8000/v1",
    "api_keys": "EMPTY",
    "request_kwargs": {
        "temperature": 0.7,
        "timeout": 360,
        "max_tokens": 16384,
    },
})

_doc2query_v2_qas = [
    ("Consider a triangle with angles \\(\\alpha\\), \\(\\beta\\), and \\(\\gamma\\) such that \\(\\alpha + \\beta + \\gamma = \\pi\\). Given that \\(\\tan \\alpha \\tan \\beta = \\csc \\frac{\\pi}{3}\\), find the value of \\(\\frac{\\cos \\alpha \\cos \\beta}{\\cos \\gamma}\\).", "\\boxed{2\\sqrt{3} + 3}", "NumericalAnswer"),
    ("Compute the hybridization energy for the given parameters θ = π/2, θ₁ = π/4, θʳ = π/4, κ₂ᵐ = 1, and θᵣ - θₗ = θ. Express your answer in terms of E₀^n.",
     "\\boxed{0.500}", "NumericalAnswer"),
    ("An investor receives a series of payments over n years. The payment in the k-th year is given by the formula \\(\\frac{1}{k(k+1)}\\) of the total investment. What is the total amount received after n years?", "\\boxed{0.999}", "NumericalAnswer"),
    (" A point P is outside a circle with radius 5 cm. A tangent from P touches the circle at point T, and a secant from P intersects the circle at points A and B. The lengths of PA and PB are 10 cm and 15 cm, respectively. Find the length of the tangent PT.",
     "\\boxed{12.245}", "NumericalAnswer"),
    ("In a laboratory experiment, 100 grams of starch is converted to sugar through enzyme action, then the sugar is fermented to produce ethanol and carbon dioxide. The ethanol is then distilled to separate it from the mixture. Assuming the fermentation produces 100% yield, what is the concentration of ethanol in the final product after distillation, expressed as a percentage?",
     "\\boxed{56.764}", "NumericalAnswer"),
    ("In a system where n=4 simple harmonic motion vectors are superimposed, each with a phase difference of 45° and an amplitude of 5, determine the magnitude R and the phase angle α of the resultant vector.",
     "\\boxed{13.066}", "NumericalAnswer"),
    ("A triangle is divided by lines parallel to its sides into smaller triangles. The area of one of the smaller triangles is 0.210 square units. Find the area of the original triangle.",
     "\\boxed{1.890}", "NumericalAnswer"),
    ("Let Γλ' be a cell defined by the set of indices Jγ'+ = {1,2} and Jγ'- = {3,4}, with signs + and -, respectively. Find ρ↓λ, the sum of tree diagrams corresponding to Γ↓λ.",
     " \\boxed{0.500}", "NumericalAnswer"),
    ("Consider a triangle with angles \\(\\alpha\\), \\(\\beta\\), and \\(\\gamma\\) such that \\(\\alpha + \\beta + \\gamma = \\pi\\). Given that \\(\\tan \\alpha \\tan \\beta = \\csc \\frac{\\pi}{3}\\), find the value of \\(\\frac{\\cos \\alpha \\cos \\beta}{\\cos \\gamma}\\).", "\\boxed{2\\sqrt{3} + 3}", "NumericalAnswer"),
    ("Question: What is the magnitude of the displacement vector after 0.4 seconds if the ball's motion is observed in 16 strobe photographs, each 0.025 seconds apart, and the velocities are as follows: v1 = (1,2), v2 = (2,3), ..., v16 = (16,17)?",
     "\\boxed{5.100}", "NumericalAnswer"),
    ("In a game with two players, each player's type is either A or B. The player with type A can choose action 1 or 2, and the player with type B can choose action 1 or 3. The payoff for player 1 is 10 if they choose action 1 and the other player chooses action 1, and 0 otherwise. The payoff for player 2 is 20 if they choose action 1 and the other player chooses action 1, and 0 otherwise. How should the mechanism designer structure the mechanism to ensure that the players choose their types in a way that maximizes the total payoff?",
     "\\boxed{30.000}", "NumericalAnswer"),
    ("A solid sphere with a radius of 0.2 meters and a mass of 5 kg is released from rest at the top of an incline with a height of 2 meters. Assuming the sphere rolls without slipping, what is the final velocity of the sphere at the bottom of the incline? (Take g = 9.8 m/s²)",
     " \\boxed{5.291}", "NumericalAnswer"),
    ("What is the mean ionic activity coefficient of a 0.1 M sodium chloride (NaCl) solution in water? Assume the Debye-Hückel constant A = 0.5. Express your answer to three decimal places.\\",
     "\\boxed{0.700}", "NumericalAnswer"),
    ("What is the main lobe width (in Hz) of the power spectral density (PSD) of an AMI-coded RZ-modulated signal with a bit rate of 100 kbps?",
     "\\boxed{100000.000}", "NumericalAnswer"),
    ("A composite Poisson process has a Poisson parameter λ = 5, and each event is associated with a uniform random variable from 0 to 10. What is the variance of the composite process?\\",
     " \\boxed{41.667}", "NumericalAnswer"),
    ("A pipe with a length of 100 meters is carrying water at a flow rate of 0.5 m³/s. The head loss due to friction in the pipe is 20 meters. The roughness of the pipe is 0.001 meters. The fluid's kinematic viscosity is 1.0×10^-6 m²/s. What is the diameter of the pipe in meters?",
     "\\boxed{0.500}", "NumericalAnswer"),
    ("A cube of side length 1 meter is submerged in a liquid whose density varies linearly from 1000 kg/m³ at the surface to 1100 kg/m³ at the bottom. What is the buoyant force acting on the cube?",
     "\\boxed{10295.500}", "NumericalAnswer")
]

_salt_qas = [
    ("In synthesizing a fragrance compound, a limonene derivative undergoes catalytic hydrogenation, epoxidation, base-induced ring-opening, and esterification with propanoic acid. If the final propionate has an (S) configuration at the carbon where the initial hydrogenation site's methyl is adjacent, what must be the configuration of that methyl group post-hydrogenation?", "R"),
    ("During genetic counseling, a 28-year-old woman reveals her brother had an X-linked recessive muscular condition, while her recent blood tests show normal levels of a muscle enzyme often elevated in affected individuals. What is the most accurate assessment of her carrier status without genetic analysis?", "Carrier status is uncertain")
]

_doc2query_v3_qas = [
    ("The reaction product of the o-toluidine method for measuring blood glucose is ( ).",
     ["Schiff base", "Molybdenum blue", "Tungstophosphoric acid", "Glucuronide", "Quinone compounds"], "A"),
    ("钢材强度设计值，下列哪一种说法是正确的?", ["同一种牌号不同质量等级的钢材，强度设计值相同", "同一种牌号不同厚度的钢材，强度设计值相同",
                              "同一种牌号的冷弯型钢钢材和普通钢材，强度设计值相同", "同一种牌号的钢材，《门式刚架轻型房屋钢结构技术规程》(CECS 102：2002)和《钢结构设计规范》(GB 50017-2003)采用不同的强度设计值"], "A"),
    ("为了确保双卷扬系统的稳定性，以下哪个措施最为关键？", [
     "增加液压系统的泄漏", "减小负载重量", "提高液压油的刚性模量", "调整PI控制器的积分系数"], "A"),
    ("Which laboratory's neutral beam system has successfully generated a 75 A positive ion beam?", [
     "Another laboratory", "LBL/LLL", "ORNL", "BNL"], "C"),
    ("EN AW 2014铝合金在高温变形过程中，相对软化现象的主要原因是什么？", [
     "粒子粗化", "温度升高", "动态回复和再结晶", "应变速率降低"], "C"),
    ("自组织系统的一个特征是什么？", ["负反馈饱和", "多稳定性", "正反馈放大", "随机游走放大"], "B"),
    ("When both the magnetic parameter \\( M \\) and the second order slip parameter \\( \\lambda_2 \\) are increased, what is the effect on the Bejan number for the nanofluid flow?", [
     "Decreases", "Increases", "Remains unchanged", "First increases then decreases"], "A")
]


def load_dataset(task_name, num=100, xml_cot=False):
    filename = "/cpfs01/shared/llm_ddd/tongjian/rl/fabricate_aio/fabricate_aio_train_0730.parquet"
    if task_name == "kg2query_v1":
        filename = "/cpfs01/shared/llm_ddd/tongjian/rl/fabricate_aio/kg2query_v1_oc_v1_7_hard_problem_0623.parquet"
    elif task_name == "doc2query_v3":
        filename = "/cpfs01/shared/llm_ddd/tongjian/rl/doc2query_v3/doc2query_v3_rl_inputs_diamond.parquet"
    elif task_name == "doc2query_v4":
        filename = "/cpfs01/shared/llm_ddd/tongjian/rl/doc2query_v4/pretrain_general_doc_8k_rl_inputs_train/index0.parquet"
    elif task_name == "learnable_cot":
        filename = "/cpfs01/shared/llm_ddd/tongjian/rl/learnable_cot/dapo_math_17k_train.parquet"
    elif task_name == "rlvr":
        filename = "/cpfs01/shared/llm_ddd/tongjian/rl/learnable_cot/aime_2024_2025.parquet"

    batch_solution_str, batch_ground_truth = [], []
    batch_data_sources = []

    df = pd.read_parquet(filename)
    count = 0
    for _, row in df.iterrows():
        row = row.to_dict()
        if task_name is not None:
            if row["data_source"] != task_name:
                continue
        batch_data_sources.append(row["data_source"])
        if row["data_source"] == "doc2query_v2":
            sample = random.choice(_doc2query_v2_qas)
            batch_solution_str.append(
                f'<think>\nUNITTEST_ONLY\n</think>\n\n<question>\nQuestion: {sample[0]}\n\nAnswer: {sample[1]}\n\nAnswer Type: {sample[2]}\n</question>'
            )
        elif row["data_source"] == "salt" or row["data_source"] == "kg2query_v1":
            sample = random.choice(_salt_qas)
            batch_solution_str.append(
                f'<think>\nUNITTEST_ONLY\n</think>\n\n<question>\nQuestion: [SYNTHETIC] {sample[0]}\n\nAnswer: [SYNTHETIC] {sample[1]}\n</question>'
            )
        elif row["data_source"] == "doc2query_v3":
            sample = random.choice(_doc2query_v3_qas)
            o = "\n".join([f'{c}) {_o}' for c, _o in zip(
                ["A", "B", "C", "D"], sample[1])])
            batch_solution_str.append(
                f'<think>\nUNITTEST_ONLY\n</think>\n\n<question>\nQuestion: [SYNTHETIC] {sample[0]}\n\nOptions: {o}\n\nAnswer: {sample[2]}\n</question>'
            )
        elif row["data_source"] == "doc2query_v4":
            batch_solution_str.append(
                f'<think>\nUNITTEST_ONLY\n</think>\n\n<question>\nQuestion:\n写篇网页风格的文章，内容无限制，内容需要包含写小错误；因为网页内容质量不会很高。\n\nReference:\n{row["reward_model"]["document"]}\n</question>'
            )
        elif row["data_source"] == "rlvr" or row["data_source"] in ("aime_2024", "aime_2025"):
            batch_solution_str.append(
                f'<think>\nUNITTEST_ONLY\n</think>\n\n答案：{row["reward_model"]["ground_truth"]}'
            )
        elif row["data_source"] == "learnable_cot":
            batch_solution_str.append(
                f'<think>\nUNITTEST_ONLY\n</think>\n\n{row["reward_model"]["answer"]}'
            )
        batch_ground_truth.append(row["reward_model"])
        if len(batch_ground_truth) == num:
            break
    if task_name is not None:
        return batch_solution_str, batch_ground_truth
    else:
        return batch_data_sources, batch_solution_str, batch_ground_truth


class TestUtils(unittest.TestCase):
    def test_batch_call_open_api(self):
        # task = JudgeTwoQuestionSimilarity()
        # task = Doc2QueryV3LooseQuestionEval()
        # task = QuestionRefineHack()
        task = QuestionDifficultyEval()

        async def main():
            results = await task.do_job(
                agent=UNITTEST_AGENT,
                batch_inputs=[(
                    "During a drama club meeting, a member erroneously attributed the Salem witch trials play \"The Crucible,\" focusing on family conflict in 1692 Massachusetts, to Eugene O'Neill. Name the correct playwright who explored societal pressures through this drama",
                    "Who is considered the most important American playwright of the 20th century?\nA) Shepard\nB) Albee\nC) Williams\nD) Wilder\nE) Mamet\nF) O'Neal\nG) Miller\nH) Wilson\nI) Hellman\nJ) Simon")],
                # batch_inputs=[
                #     "在评估抗PD-1治疗反应的分子标志物中，哪一项与临床试验中的客观缓解率（ORR）和无进展生存期（PFS）显著相关？"],
                max_concurrent_requests=64,
            )
            print(results)
        aio.run(main())


class TestSALT(unittest.TestCase):
    def test_salt_format_verify(self):
        scorer = SALTFormatVerify(
            salt_parse_solution_fn, -1.75, -1.25)
        print(scorer.get_penalty_or_reward(
            '<think>\nssssss\n</think>\n\n<question>\nQuestion: 某公司生产密码设备，其产品批次密码t为两位正整数。密码设置规则要求：将t乘以生产效率系数K后，结果最后两位必须是36。效率系数K通过如下方式计算：去年设备计划运行250天，实际因维护停机30天，另5%时间用于年度升级。K值等于[(计划运行天数 - 停机天数 - 升级天数) × 0.05] + 1。同时，公司年度维护成本为80,000元，但这不影响K的计算。求满足条件的密码t值。  \nAnswer: \\boxed{76}xxxxx xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx xx\n</question><|im_end|><｜end▁of▁sentence｜>', None
        ))

    def test_detect_bad_question(self):
        scorer = SALTBadQuestionDetection(
            salt_parse_solution_fn,  -1.75, -1.25)

        print(scorer.get_penalty_or_reward(
            '<think>\nssssss\n</think>\n\n<question>\nQuestion: During a materials analysis audit, an engineer incorrectly noted white heart malleable iron\'s microstructure matrix as containing graphite. Properly, the matrix consists of two iron phases plus cementite. Identify these two phases separated by \'+\' symbols. \nAnswer: ferrite+cementite\n</question><|im_end|><｜end▁of▁sentence｜>',
            {"question": "During a materials analysis audit, an engineer incorrectly noted white heart malleable iron's microstructure matrix as containing graphite. Properly, the matrix consists of two iron phases plus cementite. Identify these two phases separated by '+' symbols.", "lang_code": "en"}
        ))

    def test_question_similarity_penalty(self):
        scorer = QuestionSimilarityPenalty(
            salt_parse_solution_fn, 0, 0.1)

        print(scorer.get_penalty_or_reward(
            '<think>\nssssss\n</think>\n\n<question>\nQuestion: During a materials analysis audit, an engineer incorrectly noted white heart malleable iron\'s microstructure matrix as containing graphite. Properly, the matrix consists of two iron phases plus cementite. Identify these two phases separated by \'+\' symbols. \nAnswer: ferrite+cementite\n</question><|im_end|><｜end▁of▁sentence｜>',
            {"question": "In an industrial setting, a quality control technician inspects a sample labeled as malleable iron with a ferrite-pearlite matrix and hard nodules, mistakenly reported as graphite. Identify the three primary microstructural components present.", "lang_code": "en"}
        ))

    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="salt", num=4)
        task = SALTComputeScore(
            salt_parse_solution_fn, split="valid", args=SALT_DEV_PARAMS)
        print(task.compute_score(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth,
        ))

    def test_learnable_reward(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="salt", num=4)

        task = SALTComputeScore(
            salt_parse_solution_fn, split="valid", args=SALT_DEFAULT_PARAMS)
        for _ in task._penalties:
            print(_, _.min_score, _.max_score)

        async def main():
            results = await task.get_learnable_reward(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
            print(results)
        aio.run(main())

    def test_hack_detect(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="salt", num=4)

        task = SALTComputeScore(
            salt_parse_solution_fn, split="valid", args=SALT_DEFAULT_PARAMS)

        async def main():
            # get_similarity_penalty get_hack_penalty
            results = await task.get_similarity_penalty(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
            print(results)
        aio.run(main())


class TestFabricateAIO(unittest.TestCase):
    def test_compute_score(self):
        batch_data_sources, batch_solution_str, batch_ground_truth = load_dataset(
            task_name=None, num=100,)
        fabricate_aio_compute_score_valid(
            batch_data_sources, batch_solution_str, batch_ground_truth
        )


class TestRLVR(unittest.TestCase):
    def test_rlvr_verify(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="rlvr", num=6)
        task = RLVRVerify()

        async def main():
            results = await task.do_job(
                agent=UNITTEST_AGENT,
                batch_inputs=[(x, y, z) for x, y, z in zip(batch_solution_str, [
                    None]*len(batch_solution_str), batch_ground_truth)],
                max_concurrent_requests=64,
            )
            print(results)
        aio.run(main())

    def test_get_accuracy(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="rlvr", num=6)

        rlvr_shortcot_compute_score_valid(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth,
        )


class TestDoc2QueryV3(unittest.TestCase):
    def test_reward_model_agent(self):
        task = Doc2QueryV3ComputeScore(
            doc2query_v3_parse_solution_fn, split="valid", args=DOC2QUERY_V3_DEFAULT_PARAMS)
        agent = task.rm_agent
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v3", num=6)

        async def main():
            results = await agent.compute_rm_score(
                [None]*len(batch_solution_str), batch_solution_str, batch_ground_truth)
            for i, score in enumerate(results):
                print(score)
        aio.run(main())

    def test_doc2query_v3_format_verify(self):
        scorer = Doc2QueryV3FormatVerify(
            doc2query_v2_parse_solution_fn, -1.25, -0.75)
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v3", num=4)
        for x, y in zip(batch_solution_str, batch_ground_truth):
            print(scorer.get_penalty_or_reward(x, y))

    def test_doc2query_v3_get_difficulty_reward(self):
        task = Doc2QueryV3ComputeScore(
            doc2query_v3_parse_solution_fn, split="valid", args=DOC2QUERY_V3_DEFAULT_PARAMS)

        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v3", num=6)

        async def main():
            # simulate_respondent get_difficulty_reward
            results = await task.get_difficulty_reward(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
            print(results)
        aio.run(main())

    def test_quick_question_eval(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v3", num=6)
        task = Doc2QueryV3ComputeScore(
            doc2query_v3_parse_solution_fn, split="valid", args=DOC2QUERY_V3_DEFAULT_PARAMS)

        async def main():
            results = await task.quick_question_eval(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth
            )
            print(results)
        aio.run(main())

    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v3", num=6)
        task = Doc2QueryV3ComputeScore(
            doc2query_v3_parse_solution_fn, split="train", args=DOC2QUERY_V3_DEFAULT_PARAMS)
        print(task.compute_score(
            [None]*len(batch_solution_str), batch_solution_str, batch_ground_truth,
        ))
        # print(task.suspect_question_can_not_be_determined(
        #     [['D'], ['D'], ['D'], ['D'], ['D'], ['D'], ['D'], ['D']],
        #     [
        #         [
        #             "B"
        #         ],
        #         [
        #             "D"
        #         ],
        #         [
        #             "G"
        #         ],
        #         [
        #             "F"
        #         ],
        #         [
        #             "G"
        #         ]
        #     ], ["F", "G"], "D"
        # ))


class TestDoc2QueryV3FanOut(unittest.TestCase):
    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v3", num=40)
        task = Doc2QueryV3FanOutComputeScore(
            doc2query_v3_fanout_parse_solution_fn,
            doc2query_v3_parse_solution_fn, split="train", args=DOC2QUERY_V3_FANOUT_DEFAULT_PARAMS)
        print(task.compute_score(
            [None]*len(batch_solution_str), batch_solution_str, batch_ground_truth,
        ))


class TestDoc2QueryV2(unittest.TestCase):
    def test_lru_cache(self):
        cache = LRUCache(capacity=3)

        cache['a'] = 1
        cache['b'] = 2
        cache['c'] = 3
        print(cache)  # 输出: {'a': 1, 'b': 2, 'c': 3}
        print(cache._access_order)

        # 访问'a'使其变为最近使用
        print(cache['a'])  # 输出: 1
        print(cache._access_order)

        # 添加'd'会淘汰最久未使用的'b'
        cache['d'] = 4
        print(cache)  # 输出: {'a': 1, 'c': 3, 'd': 4}
        print(cache._access_order)

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
        self.assertEqual(verifier.verify("2.91 kJ/(mol·K)"), True)

        verifier = NumericalAnswer()
        self.assertEqual(verifier.verify("\\boxed{0.00000225}"), True)
        self.assertEqual(verifier.verify("\\boxed{0.000186}"), True)
        self.assertEqual(verifier.verify("\\boxed{426}"), True)
        self.assertEqual(verifier.verify("\\boxed{-854}"), True)
        self.assertEqual(verifier.verify("\\boxed{0.00474}"), True)
        self.assertEqual(verifier.verify("\\boxed{0.000180}"), True)
        self.assertEqual(verifier.verify("\\boxed{0.0213}"), True)
        self.assertEqual(verifier.verify("\\boxed{46.8}"), True)
        self.assertEqual(verifier.verify("\\boxed{1.8}"), True)
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
        self.assertEqual(verifier.verify("\\boxed{-5}"), True)
        self.assertEqual(verifier.verify("\\boxed{0.500}"), True)
        self.assertEqual(verifier.verify("\\boxed{0.210}"), True)
        self.assertEqual(verifier.verify("\\boxed{5.5}"), True)
        self.assertEqual(verifier.verify("\\boxed{-0.0500}"), True)
        self.assertEqual(verifier.verify("\\boxed{8.4}"), True)

    def test_doc2query_v2_parse_solution_fn(self):
        self.assertFalse(doc2query_v2_parse_solution_fn(
            '<think>\nssssss\n</think>\n\n<question>\n某公司生产密码设备，其产品批次密码t为两位正整数。密码设置规则要求：将t乘以生产效率系数K后，结果最后两位必须是36。效率系数K通过如下方式计算：去年设备计划运行250天，实际因维护停机30天，另5%时间用于年度升级。K值等于[(计划运行天数 - 停机天数 - 升级天数) × 0.05] + 1。同时，公司年度维护成本为80,000元，但这不影响K的计算。求满足条件的密码t值。  \nAnswer: \\boxed{76}  \nAnswer Type: NumericalAnswer  \n</question><|im_end|><｜end▁of▁sentence｜>'
        ))
        self.assertTrue(doc2query_v2_parse_solution_fn(
            "<think>\nssssss\n</think>\n\n<question>\nQuestion: A green energy factory optimizes its production parameters via a reaction governed by the equation:  \n\\[ a^2 + 3b^2 + \\frac{c^2 + 3d^2}{2} = a + b + c + d - 1 \\]  \nwhere \\( a \\), \\( b \\), \\( c \\), and \\( d \\) are operational parameters (unitless in the equation). The process involves two steps with yields of 82% and 76%, respectively. Given that the initial reagent InCl₃ has a purity of 85%, and the target is to produce 0.0350 moles of the final product, calculate the operational efficiency index \\( E = 1000a + 100b + 10c + d \\). Ignore distractor information: annual maintenance costs of ¥90,000 and equipment depreciation period of 7 years.  \nAnswer: \\boxed{527}  \nAnswer Type: NumericalAnswer  \n</question><|im_end|>"
        ))
        self.assertTrue(doc2query_v2_parse_solution_fn(
            "<think>\nOkay, let me try to create a question based on the given requirements. The respondent is an undergraduate with knowledge in calculus, vector operations, central forces, work done by a force, position and velocity vectors. The skills include integrating velocity to find position, calculating dot products, applying the work-energy theorem, and understanding central force properties. The difficulty is advanced, so I need to incorporate multiple steps and potential distractors.\n\nFirst, I need to think of a scenario that combines these elements. Maybe a physics problem involving a particle moving under a central force, where they have to calculate work done or perhaps find a position vect... [省略] ... places  correct.\n\nTherefore, the final question is as constructed above, with the answer being 479.366 kJ. Let me adjust the question\'s answer in the output.\n</think>\n\n<question>\nQuestion: A researcher compresses an ideal gas in a laboratory. Initially, the gas has a pressure of 3 bar, a volume of 2.5 m³, and a temperature of 25°C. After isobaric compression, the volume is reduced to 1.2 m³, and the temperature is measured as 158°F. The molar specific heat at constant pressure (Cp) for the gas is 35.2 J/mol·K. During the process, 500 J of non-volume work is done on the gas. Calculate the change in enthalpy (ΔH) of the gas in kJ, rounded to three decimal places. (Note: The ideal gas constant R = 8.314 J/mol·K.)\nAnswer: 479.366 kJ\nAnswer Type: WithUnitSymbol\n</question><|im_end|>"))

    def test_format_verify(self):
        scorer = Doc2QueryV2FormatVerify(
            doc2query_v2_parse_solution_fn, -1.75, -1.25)
        print(scorer.get_penalty_or_reward(
            '<think>\nssssss\n</think>\n\n<question>\nQuestion: 某公司生产密码设备，其产品批次密码t为两位正整数。密码设置规则要求：将t乘以生产效率系数K后，结果最后两位必须是36。效率系数K通过如下方式计算：去年设备计划运行250天，实际因维护停机30天，另5%时间用于年度升级。K值等于[(计划运行天数 - 停机天数 - 升级天数) × 0.05] + 1。同时，公司年度维护成本为80,000元，但这不影响K的计算。求满足条件的密码t值。  \nAnswer: \\boxed{76}  \nAnswer Type: ChemicalName_Answer  \n</question><|im_end|><｜end▁of▁sentence｜>', None
        ))

    def test_language_consistency(self):
        scorer = LanguageConsistency(
            doc2query_v2_parse_solution_fn, -1.25, -0.75)
        print(scorer.get_penalty_or_reward(
            '<think>\nssssss\n</think>\n\n<question>\nQuestion: 某公司生产密码设备，其产品批次密码t为两位正整数。密码设置规则要求：将t乘以生产效率系数K后，结果最后两位必须是36。效率系数K通过如下方式计算：去年设备计划运行250天，实际因维护停机30天，另5%时间用于年度升级。K值等于[(计划运行天数 - 停机天数 - 升级天数) × 0.05] + 1。同时，公司年度维护成本为80,000元，但这不影响K的计算。求满足条件的密码t值。  \nAnswer: \\boxed{76}  \nAnswer Type: ChemicalName_Answer  \n</question><|im_end|><｜end▁of▁sentence｜>',
            {
                "lang_code": "zh"
            }
        ))

    def test_detect_bad_question(self):
        scorer = BadQuestionDetection(
            doc2query_v2_parse_solution_fn, -0.5, -0.0)
        print(scorer.get_penalty_or_reward(
            '<think>\nssssss\n</think>\n\n<question>\nQuestion: 某公司生产密码设备，其产品批次密码t为两位正整数。密码设置规则要求：将t乘以生产效率系数K后，结果最后两位必须是36。效率系数K通过如下方式计算：去年设备计划运行250天，实际因维护停机30天，另5%时间用于年度升级。K值等于[(计划运行天数 - 停机天数 - 升级天数) × 0.05] + 1。同时，公司年度维护成本为80,000元，但这不影响K的计算。求满足条件的密码t值。  \nAnswer: \\boxed{76}  \nAnswer Type: ChemicalName_Answer  \n</question><|im_end|><｜end▁of▁sentence｜>',
            {
                "lang_code": "zh"
            }
        ))

    def test_doc2query_v2_get_difficulty_reward(self):
        task = Doc2QueryV2ComputeScore(
            doc2query_v2_parse_solution_fn, split="valid", args=DOC2QUERY_V2_DEV_PARAMS)

        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v2", num=4)

        async def main():
            # simulate_respondent get_difficulty_reward
            results = await task.simulate_respondent(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
            print(results)
        aio.run(main())

    def test_doc2query_v2_quick_question_eval(self):
        task = Doc2QueryV2ComputeScore(
            doc2query_v2_parse_solution_fn, split="valid", args=DOC2QUERY_V2_DEFAULT_PARAMS)

        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v2", num=4)

        async def main():
            results = await task.quick_question_eval(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
            print(results)
        aio.run(main())

    def test_doc2query_v2_llm_judge_difficulty(self):
        task = Doc2QueryV2ComputeScore(
            doc2query_v2_parse_solution_fn, split="valid", args=DOC2QUERY_V2_DEFAULT_PARAMS)

        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v2", num=4)

        async def main():
            results = await task.llm_judge_difficulty(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
            print(results)
        aio.run(main())

    def test_compute_score(self):
        # DOC2QUERY_V2_DEV_PARAMS
        task = Doc2QueryV2ComputeScore(
            doc2query_v2_parse_solution_fn, split="valid", args=DOC2QUERY_V2_DEFAULT_PARAMS)

        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v2", num=4)

        async def main():
            results = await task._compute_score(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
            print(results)
        aio.run(main())


class TestLearnableCoT(unittest.TestCase):
    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="learnable_cot", num=4)
        task = LearnableCoTComputeScore(
            rlvr_shortcot_parse_solution_fn, split="train", args=LEARNABLE_COT_DEFAULT_PARAMS)

        async def main():
            results = await task._compute_score(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
            print(results)
        aio.run(main())


class TestKG2QueryV1(unittest.TestCase):
    def test_salt_format_verify(self):
        scorer = SALTFormatVerify(
            salt_parse_solution_fn, -1.75, -1.25)
        print(scorer.get_penalty_or_reward(
            '<think>\nssssss\n</think>\n\n<question>\nQuestion: 某公司生产密码设备，其产品批次密码t为两位正整数。密码设置规则要求：将t乘以生产效率系数K后，结果最后两位必须是36。效率系数K通过如下方式计算：去年设备计划运行250天，实际因维护停机30天，另5%时间用于年度升级。K值等于[(计划运行天数 - 停机天数 - 升级天数) × 0.05] + 1。同时，公司年度维护成本为80,000元，但这不影响K的计算。求满足条件的密码t值。  \nAnswer: \\boxed{76}xxxxx xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx xx\n</question><|im_end|><｜end▁of▁sentence｜>', None
        ))

    def test_question_similarity_reward(self):
        task = KG2QueryV1ComputeScore(
            salt_parse_solution_fn, split="valid", args=KG2QUERY_V1_DEFAULT_PARAMS)

        print(task._penalties[-1].get_penalty_or_reward(
            '<think>\nssssss\n</think>\n\n<question>\nQuestion: During a materials analysis audit, an engineer incorrectly noted white heart malleable iron\'s microstructure matrix as containing graphite. Properly, the matrix consists of two iron phases plus cementite. Identify these two phases separated by \'+\' symbols. \nAnswer: ferrite+cementite\n</question><|im_end|><｜end▁of▁sentence｜>',
            {"question": "In an industrial setting, a quality control technician inspects a sample labeled as malleable iron with a ferrite-pearlite matrix and hard nodules, mistakenly reported as graphite. Identify the three primary microstructural components present.", "lang_code": "en"}
        ))

    def test_similarity_reward(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="kg2query_v1", num=4)
        task = KG2QueryV1ComputeScore(
            salt_parse_solution_fn, split="valid", args=KG2QUERY_V1_DEFAULT_PARAMS)

        async def main():
            results = await task.get_similarity_reward(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
            print(results)
        aio.run(main())

    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="kg2query_v1", num=4)
        task = KG2QueryV1ComputeScore(
            salt_parse_solution_fn, split="valid", args=KG2QUERY_V1_DEFAULT_PARAMS)
        print(task.compute_score(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth,
        ))


class TestDoc2QueryV3Multiturn(unittest.TestCase):
    def test_multiturn_parse_question_solution_fn(self):
        test_file = "/cpfs01/shared/llm_ddd/tongjian/sft/self_improvement/doc2query_v3_pdf_multi_turn_rft_v2_filtered.done.jsonl"

        with open(test_file, "rt") as f:
            for line in f:
                example = json.loads(line)
                multiturn = example["self_improvement"]["prompt_generation_multiturn_process"]
                multiturn_str = "\n\n".join(
                    [f'{_["role"]}\n{_["content"]}\n{_["role"].replace("<", "</")}' for _ in multiturn])
                if multiturn_parse_question_solution_fn(multiturn_str) is None:
                    print(1)
                else:
                    print(2)

    def test_compute_score(self):
        task = _multiturn_doc2query_v3_compute_score_train
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v3", num=1)
        batch_solution_str[
            0] = '<think>\n好的，我需要根据用户提供的文档内容来设计一个高难度的单项选择题。首先，我需要仔细阅读并理解文档中的信息。文档列出了多个组织的地址、地区和层级等信息，但看起来主要涉及的是组织的地址和层级分类。例如，很多条目都提到了“Tier 2 (A rating)”和“Tier 2 General”，这可能是指某个评分系统中的层级。\n\n首先，根据用户提供的指导方针，我需要选择一个核心知识点，这个知识点需要有明确的学科知识，并且不能太依赖材料中的具体数据，而是要抽象化。文档中的“Tier”和“rating”可能与某种评分系统有关，比如建筑等级或者某种服务等级。考虑到用户提到的高难度题目需要综合应用知识，我需要设计一个需要多步骤推理的问题。\n\n接下来，我需要确保题目中的信息是自包含的，不依赖原文的直接数据。可能的思路是围绕组织地址、层级分类和业务类型来设计。例如，假设这些组织被分类为不同的层级，每个层级对应不同的业务类型，然后设计一个需要结合多个因素来判断层级的问题。\n\n用户提到要避免直接引用材料中的具体数据，所以不能直接用文档中的组织名称。我需要构造一个虚构但合理的情境。例如，可以设计一个关于医院或医疗机构的组织层级划分问题，其中不同的层级对应不同的服务类型或成本效率，需要结合地址、区域和业务类型来判断最终的层级。\n\n例如，题目可以设定一个医院系统，不同层级的医院根据地址、区域和业务类型来确定其层级，然后给出几个组织的参数，要求选择正确的层级。但需要确保问题中的信息完整，比如地址、区域、业务类型，而不仅仅是原文中的具体条目。\n\n另外，根据用户提供的出题步骤，需要多阶段思维，比如第...</think> [省略] ...er 2 (A) 的条件，因此没有哪个组织符合 Tier 2 (A) 的条件。\n\n答案：没有组织被归为 Tier 2 (A)。<draft></draft>\n<self-test></self-test>\n\n<reflection>\n### 分析过程：\n1. **理解题意和标准答案**：题目要求根据地址区域、业务类型和成立年份来判断哪个组织被归为 Tier 2 (A)。标准答案为选项 B) 地区：郡，业务类型：医疗护理，成立年份：2022。\n2. **提取用户回答的最终结论**：用户回答经过详细的分析后，最终结论是“没有组织被归为 Tier 2 (A)”，因为所有选项都不符合 Tier 2 (A) 的条件。\n3. **对比结果**：用户回答的最终结论与标准答案不一致。用户回答得出结论是没有组织被归为 Tier 2 (A)，而标准答案明确指出选项 B 被归为 Tier 2 (A)。因此，用户回答与标准答案不一致，判定为错误。\n\n```json\n{\n "判断结果": "错误"\n}\n</reflection>\n\n<question>\nQuestion: 某组织的地址区域、业务类型和成立年份决定其层级分类。层级标准如下：\n\n- 地区地址为城市，业务类型为医疗护理或医疗保健，成立年份在2021年后： Tier 2 (A)\n\n- 地区地址为郡，业务类型为医疗护理或医疗保健： Tier 2 General\n\n- 其他情况： Tier 1\n\n现有四个组织的例子，判断哪个组织被归为 Tier 2 (A)。\n\nOptions:\nA) 地区：郡，业务类型：医疗保健，成立年份：2021\n\nB) 地区：郡，业务类型：医疗护理，成立年份：2022\n\nC) 地区：城市，业务类型：医疗保健，成立年份：2020\n\nD) 地区：城市，业务类型：医疗护理，成立年份：2020\n\nAnswer: B\n</question>'

        # batch_solution_str[
        #     0] = '<think>\n好的，我现在需要根据提供的小说《A Shilling for the Cat》来设计一个高难度的单项选择题。首先，我需要仔细阅读并理解小说内容，找出其中的关键知识点和主题。小说主要讲述了作者 experience（即“我”）在ábados小学的经历，以及他在学校生活中的经历，包括与大家的互动、与争议人物的冲突、以及他在学校外的观察等。\n\n首先，我需要选择一个核心考点。根据用户提供的指导，题目需要与材料中的内容紧密相关，但不能直接引用原文。同时，题目要符合学科知识，但这里可能不需要严格的学科，因为小说是文学作品，但可以通过分析其中的主题或人物关系来设计题目。\n\n接下来，我需要考虑小说中的核心冲突和人物关系。例如，小说中，experience与绿色headlight（船夫的儿子）的冲突，以及他与Larkins（新 Draughtsale的主妇）的浪漫关系。这些可以作为考点。但题目不能直接引用这些人物，所以需要将这些元素转化为一个可以考察考生分析人物动机和关系的题目。\n\n根据出题案例中的技巧，题目需要隐藏信息，让考生通过分析原文推断。例如，可以设计一个关于人物动机的题目，让考生通过情境推断人物的动机，而不是直接问问题。\n\n再考虑题目难度提升，需要加入干扰项和多步骤推理。例如，设计一个情境，让考生分析在特定情境下的最佳应对策略，涉及人物的动机、背景和相关细节。\n\n另外，题目需要避免直接引用原文中的特定事件，而是通过相似的情境来呈现。例如，可以设置一个学校同学之间的冲突，让考生分析原因和最佳解决方法，但需要隐藏在情境描述中。\n\n可能的题目方向：在学校同学之间出现的冲突，需... [省略] ...在同学之间建立良好的沟通和理解。\n\nD) 试图找到同学的错误：这可能会导致进一步的冲突，增加学生之间的紧张气氛，因为这要求学生在同学之间进行争执，而不是寻求合作和理解。\n\n综上所述，最佳应对方式是C) 与同学沟通寻求解决问题的方法。通过与同学沟通，学生可以得到直接的建议和解决方案，有助于解决问题并找到一个既符合个人兴趣又不引起同学不满的方法。这不仅有助于解决当前的冲突，还能够在同学之间建立更和谐的关系。\n</self-test>\n\n<correctness-check>\n### 分析过程：\n1. **理解题意和标准答案**：\n - 题目要求选择最佳应对方式来处理学生因个人兴趣缺乏导致与同学产生冲突，以及在课间独自练习个人兴趣活动违反学校规定而感到困惑的问题。\n - 标准答案是C) 与同学沟通寻求解决问题的方法。\n\n2. **提取用户回答的最终结论**：\n - 用户回答详细分析了每个选项，并最终得出结论，认为最佳应对方式是C) 与同学沟通寻求解决问题的方法，因为这种方法有助于解决问题，建立和谐关系。\n\n3. **对比结果**：\n - 用户回答的最终结论C) 与同学沟通寻求解决问题的方法与标准答案C) 与同学沟通寻求解决问题的方法一致。\n\n```json\n{\n "判断结果": "正确"\n}\n```\n</correctness-check>\n\n<question>\nQuestion: 在一所学校里，学生因个人兴趣缺乏而无法满足学校指定的课外活动要求，导致与同学产生冲突。学校规定必须选择指定课程，若拒绝后学生感到困惑。当学生课间独自练习个人兴趣活动违反规定，感到困惑。最佳应对方式是？\n\nOptions:\nA) 忽略冲突专注于其他课程\nB) 报告学校以了解规定\nC) 与同学沟通寻求解决问题的方法\nD) 试图找到同学的错误\n\nAnswer: C\n</question>'
        # batch_solution_str[0] = '<think>\n好，我现在需要根据提供的阅读材料来设计一个高难度的单项选择题。首先，我得仔细阅读并理解材料内容，找出其中的关键信息和主题。材料主要涉及澳大利亚南澳的乡报协会的活动、广告销售情况、推广策略、和国家媒体的动态等。核心问题可能是关于乡报广告销售下降的原因或解决方案。\n\n首先，我需要找出材料中的关键点。例如，材料中提到乡报的广告销售下降，尤其是广告费的减少和刊登广告的下降。还有，政府资助项目减少，导致了中游媒体的流失。还有推广策略如“Longer Country”计划，以及如何提高销量的措施，如增加登上国报的广告，或者引入新的广告业务。\n\n接下来，按照提供的指导方针，我需要设计一个题目，结合材料中的信息，但不直接引用原文，而是通过情境设置来考察理解。题目需要多步骤推理，有干扰项，并且答案在材料中有依据。\n\n考虑到材料中提到的广告销售下降，特别是乡报的广告费下降，可以作为核心。比如，为什么乡报的广告销售下降，可能是因为政府资助减少导致中游媒体流失，或者乡报缺乏宣传策略。另外，国家媒体仍在运行，但乡报可能因为自身策略不力而收入下降。\n\n根据出题步骤，我需要将问题设计成需要多步骤推理。例如，设计一个关于媒体下降的原因，但需要结合政府资助的减少和媒体战略变化。题目需要隐藏条件，不直接提到“政府资助减少”，而是描述一个情境，让答题者需要推断原因。\n\n可能的题目方向是：某地区报纸销售下降，原因是什么？但需要加入干扰项，比如经济因素、市场需求变化、管理问题等。但要确保所有必要信息在题干中给出，不能依赖原文。\n\n例如，可以构造一个关于媒体销售下降的原因，结合政府资助减少导致中游媒体流失... [省略] ...最直接的原因。\n</self-test>\n\n<correctness-check>\n### 分析过程：\n1. **理解题意和标准答案**：题目要求确定澳大利亚南澳州报纸广告销售下降的主要原因。标准答案为D，即“政府资助减少”。\n2. **提取用户回答的最终结论**：用户回答通过分析政府资助减少、本地媒体自身推广策略不佳以及商业广告成本上升的影响，最终得出结论，认为广告收入下降的主要原因是A) 经济下滑，但实际回答中提到D) 政府资助减少也是间接原因。\n3. **对比结果**：用户回答虽然详细分析了多种可能的原因，但最终结论并未明确指出标准答案D（政府资助减少），而是倾向于经济下滑。因此，根据题目要求严格对照标准答案，用户回答与标准答案不一致，判定为错误。\n\n```json\n{\n "判断结果": "错误"\n}\n```\n</correctbidden>\n</correctness-check>\n\n<question>\nQuestion: 在澳大利亚南澳州，一份报纸的广告销售下降，尽管广告需求保持稳定。政府削减了本地媒体的资助，导致周边媒体减少了广告投放。同时，该报纸自身推广策略不佳，导致客户转向其他媒体。商业广告成本上升。广告收入下降的主要原因是什么？\n\nOptions:\nA) 经济下滑\nB) 经济下滑\nC) 经济下滑\nD) 政府资助减少\n\nAnswer: D\n</question>\n\n<self-test>\nQuestion: 在澳大利亚南澳州，一份报纸的广告销售下降，尽管广告需求保持稳定。政府削减了本地媒体的资助，导致周边媒体减少了广告投放。同时，该报纸自身推广策略不佳，导致客户转向其他媒体。商业广告成本上升。广告收入下降的主要原因是什么？\n\nOptions:\nA) 经济下滑\nB) 经济下滑\nC) 经济下滑\nD) 政府资助减少\n\nAnswer: D\n</question>'
        print(task.compute_score(
            [None]*len(batch_solution_str), batch_solution_str, batch_ground_truth,
        ))


class TestDoc2QueryV4(unittest.TestCase):
    def test_evaluate_reference(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v4", num=4)
        task = Doc2QueryV4ComputeScore(
            doc2query_v4_parse_solution_fn, split="valid", args=DOC2QUERY_V4_DEFAULT_PARAMS)

        async def main():
            await task.polar_agent.compute_rm_score(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
            # results = await task.eval_reference(
            #     [None] *
            #     len(batch_solution_str), batch_solution_str, batch_ground_truth,
            # )
            # print(results)
        aio.run(main())


if __name__ == '__main__':
    pass
