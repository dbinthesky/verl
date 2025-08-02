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
    JudgeTwoQuestionSimilarity,
    MultichoiceKnowledgeQuestionQualityEval,
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
    RLVRVerify,
    RLVRComputeScore,
    fabricate_aio_compute_score_valid,
    xml_cot_fabricate_aio_compute_score_valid,
    rlvr_compute_score_valid,
    KG2QueryV1ComputeScore
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
        filename = "/cpfs01/shared/llm_ddd/tongjian/rl/doc2query_v3/doc2query_v3_kcle_rl_8k_train.parquet"

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
            if not xml_cot:
                batch_solution_str.append(
                    f'<think>\nUNITTEST_ONLY\n</think>\n\n<question>\nQuestion: {sample[0]}\n\nAnswer: {sample[1]}\n\nAnswer Type: {sample[2]}\n</question>'
                )
            else:
                batch_solution_str.append(
                    f'```xml\n<think>\nUNITTEST_ONLY\n</think>\n\n<conclusion>\n\nxxxxx\n<question>\nQuestion: {sample[0]}\n\nAnswer: {sample[1]}\n\nAnswer Type: {sample[2]}\n</question>\nsdfdsfs\n</conclusion>\n```'
                )
        elif row["data_source"] == "salt" or row["data_source"] == "kg2query_v1":
            sample = random.choice(_salt_qas)
            if not xml_cot:
                batch_solution_str.append(
                    f'<think>\nUNITTEST_ONLY\n</think>\n\n<question>\nQuestion: [SYNTHETIC] {sample[0]}\n\nAnswer: [SYNTHETIC] {sample[1]}\n</question>'
                )
            else:
                batch_solution_str.append(
                    f'```xml\n<think>\nUNITTEST_ONLY\n</think>\n\n<conclusion>\nxxxxx\n<question>\nQuestion: [SYNTHETIC] {sample[0]}\n\nAnswer: [SYNTHETIC] {sample[1]}\n</question>\n\nxxxxx\n</conclusion>\n```'
                )
        elif row["data_source"] == "doc2query_v3":
            sample = random.choice(_doc2query_v3_qas)
            o = "\n".join([f'{c}) {_o}' for c, _o in zip(
                ["A", "B", "C", "D"], sample[1])])
            if not xml_cot:
                batch_solution_str.append(
                    f'<think>\nUNITTEST_ONLY\n</think>\n\n<question>\nQuestion: [SYNTHETIC] {sample[0]}\n\nOptions: {o}\n\nAnswer: {sample[2]}\n</question>'
                )
            else:
                batch_solution_str.append(
                    f'```xml\n<think>\nUNITTEST_ONLY\n</think>\n\n<conclusion>\nxxxxx\n<question>\nQuestion: [SYNTHETIC] {sample[0]}\n\nOptions: {o}\n\nAnswer: {sample[2]}\n</question>\nyyyyyy\n</conclusion>\n```'
                )
        elif row["data_source"] == "rlvr":
            if not xml_cot:
                batch_solution_str.append(
                    f'<think>\nUNITTEST_ONLY\n</think>\n\n答案：{row["reward_model"]["ground_truth"]}'
                )
            else:
                batch_solution_str.append(
                    f'```xml\n<think>\nUNITTEST_ONLY\n</think>\n<conclusion>\n\n答案：{row["reward_model"]["ground_truth"]}\n</conclusion>\n```'
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
        # task = MultichoiceKnowledgeQuestionQualityEval()
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
        # batch_data_sources, batch_solution_str, batch_ground_truth = load_dataset(
        #     task_name=None, num=100,)
        # fabricate_aio_compute_score_valid(
        #     batch_data_sources, batch_solution_str, batch_ground_truth
        # )
        batch_data_sources, batch_solution_str, batch_ground_truth = load_dataset(
            task_name=None, num=100, xml_cot=True)
        xml_cot_fabricate_aio_compute_score_valid(
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

        rlvr_compute_score_valid(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth,
        )


class TestDoc2QueryV3(unittest.TestCase):
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
            doc2query_v3_parse_solution_fn, split="valid", args=DOC2QUERY_V3_DEFAULT_PARAMS)
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


class TestCriteriaRM(unittest.TestCase):
    def test_criteria_parse_solution_fn(self):
        batch_solution_str, batch_ground_truth = load_criteria_rm_data()
        print(criteria_parse_solution_fn(batch_solution_str[0]))

    # def test_compute_score(self):
    #     batch_solution_str, batch_ground_truth = load_criteria_rm_data()

    #     criteria_rm_default_compute_score_valid(
    #         [None] *
    #         len(batch_solution_str), batch_solution_str, batch_ground_truth,
    #     )


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


if __name__ == '__main__':
    pass
