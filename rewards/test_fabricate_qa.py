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
    QuestionSimilarity,
    ThoughtBonus,
    CalculationAnswerFormatVerify,
    LanguageConsistency,
    Doc2QueryV2ComputeScore,
    DOC2QUERY_DEFAULT_PARAMS,
    doc2query_v2_default_stage1_compute_score_valid,
    fabricate_qa_default_stage1_compute_score_valid,
    fabricate_aio_default_stage1_compute_score_valid,
    fabricate_aio_default_stage2_compute_score_valid,
    calc_qa_parse_solution_fn,
    calc_qa_parse_thought_fn,
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
    count = 0
    for _, row in df.iterrows():
        row = row.to_dict()
        if format == "wrong_question":
            gt = row["reward_model"]["question"]
            if gt is not None:
                batch_solution_str.append(
                    f'<think>\n{generate_random_string(100)}\n</think>\n\n<question>\nQuestion: {gt}\n\nAnswer: \\boxed{{78}}\n\nAnswer Type: NumericalAnswer\n</question>')
                batch_ground_truth.append(row["reward_model"])
                count += 1
            else:
                continue
        if format == "zh_question":
            if row["reward_model"]["lang_code"] != "zh":
                continue
            gt = row["reward_model"]["question"]
            if gt is not None:
                batch_solution_str.append(
                    f'<think>\n{generate_random_string(100)}\n</think>\n\n<question>\nQuestion: {gt}\n\nAnswer: \\boxed{{78}}\n\nAnswer Type: NumericalAnswer\n</question>')
                batch_ground_truth.append(row["reward_model"])
                count += 1
            else:
                continue
        if format == "fabricate_qa":
            if row["data_source"] != "fabricate_qa":
                continue
            gt = row["reward_model"]["question"]
            batch_solution_str.append(
                f'<think>\nSelf-Validation\n</think>\n\n<question>\nQuestion: {gt}\n\nAnswer: \\boxed{{78}}\n\nAnswer Type: NumericalAnswer\n</question>')
            batch_ground_truth.append(row["reward_model"])
            count += 1
            if count > num-1:
                break
            continue
        for q, a, _type in [
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
        ]:
            batch_solution_str.append(
                f'<think>\n{generate_random_string(100)}\n</think>\n\n<question>\nQuestion: {q}\n\nAnswer: {a}\n\nAnswer Type: {_type}\n</question>')
            if format == "doc2query_v2":
                batch_ground_truth.append({
                    "document": "<skip_content>",
                    "question": q,
                    "lang_code": "en",
                    "source": ""
                })
            count += 1
        if count > num-1:
            break
    return batch_solution_str, batch_ground_truth


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
        self.assertEqual(verifier.verify("\\boxed{-5}"), True)
        self.assertEqual(verifier.verify("\\boxed{0.500}"), True)
        self.assertEqual(verifier.verify("\\boxed{0.210}"), True)
        self.assertEqual(verifier.verify("\\boxed{5.5}"), False)
        self.assertEqual(verifier.verify("\\boxed{-0.0500}"), True)

    def test_calc_qa_parse_solution_fn(self):
        self.assertFalse(calc_qa_parse_solution_fn(
            '<think>\nssssss\n</think>\n\n<question>\n某公司生产密码设备，其产品批次密码t为两位正整数。密码设置规则要求：将t乘以生产效率系数K后，结果最后两位必须是36。效率系数K通过如下方式计算：去年设备计划运行250天，实际因维护停机30天，另5%时间用于年度升级。K值等于[(计划运行天数 - 停机天数 - 升级天数) × 0.05] + 1。同时，公司年度维护成本为80,000元，但这不影响K的计算。求满足条件的密码t值。  \nAnswer: \\boxed{76}  \nAnswer Type: NumericalAnswer  \n</question><|im_end|><｜end▁of▁sentence｜>'
        ))
        self.assertTrue(calc_qa_parse_solution_fn(
            "<think>\nssssss\n</think>\n\n<question>\nQuestion: A green energy factory optimizes its production parameters via a reaction governed by the equation:  \n\\[ a^2 + 3b^2 + \\frac{c^2 + 3d^2}{2} = a + b + c + d - 1 \\]  \nwhere \\( a \\), \\( b \\), \\( c \\), and \\( d \\) are operational parameters (unitless in the equation). The process involves two steps with yields of 82% and 76%, respectively. Given that the initial reagent InCl₃ has a purity of 85%, and the target is to produce 0.0350 moles of the final product, calculate the operational efficiency index \\( E = 1000a + 100b + 10c + d \\). Ignore distractor information: annual maintenance costs of ¥90,000 and equipment depreciation period of 7 years.  \nAnswer: \\boxed{527}  \nAnswer Type: NumericalAnswer  \n</question><|im_end|>"
        ))
        self.assertTrue(calc_qa_parse_solution_fn(
            "<think>\nOkay, let me try to create a question based on the given requirements. The respondent is an undergraduate with knowledge in calculus, vector operations, central forces, work done by a force, position and velocity vectors. The skills include integrating velocity to find position, calculating dot products, applying the work-energy theorem, and understanding central force properties. The difficulty is advanced, so I need to incorporate multiple steps and potential distractors.\n\nFirst, I need to think of a scenario that combines these elements. Maybe a physics problem involving a particle moving under a central force, where they have to calculate work done or perhaps find a position vect... [省略] ... places  correct.\n\nTherefore, the final question is as constructed above, with the answer being 479.366 kJ. Let me adjust the question\'s answer in the output.\n</think>\n\n<question>\nQuestion: A researcher compresses an ideal gas in a laboratory. Initially, the gas has a pressure of 3 bar, a volume of 2.5 m³, and a temperature of 25°C. After isobaric compression, the volume is reduced to 1.2 m³, and the temperature is measured as 158°F. The molar specific heat at constant pressure (Cp) for the gas is 35.2 J/mol·K. During the process, 500 J of non-volume work is done on the gas. Calculate the change in enthalpy (ΔH) of the gas in kJ, rounded to three decimal places. (Note: The ideal gas constant R = 8.314 J/mol·K.)\nAnswer: 479.366 kJ\nAnswer Type: WithUnitSymbol\n</question><|im_end|>"))
        batch_solution_str, batch_ground_truth = load_fabricate_aio_data(
            format="doc2query_v2", num=100)
        for _ in batch_solution_str:
            self.assertTrue(calc_qa_parse_solution_fn(_))

    def test_question_similarity(self):
        batch_solution_str, batch_ground_truth = load_fabricate_aio_data(
            format="wrong_question", num=100)

        scorer = QuestionSimilarity(calc_qa_parse_solution_fn)
        for response, gt in zip(batch_solution_str, batch_ground_truth):
            score = scorer.get_penalty_or_reward(response, gt)
            self.assertTrue(score > 0.5)

    def test_calc_answer_verify(self):
        batch_solution_str, batch_ground_truth = load_fabricate_aio_data(
            format="wrong_question", num=100)

        scorer = CalculationAnswerFormatVerify(calc_qa_parse_solution_fn)
        for response, gt in zip(batch_solution_str, batch_ground_truth):
            score = scorer.get_penalty_or_reward(response, gt)
            self.assertTrue(score >= 0)

    def test_language_consistency(self):
        batch_solution_str, batch_ground_truth = load_fabricate_aio_data(
            format="zh_question", num=100)

        scorer = LanguageConsistency(calc_qa_parse_solution_fn)
        for response, gt in zip(batch_solution_str, batch_ground_truth):
            score = scorer.get_penalty_or_reward(response, gt)
            self.assertTrue(score < 0)

    def test_thought_bonus(self):
        batch_solution_str, batch_ground_truth = load_fabricate_aio_data(
            format="zh_question", num=100)

        scorer = ThoughtBonus(calc_qa_parse_thought_fn)
        for response, gt in zip(batch_solution_str, batch_ground_truth):
            self.assertTrue(scorer.get_penalty_or_reward(response, gt) == 0)

    def test_doc2query_v2_get_difficulty_reward(self):
        batch_solution_str, batch_ground_truth = load_fabricate_aio_data(
            format="doc2query_v2", num=32)
        task = Doc2QueryV2ComputeScore(
            calc_qa_parse_solution_fn, split="valid", args=DOC2QUERY_DEFAULT_PARAMS)

        async def main():
            results = await task.get_difficulty_reward(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
                run_args={
                    "w/o_content": {
                        "model": task.get_weak_agent(),
                        "repeat": 8,
                        "fn": task.respond_wo_context,
                        "desc": 'w/o ctx'
                    },
                    "w_content": {
                        "model": task.get_strong_agent(),
                        "repeat": 2,
                        "fn": task.respond_w_context,
                        "desc": 'w ctx'
                    }
                }, debug=False,
                metric_args={
                    "advantage": 'w_content',
                    "weakness": 'w/o_content',
                    "advantage_oversimplified_threshold": 1.0,
                    "weakness_oversimplified_threshold": 7/8,
                    # "advantage_overcomplex_threshold": 1/2,
                    "advantage_overcomplex_threshold": 0.0,
                    "weakness_overcomplex_threshold": 1/8,
                    # "advantage_threshold": 1e-5,
                    "advantage_threshold": -9999.,
                    "advantage_weight": 0.5,
                    "weakness_weight": 0.5,
                    "confidence_bonus_threshold": 2/6,
                    "confidence_bonus_weight": 0.25
                }
            )
            assert len(results[0]) == len(results[1])
            print(results)
        aio.run(main())

    def test_doc2query_v2_get_similarity_reward(self):
        batch_solution_str, batch_ground_truth = load_fabricate_aio_data(
            format="doc2query_v2", num=32)
        task = Doc2QueryV2ComputeScore(
            calc_qa_parse_solution_fn, split="valid", args=DOC2QUERY_DEFAULT_PARAMS)

        async def main():
            results = await task.get_similarity_reward(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
                run_args={
                    "threshold": {
                        3: 0.5,
                        4: 1.0
                    },
                    "weight": 0.25,
                },
            )
            self.assertTrue(all(_ == 1.0 * 0.25 for _ in results))
        aio.run(main())

    def test_doc2query_v2_compute_score(self):
        batch_solution_str, batch_ground_truth = load_fabricate_aio_data(
            format="doc2query_v2", num=32)

        task = Doc2QueryV2ComputeScore(calc_qa_parse_solution_fn,
                                       split="valid", args=DOC2QUERY_DEFAULT_PARAMS)
        # print(task.compute_score([None]*len(batch_solution_str),
        #                          batch_solution_str, batch_ground_truth, stage="1"))
        print(doc2query_v2_default_stage1_compute_score_valid(
            [None]*len(batch_solution_str), batch_solution_str, batch_ground_truth,
        ))

    def test_fabricate_qa_compute_score(self):
        batch_solution_str, batch_ground_truth = load_fabricate_aio_data(
            format="fabricate_qa", num=32)
        print(fabricate_qa_default_stage1_compute_score_valid(
            [None]*len(batch_solution_str), batch_solution_str, batch_ground_truth,
        ))

    def test_fabricate_aio_compute_score(self):
        batch_solution_str, batch_ground_truth = load_fabricate_aio_data(
            format="doc2query_v2", num=32)
        sources = []
        for _ in range(len(batch_solution_str)):
            if random.random() > 0.5:
                sources.append("doc2query_v2")
            else:
                sources.append("fabricate_qa")
        rewards = fabricate_aio_default_stage2_compute_score_valid(
            sources, batch_solution_str, batch_ground_truth,
        )
        print(len(rewards), len(batch_solution_str))
        self.assertTrue(len(rewards) == len(batch_solution_str))


if __name__ == '__main__':
    pass
