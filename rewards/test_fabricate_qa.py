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
    doc2query_v3_fanout_parse_solution_fn,
    rlvr_shortcot_compute_score_valid,
    KG2QueryV1ComputeScore,
    LEARNABLE_COT_DEFAULT_PARAMS,
    LearnableCoTComputeScore,
    rlvr_shortcot_parse_solution_fn,
    DOC2QUERY_V4_DEFAULT_PARAMS,
    Doc2QueryV4ComputeScore,
    doc2query_v4_parse_solution_fn,
    Doc2QueryV5ComputeScore,
    doc2query_v5_parse_solution_fn,
    DOC2QUERY_V5_DEFAULT_PARAMS,
    Doc2QueryV6ComputeScore,
    Doc2QueryV6RefineComputeScore,
    _default_doc2query_v4_compute_score_train,
    doc2query_v6_refine_parse_solution_fn,
    DocRefinementHonestyJudge,
    DocRefinementHelpfulJudge,
    DocRefinementRedundencyJudge,
    DocRefinementReasonDepthJudge,
    DocRefinementHonestyFinegrainJudge,
    DocRefinementHelpfulFinegrainJudge,
    ConceptionClarityJudge,
    _default_doc2query_v6_refine_compute_score_train,
    kd_rewrite_parse_solution_fn,
    KDRewriteCriticalJudge,
    KDRewriteAtomicityJudge,
    KDRewriteIndependencyJudge,
    KDRewriteFactAndWeightJudge,
    KDRewriteQAQualityJudge,
    KDRewriteNLIJudge,
    KDRewriteComputeScore,
    KD_REWRITE_DEFAULT_PARAMS,
    cl_rewrite_parse_solution_fn,
    CLRewriteComputeScore,
    CL_REWRITE_DEFAULT_PARAMS,
    CLRewriteRubricsComputeScore,
    cl_rewrite_rubrics_parse_solution_fn,
    CL_REWRITE_RUBRICS_DEFAULT_PARAMS,
    CLRewriteRubricsOverlapJudge,
    CLRewriteSingleRubricJudge,
    CLRewriteSingleRubricRecallJudge,
    CLRewriteSingleRubricPrecisionJudge
)

UNITTEST_AGENT = Agent(**{
    "model": "qwen3_30b_a3b",
    "base_url": "http://10.102.202.11:28000/v1",
    "api_keys": "EMPTY",
    "request_kwargs": {
        "temperature": 0.7,
        "timeout": 360,
        "max_tokens": 2048,
    },
})

UNITTEST_FAST_AGENT = Agent(**{
    "model": "qwen3_30b_a3b",
    "base_url": "http://10.102.202.11:28000/v1",
    "api_keys": "EMPTY",
    "request_kwargs": {
        "temperature": 0.6,
        "timeout": 360,
        "max_tokens": 32,
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


def load_dataset(task_name, num=100, xml_cot=False):
    filename = "/mnt/shared-storage-user/ailab-hx/tongjian/rl/fabricate_aio/fabricate_aio_train_0730.parquet"
    if task_name == "kg2query_v1":
        filename = "/mnt/shared-storage-user/ailab-hx/tongjian/rl/fabricate_aio/kg2query_v1_oc_v1_7_hard_problem_0623.parquet"
    elif task_name == "doc2query_v3":
        filename = "/mnt/shared-storage-user/ailab-hx/tongjian/rl/doc2query_v3/mega_science_rl_inputs_train.parquet"
    elif task_name == "doc2query_v4":
        filename = "/mnt/shared-storage-user/ailab-hx/tongjian/rl/doc2query_v4/pretrain_general_doc_8k_rl_inputs_train_sample5k.parquet"
    elif task_name == "doc2query_v5":
        filename = "/mnt/shared-storage-user/ailab-hx/tongjian/rl/doc2query_v5/pretrain_general_doc_8k_rl_inputs_test.parquet"
    elif task_name == "doc2query_v6":
        filename = "/mnt/shared-storage-user/ailab-hx/tongjian/rl/doc2query_v6/pretrain_general_doc_8k_ms0_rl_inputs_test.parquet"
    elif task_name == "learnable_cot":
        filename = "/mnt/shared-storage-user/ailab-hx/tongjian/rl/learnable_cot/dapo_math_17k_train.parquet"
    elif task_name == "rlvr":
        filename = "/mnt/shared-storage-user/ailab-hx/tongjian/rl/rlvr/reason_v1_22_2025_12_14_nothink_subset2/index0.parquet"
    elif task_name == "doc2query_v6_refine":
        filename = "/mnt/shared-storage-user/ailab-hx/tongjian/rl/doc2query_v6_refine/dev_sample.parquet"
    elif task_name == "cl_rewrite":
        filename = "/mnt/shared-storage-user/ailab-hx/tongjian/rl/cl_rewrite/rl_inputs_dclm_2k_7k_with_rubrics/index0.parquet"

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
            def mock_solution(example):
                process = example["reward_model"]["extra_info"]["prompt_generation_process"]
                extra = example["reward_model"]["extra_info"]
                process = re.findall(r'<think>(.*)</think>', process, re.DOTALL)[0].strip()
                opt_str = "\n".join([f'{x}) {y}' for x, y in zip(["A", "B", "C", "D", "E", "F", "G"], extra["options"])])
                ans_str = ["A", "B", "C", "D", "E", "F", "G"][int(extra["options"].tolist().index(extra["answer"]))]
                
                return f'<think>\n{process}\n</think>\n\n<question>\nQuestion: {extra["question"]}\n\nOptions:\n{opt_str}\n\nAnswer: {ans_str}\n</question>'

            try:
                mock_solution(row)
            except Exception as err:
                continue
            batch_solution_str.append(mock_solution(row))
        elif row["data_source"] == "doc2query_v4":
            batch_solution_str.append(
                f'<question>\你是\n\n</question>\n\n<doc>\n{row["reward_model"]["document"]}\n</doc>'
            )
            row["reward_model"]["lang_code"] = "zh"
        elif row["data_source"] == "doc2query_v5":
            thought = 'I need to create a complete document based on the provided template. The document appears to be an academic paper about domain walls in magnetic materials with a focus on their thermal properties.\n\nThe document has many [MASK] placeholders that need to be filled in with appropriate text or content. Let me go through the document and identify all the [MASK] elements and determine what should be there to create a coherent and complete academic paper.\n\nI need to be careful to maintain the academic style and technical content of the paper while filling in the missing pieces.\n\nLet\'s go through the document section by section:\n\n1. The \\documentclass line has [MASK] which should probably be "12pt" or similar, but since it\'s a template, I\'ll keep it as [MASK] but in a completed version it should be filled.\n\n2. The author line has [MASK] which should probably be the title of the paper.\n\n3. The affiliation line has [MASK] which should be the institution name.\n\n4. The abstract has several [MASK] that need to be filled.\n\n5. The pacs line has [MASK] that needs to be filled with a PACS code.\n\n6. The main body of the document has numerous [MASK] that need to be completed.\n\nLet me go through each instance of [MASK] and determine what should be there:\n\n1. \\documentclass[prl,aps, [MASK] showpacs]{revtex4}\n   - This should probably be "[12pt]" or similar for the document size.\n\n2. \\author{N.\\ Kazantseva, R.\\ Wieser, U.\\ Nowak}\n   - No [MASK] here\n\n3. \\affiliation{Fachbereich Physik, [MASK]   47048 Duisburg, Germany}\n   - This should be the university or institution name (e.g., "University of Duisburg-Essen")\n\n4. [MASK] thermal properties. In general, the magnetization component...\n   - This should be the title of the paper (e.g., "Thermal Properties of Domain Walls in Constrained Geometries")\n\n5. \\pacs{\n  75.10.Hk \n  75.75.+a \n  75.60.Ch \n [MASK] 47.Jn \n}\n   - The PACS code should be the one that best fits the paper (e.g., 75.70.Cn for domain walls)\n\n6. [MASK] future magneto-electronic devices \\cite{allwoodSCIENCE02}. Especially,\n   - Should be a phrase like "Domain walls are expected to play a key role in"\n\n7. [MASK] important since those can show behavior deviating from their usual\n   - "Domain walls are particularly important since they can show"\n\n8. [MASK] as e. g. a controlled pinning and a strongly reduced\n   - "as e.g. a controlled pinning and a strongly reduced"\n\n9. [MASK]  and nano-contacts \\cite{garciaPRL99,chopraPRB02}.\n   - Should be "nano-constrictions"\n\n10. \\begin{figure}\n    \\begin{center} \n      \\includegraphics[width=7.5cm]{fig1.eps}\n    \\end{center} \n   [MASK]  The magnetization of the first\n     and the last plane are antiparallel along the easy axis. The\n     magnetization rotates in a plane via the hard axis.}\n   [MASK]  In the following we consider a domain wall structure as shown in\n   [MASK]  \\ref{f:scetch}. In the picture, each arrow represents the mean\n   magnetization of a plane. [MASK] within the planes while in the mean-field (MF) approximation each\n   (infinite) plane is represented as one magnetization vector so that\n   the model is effectively one dimensional. Depending on the details of\n   [MASK] first and the last plane of the system are fixed as shown in the\n   figure or we use anti-periodic boundary conditions. [MASK] force a domain wall into the system which for large system size\n   (number of planes $N$) will not fill up the whole system. However, for\n   smaller system size the boundary conditions force the domain wall to\n   adopt the system size. This models a domain wall caught in a\n   nano-constriction \\cite{brunoPRL99} where perfect pinning is assumed.\n   - The [MASK] should be the caption for the figure (e.g., "Schematic representation of a domain wall in a confined geometry,")\n\n11. [MASK]  the one dimensional integral\n    - Should be "d-dimensional"\n\n12. [MASK] = \\,\\int\\limits_{-\\pi/2}^\\phi\\frac{\\delta_0 {\\rm\n     d}\\theta}{\\sqrt{c^2-\\sin^2\\theta}} = \\,\\frac{\\delta_0}{c}{\\rm\n     F}\\left(\\phi, \\frac{1}{c^2}\\right)\n    - Should be "x(\\phi)"\n\n13. [MASK]  given by the boundary condition $x(\\pi/2) = L/2$ with $L = (N-1) a$.\n    - Should be "width"\n\n14. [MASK] \\begin{equation}\n     S_e(x) \\, = \\, \\tanh (x / \\delta_0)  \\quad\\quad S_h(x)\n     \\, = \\, \\cosh^{-1} (x / \\delta_0 ).\n     \\label{e:hyper}\n    \\end{equation}\n    - Should be "limit for the unconstrained wall"\n\n15. [MASK] and the profiles follow simple trigonometric functions,\n    - Should be "for a very constrained wall"\n\n16. [MASK]  Data points are from low temperature MF\n    calculations, solid lines correspond to\n    Eq. \\ref{e:jacob}. $D/J=0.003$.\n    - Should be "Figure"\n\n17. [MASK] expressions numerical MF results, obtained as described below in the\n    low temperature limit $\\kB T = 0.02J$. The smaller system shows the\n    case of a very constrained wall where the wall profiles are already\n    [MASK] intermediate case.  Note that in Ref. \\cite{brunoPRL99} corresponding\n    calculations were made, but for other boundary conditions and system\n    geometries, [MASK] However, we want to focus on thermal properties and in order to obtain\n    results for finite temperatures we start with the MF Hamilton\n    operator which (neglecting terms without $\\MB{S}$) is\n    [MASK]   {\\cal H}_{\\rm MF} = - \\sum_{i} \\Big( J \\MB{S}_i \\cdot (\\MB{m}_{i-1} \\! + \\!\n     4 \\MB{m}_i \\! + \\! \\MB{m}_{i+1} ) + D \\left(S_i^e\\right)^2 \\Big)\n     \\label{e:mf-ham}\n    \\end{equation}\n    - The first [MASK] should be "The figure", the second [MASK] should be "the," and the third [MASK] should be "the\n\n18. [MASK]   \\mathrm{Tr} \\; {\\mathbf S}_i\\EXP{-{\\cal H}_{\\rm MF}/\\kB T} \\nonumber\n    - Should be "m_i = \\frac{1}{Z} \\int {\\mathbf S}_i \\EXP{-{\\cal H}_{\\rm MF}/\\kB T} d\\Omega"\n\n19. [MASK]  ${\\mathbf m}_i$ is the thermally averaged\n    magnetization of the $i$th plane and the trace is an integral over the\n    unit sphere. [MASK]  starting with\n    an arbitrary magnetization profile and then let the equations evolve\n    until a stationary state is reached.\n    - First [MASK] should be "where", second [MASK] should be "We solve these equations by"\n\n20. [MASK]   \\caption {Easy- and hard-axis magnetization profiles in MF\n    approximation for two different temperatures. $D/J = 0. [MASK]   \\label{f:profile}\n    - First [MASK] should be "Figure", second [MASK] should be "03"\n\n21. [MASK]  Eqs. \\ref{e:hyper} but with a temperature dependent domain wall width\n    $\\delta(T)$ and temperature dependent amplitudes $M_h(T)$ and\n    $M_e(T)$.\n    - Should be "is reached when L >> δ_0"\n\n22. [MASK]  which describes the phase\n    transition of the hard-axis component of the magnetization vector. For\n    temperatures $T_h < T < T_c$ the domain wall is linear with an\n    easy-axis component of the magnetization only. For $T<T_h$ in general\n    one finds an elliptical wall profile which for lower temperature goes\n    over to the usual Bloch wall profile.\n    - Should be "the lower critical temperature"\n\n23. [MASK]  shows a kink at $T_h$, [MASK] regime approaching $T_c$.\n    - First [MASK] should be "The domain wall width", second [MASK] should be "The linear"\n\n24. [MASK]  Data points are from MF calculations as in\n    Fig. [MASK]  solid lines from Eq. \\ref{e:dtc}.\n    - First [MASK] should be "Figure", second [MASK] should be "4"\n\n25. [MASK] [D/J]^2$. [MASK]  \\ref{f:tcs} demonstrates, $T_h$ has also a linear\n    dependence on $D/J$ in the range of anisotropies which is shown in the\n    figure and our numerical results suggest $\\kB T_h/J \\approx 2 - 0.53\n    D/J$.\n    - First [MASK] should be "T_c = (2/3) * 2 * (D/J) + (2/3) * (D/J)^2"\n    - Second [MASK] should be "As seen in"\n\n26. [MASK] These findings, the second phase transition, its dependence on\n    [MASK] agreement with the earlier calculation within the framework of the\n    Ginzburg-Landau theory \\cite{bulaevskiiJETP64} --- an expansion close\n    [MASK] wall has a tanh-profile.  However, [MASK] anisotropies are rather small, so that in general the two critical\n    temperatures should nearly coincide and $M_h$ should be close to the\n    [MASK]  Nevertheless, experimental\n    investigations of linear domain walls exist. In \\cite{koetzlerPRL93}\n    [MASK] the domain wall mobility was investigated. Here, [MASK] $T_h$ from $T_c$ was 1\\\n    [MASK]  e.\\ g.\\, in\n    constrained geometries.\n    - [MASK] 1: "the anisotropy"\n    - [MASK] 2: "there is good"\n    - [MASK] 3: "to the one dimensional domain"\n    - [MASK] 4: "same as $M_e$"\n    - [MASK] 5: "a small difference between"\n    - [MASK] 6: "between"\n    - [MASK] 7: "K"\n\n27. [MASK] can be estimated analytically. For a system with antiperiodic boundary\n    [MASK]  so that $L \\ll \\delta_0$, the wall\n    profiles follow trigonometric functions which means that the angle of\n    rotation from plane to plane changes in each plane by the same amount\n    [MASK]  This leads to a decrease of the MF coming from the two\n    adjacent planes proportional to $\\cos(\\pi a /L)$.\n    - [MASK] 1: "conditions"\n    - [MASK] 2: "non-zero"\n    - [MASK] 3: "of $\\pi/L$"\n\n28. [MASK]   \\label{e:dth}\n    - Should be "\\kappa T_h/J = 2.0 [1 - \\cos(\\pi a / L)]"\n\n29. [MASK] can be estimated analytically. For a system with antiperiodic boundary\n    conditions. [MASK] so that $L \\ll \\delta_0$, the wall profiles follow trigonometric functions...\n    - [MASK] should be "To estimate the critical temperature $T_h$"\n\n30. [MASK]  The lateral dimension of the system is up to $128\n    [MASK]  The number of planes is\n    varied from 3 to 21 where we use fixed boundary conditions for the\n    [MASK]   We start the simulations with an abrupt domain\n    wall and let it relax for 10000MCS.  Then we calculate the absolute\n    value of the magnetization component perpendicular to the easy axis,\n    averaged over the whole system and for another 100000MCS as order\n    parameter of the phase transition. Note that the precise definition of\n    the order parameter is important: without calculating the absolute\n    value of the perpendicular magnetization the long time average of one\n    magnetization component will always be zero since the wall\n    [MASK]  Furthermore, to average\n    over the whole system is also important since the wall can move\n    diffusively.\n    - [MASK] 1: "x 128",\n    - [MASK] 2: "first and last plane",\n    - [MASK] 3: "rotates in the plane perpendicular to the easy axis"\n\n31. [MASK] $T_h$ is then determined from a finite-size scaling analysis of the\n    order parameter above where the lateral system size is varied from $8\n    \\times 8$ to $128 \\times 128$. The resulting data points are also\n    shown in Fig. \\ref{f:tcs2}. The scaling analysis works well with the\n    exponents $ \\beta$ and $\\nu$ from the three dimensional Heisenberg\n    [MASK]  Only for a very small number of planes deviations occur\n    [MASK] \\cite{garaninJPA96}.  As expected $T_h$ now goes to zero in the\n    limiting case of a tri-layer system so that the change of the critical\n    temperature is more dramatic. E. g., for a wall consisting of 5 atomic\n    planes ($L=4a$) $T_h$ is reduced by about 35\\ [MASK] Curie temperature and even for a temperature of 0.5$T_c$ the hard-axis\n    [MASK] about 20\\\n    - [MASK] 1: "model"\n    - [MASK] 2: "as discussed in"\n    - [MASK] 3: "of the"\n    - [MASK] 4: "magnetization is still 15\\%"\n\n32. To summarize, investigating the influence of thermal activation on the\n    properties of domain walls in nano-constrictions we have demonstrated\n    that with increasing temperature Bloch wall profiles change via\n    elliptical walls to linear domain walls. The temperature range where\n    [MASK] so that in general it is rather small. However, since in confined\n    geometries the relevant quantity is the size of the constriction\n    larger effects can be found. The break-down of ferromagnetic order is\n    [MASK] decreases due to the finite angle of rotation between neighboring\n    magnetic moments. [MASK]  it is a general effect which will also occur\n    [MASK]  g., [MASK]  Our findings\n    may have an impact on the understanding of domain wall\n    magneto-resistance properties\n    \\cite{dzeroPRB03,levyPRL97,yavorskyPRB02,bergeretPRB03}, especially its\n    temperature dependence, for two reasons: first, [MASK] linear domain wall the degree of spin disorder is larger in the wall\n    than in the bulk of the domain since the value of the order parameter\n    is lower. Second, as suggested in \\cite{dzeroPRB03}, in a linear\n    domain wall the change of the magnetization direction is abrupt while\n    only its value is changing. This means that the spin of a conductance\n    electron passing a linear wall cannot follow a continuously rotating\n    magnetization direction.\n    - [MASK] 1: "the transition to linear walls occurs over a narrow temperature range"\n    - [MASK] 2: "also reflected in the critical temperature"\n    - [MASK] 3: "in two- or three-dimensional systems, for example"\n    - [MASK] 4: "such as"\n    - [MASK] 5: "in a"\n\n33. [MASK] \\bibitem{allwoodSCIENCE02}\n    D.~A. Allwood {\\it et~al.}, Science {\\bf 296},  2003  (2002).\n    - This is a citation, should stay as is.\n\n34. [MASK]  {\\it et~al.}, Phys. Rev. Lett. {\\bf 83},  2425  (1999).\n    - Should be "P. Bruno, Phys. Rev. Lett. {\\bf 83}, 2425 (1999)."\n\n35. [MASK]  Pietzsch, A. Kubetzka, M. Bode, and R. Wiesendanger, Phys. Rev. Lett. [MASK]   84},  5212  (2000).\n    - Should be "T. S. K. Baron, B. Pietzsch, A. Kubetzka, M. Bode, and R. Wiesendanger, Phys. Rev. Lett. {\\bf 84}, 5212 (2000)."\n\n36. [MASK]  J. Appl. [MASK]  {\\bf 91},  3468  (2002).\n    - Should be "K. Miyake, J. Appl. Phys. {\\bf 91}, 3468 (2002)."\n\n37. [MASK]  M. [MASK] }, Phys. Rev. Lett. [MASK]   097202  (2003).\n    - Should be "M. Klaui, D. A. Back, J. A. Katine, and M. L. Schneider, Phys. Rev. Lett. {\\bf 91}, 097202 (2003)."\n\n38. [MASK]  {\\bf 84}, [MASK] \\bibitem{ruesterPRL03}\n    - Should be "U. Ebels et al., Phys. Rev. Lett. {\\bf 84}, 1205 (2000)."\n\n39. [MASK]  N. Munoz, and Y.-W. Zhao, Phys. Rev. [MASK]   2923  (1999).\n    - Should be "N. Garcia, J. N. Munoz, and Y.-W. Zhao, Phys. Rev. B {\\bf 59}, 2923 (1999)."\n\n40. [MASK]   \\bibitem{labayeJAP02}\n    Y. Labaye, L. [MASK]  and J.~M. [MASK]  J. Appl. Phys. {\\bf 91},  5341\n    [MASK]\n    - Should be "Y. Labaye, L. Berger, and J. M. D. Coey, J. Appl. Phys. {\\bf 91}, 5341 (2002)."\n\n41. [MASK]  D.  [MASK]  Phys. [MASK]  B {\\bf 64},  20407  (2003).\n    - Should be "J. M. D. Coey, L. Berger, and Y. Labaye, Phys. Rev. B {\\bf 64}, 20407 (2003)."\n\n42. [MASK]  L. N. [MASK] ~L. Ginzburg, [MASK]  JETP {\\bf 18},  530\n    (1964).\n    - Should be "L. N. Bulaevskii and V. L. Ginzburg, Zh. Eksp. Teor. Fiz. {\\bf 18}, 530 (1964)."\n\n43. [MASK]  K{\\"o}tzler, D.~A. Garanin, M. Hartl, and L. Jahn, Phys. Rev. Lett. {\\bf\n    71},  177  (1993).\n    - Should be "G. Koetzler, D. A. Garanin, M. Hartl, and L. Jahn, Phys. Rev. Lett. {\\bf 71}, 177 (1993)."\n\n44. [MASK]  M. Dzero, L.~P. Gor\\\'kov, A.~K. [MASK]  and K.~A. Zvezdin, [MASK]  Rev. B {\\bf\n    67},  100402  (2003).\n    - Should be "M. Dzero, L. P. Gor\'kov, A. K. Kolezhuk, and K. A. Zvezdin, Phys. Rev. B {\\bf 67}, 100402 (2003)."\n\n45. [MASK]  \\bibitem{nowakARCP01}\n    U. Nowak,  in {\\em Annual Reviews of Computational Physics IX}, edited by D.\n    Stauffer (World Scientific, Singapore, 2001), p.\\ 105.\n    - This is fine as is, but the citation might need "Phys. Rev." instead of "Rev.".\n\n46. [MASK]  Phys. A {\\bf 29},  2349  (1996).\n    - Should be "D. A. Garanin, J. Phys. A {\\bf 29}, 2349 (1996)."\n\n47. [MASK]  Rev. Lett. [MASK]   5110  (1997).\n    - Should be "P. M. Levy and S. Zhang, Phys. Rev. Lett. {\\bf 79}, 5110 (1997)."\n\n48. [MASK]  Y. [MASK] }, Phys. Rev. B {\\bf 66}, [MASK]\n    - Should be "S. Yavorsky, Y. Kozlov, and R. E. Camley, Phys. Rev. B {\\bf 66}, 174425 (2002)."\n\n49. [MASK]  Bergeret, A.~F. Volkov, [MASK]  Efetov, [MASK]  Rev. [MASK]   184403\n    (2003).\n    - Should be "F. S. B[2025-10-18T14:59:01+08:00] doc2query-v6-74376275-a787f-8f14e >> ergeret, A. F. Volkov, and K. B. Efetov, Phys. Rev. B {\\bf 67}, 184403 (2003)."\n\n50. [MASK] \n    - Should be "\\end{thebibliography}"\n\nNow I\'ll use these to create a completed document.'
            batch_solution_str.append(
                f'<think>\n我是{thought}\n\n</think>\n\n<doc>\n{row["reward_model"]["document"]}\n</doc>'
            )
            row["reward_model"]["lang_code"] = "zh"
        elif row["data_source"] == "doc2query_v6":
            thought = 'Let me analyze the task. I\'m supposed to construct the given document through the method of "posing questions and answering them myself," following a specific output format.\n\nThe document appears to be a collection of book reviews from an academic journal (likely an American Studies or literary studies journal) published by various university professors. There are reviews of books on Asian American literature, tourism, cookbooks, tennis history, European perceptions of American Indians, language history, female preaching in America, Benjamin Franklin\'s life, and early American nationalism.\n\nMy task is to create a "thinking process" where I pose questions about how I would have created this document and then answer them. I\'ll need to construct the entire document through this question-answering method before including the original document in the <doc> tag.\n\nLet me think about the questions I might ask and how I\'d answer them:\n\n1. What is the purpose of this document?\n2. What context and background led to creating this document?\n3. Who would be the likely author or editor of this document?\n4. What is the intended audience for this document?\n5. How was the content selected for this document?\n6. What criteria were used to evaluate the books being reviewed?\n7. How were these reviews structured?\n8. What were the challenges in writing these reviews?\n9. How did the reviews reflect the scholarly interests of the time?\n10. What was the editorial process for this document?\n\nI\'ll now construct the thinking process through posing and answering these questions. Then I\'ll include the original document exactly as provided.\n\nLet\'s create the thinking process:\n\nQ: What is the purpose of this document?\nA: This document is a collection of book reviews from an academic journal, likely focused on American Studies, Cultural Studies, or Literary Studies. The purpose is to provide scholarly evaluations of recently published works in these fields, helping readers understand the significance and scholarly contributions of each book.\n\nQ: Who would be the likely author or editor of this document?\nA: This document would be created by a scholarly journal editor or editorial board, possibly with contributions from multiple reviewers who are experts in their respective fields. The author of the reviews would be scholars at universities across the United States, as evidenced by the university affiliations listed at the end of each review.\n\nQ: What is the intended audience for this document?\nA: The intended audience would be scholars in American Studies, Cultural Studies, Literary Studies, History, and related fields, as well as graduate students and researchers looking for critical evaluations of recent scholarship.\n\nQ: How was the content selected for this document?\nA: The content was likely selected based on scholarly importance, timeliness, and relevance to current academic conversations. The books reviewed cover a diverse range of topics but are all connected to the broader fields of American culture, history, and identity.\n\nQ: What criteria were used to evaluate the books being reviewed?\nA: The reviewers seem to use criteria including scholarly rigor, contribution to the field, innovative methodology, scope of the argument, engagement with existing literature, and relevance to current debates. They also consider how the books address issues of race, gender, class, and cultural identity.\n\nQ: How were these reviews structured?\nA: Each review follows a consistent structure: 1) a summary of the book\'s main arguments and contributions; 2) a critical evaluation of strengths and weaknesses; 3) a discussion of how the book fits into broader scholarly conversations; 4) a conclusion about the book\'s significance.\n\nQ: What challenges might have been encountered in writing these reviews?\nA: The main challenges would include balancing thorough critical analysis with conciseness, addressing potential blind spots in the reviewed work, engaging with the complexity of the subject matter while keeping the review accessible, and maintaining scholarly objectivity while expressing critical insights.\n\nQ: How did these reviews reflect the scholarly interests of the time?\nA: These reviews reflect the dominant concerns of late 1990s American Studies: postcolonial theory, cultural studies, issues of race, gender, and identity, attention to marginalized voices, and concern with national narratives and representations.\n\nQ: What makes this document distinctive?\nA: This document stands out for its diverse range of subjects and the integration of current theoretical perspectives with substantive historical and literary analysis. The reviews also demonstrate a strong sense of how these books fit into ongoing academic conversations.\n\nQ: What was the editorial process for this document?\nA: The editorial process would likely involve the journal editor selecting books to be reviewed, soliciting reviews from experts in the field, editing the reviews for coherence and adherence to journal standards, and assembling them into a cohesive issue.\n\nQ: How might these reviews be useful for scholars?\nA: These reviews would be useful for scholars to determine which books are worth reading in depth, to understand how the book fits into the field, to identify potential methodological approaches, and to recognize debates and tensions within the scholarship on these topics.\n\nNow I\'ll present my constructed thinking process followed by the exact original document as instructed.'

            batch_solution_str.append(
                f'<think>\n{thought}\n</think>\n\n<doc>\n{row["reward_model"]["document"]}\n</doc>'
            )
            row["reward_model"]["lang_code"] = "zh"
        elif row["data_source"] == "rlvr" or row["data_source"] in ("aime_2024", "aime_2025"):
            batch_solution_str.append(
                f'<think>\nUNITTEST_ONLY\n</think>\n\n答案：{row["reward_model"]["ground_truth"]}'
            )
        elif row["data_source"] == "doc2query_v6_refine":
            if "reference" not in row["reward_model"]:
                continue
            batch_solution_str.append(
                row["reward_model"]["reference"]
            )
        elif row["data_source"] == "learnable_cot":
            batch_solution_str.append(
                f'<think>\nUNITTEST_ONLY\n</think>\n\n{row["reward_model"]["answer"]}'
            )
        elif row["data_source"] == "cl_rewrite":
            batch_solution_str.append(
                f'<think>\nUNITTEST_ONLY\n</think>\n\n<analysis>\nANALYSIS\n</analysis>\n\n<prev_chunk_id>{row["reward_model"]["prev_chunk_id"]}</prev_chunk_id>\n<next_chunk_id>{row["reward_model"]["next_chunk_id"]}</next_chunk_id>\n\n<reconstruction>\n{row["reward_model"]["completion"]}\n</reconstruction>'
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
                max_concurrent_requests=16,
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
        aio.run(main())

    def test_get_accuracy(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="rlvr", num=6)

        print(rlvr_shortcot_compute_score_valid(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth,
        ))


class TestDoc2QueryV3(unittest.TestCase):
    def test_doc2query_v3_format_verify(self):
        scorer = Doc2QueryV3FormatVerify(
            doc2query_v2_parse_solution_fn, -1.25, -0.75)
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v3", num=4)
        for x, y in zip(batch_solution_str, batch_ground_truth):
            self.assertEqual(scorer.get_penalty_or_reward(x, y), 0.0)

    def test_judgement(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v3", num=50)
        task = Doc2QueryV3ComputeScore(
            doc2query_v3_parse_solution_fn, split="valid", args=DOC2QUERY_V3_DEFAULT_PARAMS)

        async def main():
            # results = await task.explicit_question_design_judge(
            # results = await task.correctness_judge(
            # results = await task.close_book_solve_question(
            results = await task.judge_main_loop(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth
            )
            print(results)
        aio.run(main())

    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v3", num=90)
        task = Doc2QueryV3ComputeScore(
            doc2query_v3_parse_solution_fn, split="train", args=DOC2QUERY_V3_DEFAULT_PARAMS)
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


class TestDoc2QueryV4(unittest.TestCase):
    def test_agent(self):
        async def main():
            prompts = ["礼拜六后面的第三天是礼拜几？"]
            agent = UNITTEST_AGENT
            outputs = await agent.run(
                prompts,
                10, desc=f"[测试 {agent.model}=32]", postprocess_fns=[lambda x: x]*len(prompts))
            print(outputs)
        aio.run(main())


    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v4", num=4)

        async def main():
            await _default_doc2query_v4_compute_score_train._compute_score(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
        aio.run(main())


class TestDoc2QueryV6(unittest.TestCase):
    def test_calc_qa_coverage(self):
        testcase = 'Let me construct the document content through the method of "posing questions and answering them myself".\n\nQ: What is the main ethical dilemma presented in this question about a tech company implementing advanced privacy features?\nA: The ethical dilemma is whether a tech company should implement advanced privacy features in its products even if it means higher development costs and potentially lower profit margins.\n\nQ: What are the main ethical frameworks that could potentially address this dilemma?\nA: The main ethical frameworks that could address this dilemma are:\n1. Utilitarianism\n2. Deontological ethics\n3. Virtue ethics\n4. Stakeholder theory\n5. Shareholder theory\n6. Social contract theory\n7. Corporate social responsibility (CSR)\n8. Ethical relativism\n9. Ethical egoism\n10. None of the above\n\nQ: How would Utilitarianism approach this dilemma?\nA: Utilitarianism would approach this dilemma by considering the overall happiness and well-being of all affected parties. It would require a calculation of whether the benefits of protecting users\' privacy (e.g., increased user trust, potentially more long-term customers) outweigh the costs (higher development costs, lower short-term profits). It would focus on the consequences of implementing the privacy features.\n\nQ: How would Deontological ethics approach this dilemma?\nA: Deontological ethics would approach this dilemma by focusing on the moral duty to respect individuals\' rights to privacy. It would argue that privacy is a fundamental right that should be respected regardless of the consequences, including higher development costs or lower profit margins. It would emphasize that the company has an obligation to protect user privacy as a moral imperative.\n\nQ: How would Virtue ethics approach this dilemma?\nA: Virtue ethics would approach this dilemma by focusing on the moral character of the company and its commitment to honesty, integrity, and other virtues. It would ask whether implementing advanced privacy features aligns with the virtues the company should embody, rather than focusing on consequences or rights.\n\nQ: How would Stakeholder theory approach this dilemma?\nA: Stakeholder theory would approach this dilemma by considering the interests and well-being of all stakeholders, including users (privacy concerns), employees (company culture), shareholders (profit considerations), and the broader community (societal impact of privacy issues). It would weigh these different stakeholder interests to make a decision.\n\nQ: How would Shareholder theory approach this dilemma?\nA: Shareholder theory would approach this dilemma by prioritizing the interests of shareholders above all else. It would argue that the company should focus on maximizing returns for shareholders, and therefore should only implement privacy features if they clearly contribute to profits. If implementing privacy features would lead to lower profits, shareholder theory would likely recommend against doing so.\n\nQ: How would Social contract theory approach this dilemma?\nA: Social contract theory would approach this dilemma by considering the company\'s responsibility to uphold societal norms and agreements regarding privacy. It would examine whether implementing advanced privacy features aligns with implicit or explicit social contracts regarding privacy, and whether the company has an obligation to uphold these norms.\n\nQ: How would Corporate social responsibility (CSR) approach this dilemma?\nA: Corporate social responsibility would approach this dilemma by considering the company\'s voluntary commitment to go beyond legal requirements to protect user privacy. It would view privacy as part of the company\'s responsibility to society, even if it means taking on additional costs that aren\'t required by law.\n\nQ: How would Ethical relativism approach this dilemma?\nA: Ethical relativism would approach this dilemma by suggesting that ethical standards regarding privacy are culturally relative and should be adapted to local norms. It would argue that privacy standards and expectations vary across different cultures, so the company should implement privacy features appropriate for the specific cultural context of each market.\n\nQ: How would Ethical egoism approach this dilemma?\nA: Ethical egoism would approach this dilemma by arguing that acting in one\'s own self-interest (in this case, the company\'s self-interest) will ultimately lead to the greatest good. It would suggest that while implementing privacy features might seem to lower short-term profits, it would build trust with customers, potentially leading to greater long-term profits, so it is in the company\'s self-interest to do so.\n\nQ: How would "None of the above" approach this dilemma?\nA: "None of the above" would argue that no ethical framework can justify the higher costs and lower profit margins that would result from implementing advanced privacy features. It would suggest that the company should prioritize financial considerations over ethical ones in this case.\n\nQ: Which ethical framework best supports the decision to implement advanced privacy features, even if it means higher development costs and potentially lower profit margins?\nA: Deontological ethics best supports the decision to implement advanced privacy features, even if it means higher development costs and potentially lower profit margins. Deontological ethics emphasizes the moral duty to respect individuals\' rights to privacy, regardless of the consequences. In this scenario, the decision to implement privacy features aligns with the duty to uphold users\' rights, even if it negatively impacts financial outcomes.\n\nQ: Why isn\'t Utilitarianism the best answer?\nA: Utilitarianism is not the best answer because it would require a calculation of net benefits versus costs. While privacy features might eventually lead to more users trusting the company and thus more long-term profits, Utilitarianism would require an assessment of whether the benefits outweigh the costs. Deontological ethics, in contrast, would say the right thing to do is to respect privacy rights regardless of the consequences.\n\nQ: Why isn\'t Stakeholder theory the best answer?\nA: Stakeholder theory considers the interests of all stakeholders, but it still operates within a framework of considering interests rather than a strict moral duty. Deontological ethics goes beyond merely considering stakeholder interests; it says the company has a moral obligation to respect privacy as a right, regardless of how that affects other stakeholders.\n\nQ: Why isn\'t Corporate social responsibility (CSR) the best answer?\nA: CSR focuses on voluntary commitments, but it still implies a choice rather than a moral obligation. CSR would suggest that the company should go beyond legal requirements to protect privacy as a voluntary commitment, but it doesn\'t argue that it\'s morally wrong not to do so. Deontological ethics argues it\'s a moral duty, not just a voluntary social responsibility.\n\nQ: Why isn\'t the "None of the above" option correct?\nA: The "None of the above" option is incorrect because deontological ethics does provide a strong ethical justification for implementing privacy features, even at the cost of higher development expenses and potentially lower profit margins.\n\nQ: What is the final answer to the question?\nA: The answer is (B). Deontological ethics, as it emphasizes the moral duty to respect individuals\' rights to privacy, regardless of the consequences.'
        print(testcase)
        print(Doc2QueryV5ComputeScore.calc_qa_coverage(testcase))

    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v6", num=4)
        task = Doc2QueryV6ComputeScore(
            doc2query_v5_parse_solution_fn, split="valid", args=DOC2QUERY_V5_DEFAULT_PARAMS)

        async def main():
            await task._compute_score(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
        aio.run(main())


class TestDoc2QueryV6Refine(unittest.TestCase):
    def test_self_critique(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v6_refine", num=10)

        # task = DocRefinementHonestyJudge()
        # task = DocRefinementHelpfulJudge()
        # task = DocRefinementRedundencyJudge()
        # task = DocRefinementReasonDepthJudge()
        # task = DocRefinementHonestyFinegrainJudge()
        # task = DocRefinementHelpfulFinegrainJudge()
        task = ConceptionClarityJudge()

        eval_inputs = [(x, y)for x, y in zip(batch_solution_str, batch_ground_truth)]

        async def main():
            results = await task.do_job(
                agent=UNITTEST_AGENT,
                batch_inputs=eval_inputs,
                max_concurrent_requests=512,
            )
            print(results)
        aio.run(main())
                
    
    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            task_name="doc2query_v6_refine", num=10)
        async def main():
            await _default_doc2query_v6_refine_compute_score_train._compute_score(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
        aio.run(main())



class TestKDRewrite(unittest.TestCase):
    def load_dataset(self):
        solution_str = [
  {
    "content": "Tomball High School is a public high school in the United States, located at 30330 Quinn Road, Tomball, Harris County, Texas, USA, ZIP code 77375.",
    "fact": True,
    "weight": 4,
    "qas": [
      {
        "question": "Where is Tomball High School located?",
        "answer": "30330 Quinn Road, Tomball, Harris County, Texas, USA (ZIP code 77375)."
      },
      {
        "question": "In which city and state is Tomball High School located?",
        "answer": "Tomball, Texas."
      }
    ]
  },
  {
    "content": "The geographic coordinates of Tomball High School (a public high school in the United States) are 30°6′40″N, 95°37′43″W.",
    "fact": True,
    "weight": 3,
    "qas": [
      {
        "question": "What are the geographic coordinates of Tomball High School?",
        "answer": "30°6′40″N, 95°37′43″W."
      },
      {
        "question": "What are the latitude and longitude of Tomball High School?",
        "answer": "30°6′40″N, 95°37′43″W."
      }
    ]
  },
  {
    "content": "The decimal degree coordinates of Tomball High School (a public high school in the United States) are 30.11111°N, 95.62861°W.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What are the decimal degree coordinates of Tomball High School?",
        "answer": "30.11111°N, 95.62861°W."
      },
      {
        "question": "What are the coordinates (in decimal degrees) of Tomball High School?",
        "answer": "30.11111°N, 95.62861°W."
      }
    ]
  },
  {
    "content": "Tomball High School (a public high school in the United States) is an accredited public high school.",
    "fact": True,
    "weight": 3,
    "qas": [
      {
        "question": "What type of school is Tomball High School?",
        "answer": "An accredited public high school."
      },
      {
        "question": "Is Tomball High School accredited?",
        "answer": "Yes, Tomball High School is an accredited school."
      }
    ]
  },
  {
    "content": "The school motto of Tomball High School (a public high school in the United States) is \"The Pride of Texas\".",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What is the school motto of Tomball High School?",
        "answer": "\"The Pride of Texas\"."
      },
      {
        "question": "Which school has the motto \"The Pride of Texas\"?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "Tomball High School (a public high school in the United States) was founded in 1908.",
    "fact": True,
    "weight": 4,
    "qas": [
      {
        "question": "In which year was Tomball High School founded?",
        "answer": "1908."
      },
      {
        "question": "Which school was founded in 1908?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "The current campus of Tomball High School (a public high school in the United States) was built in 1974.",
    "fact": True,
    "weight": 3,
    "qas": [
      {
        "question": "In which year was the current campus of Tomball High School built?",
        "answer": "1974."
      },
      {
        "question": "Which school had its current campus built in 1974?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "Tomball High School (a public high school in the United States) is affiliated with the Tomball Independent School District.",
    "fact": True,
    "weight": 4,
    "qas": [
      {
        "question": "Which school district is Tomball High School part of?",
        "answer": "Tomball Independent School District."
      },
      {
        "question": "Which high school is part of the Tomball Independent School District?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "The principal of Tomball High School (a public high school in the United States) is Kimberle McKinney.",
    "fact": True,
    "weight": 3,
    "qas": [
      {
        "question": "Who is the principal of Tomball High School?",
        "answer": "Kimberle McKinney."
      },
      {
        "question": "Which school does Kimberle McKinney serve as principal?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "The faculty size of Tomball High School (a public high school in the United States) is 162.88 FTE.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What is the faculty size (FTE) of Tomball High School?",
        "answer": "162.88 FTE."
      },
      {
        "question": "Which school has a faculty size of 162.88 FTE?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "Tomball High School (a public high school in the United States) serves grades 9–12.",
    "fact": True,
    "weight": 3,
    "qas": [
      {
        "question": "Which grades does Tomball High School serve?",
        "answer": "Grades 9–12."
      },
      {
        "question": "Which school serves grades 9–12?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "In the 2023–2024 school year, Tomball High School (a public high school in the United States) had an enrollment of 3,078 students.",
    "fact": True,
    "weight": 3,
    "qas": [
      {
        "question": "What was the student enrollment of Tomball High School in the 2023–2024 school year?",
        "answer": "3,078 students."
      },
      {
        "question": "Which school had an enrollment of 3,078 students in the 2023–2024 school year?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "The student-teacher ratio of Tomball High School (a public high school in the United States) is 18.90.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What is the student-teacher ratio of Tomball High School?",
        "answer": "18.90."
      },
      {
        "question": "Which school has a student-teacher ratio of 18.90?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "The school colors of Tomball High School (a public high school in the United States) are red and white.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What are the school colors of Tomball High School?",
        "answer": "Red and white."
      },
      {
        "question": "Which school has red and white as its school colors?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "Tomball High School (a public high school in the United States) competes in the University Interscholastic League.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which athletic conference does Tomball High School compete in?",
        "answer": "University Interscholastic League."
      },
      {
        "question": "What is the University Interscholastic League to Tomball High School?",
        "answer": "Its athletics conference."
      }
    ]
  },
  {
    "content": "The mascot of Tomball High School (a public high school in the United States) is the Cougar.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What is the mascot of Tomball High School?",
        "answer": "The Cougar."
      },
      {
        "question": "Which school has the Cougar as its mascot?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "The contact phone number of Tomball High School (a public high school in the United States) is +1-281-357-3220.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What is the contact phone number of Tomball High School?",
        "answer": "+1-281-357-3220."
      },
      {
        "question": "Which school has the phone number +1-281-357-3220?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "Before Tomball Memorial High School opened in 2011, Tomball High School (a public high school in the United States) was the only high school in the Tomball Independent School District.",
    "fact": True,
    "weight": 3,
    "qas": [
      {
        "question": "Before Tomball Memorial High School opened in 2011, which was the only high school in Tomball ISD?",
        "answer": "Tomball High School."
      },
      {
        "question": "What was the status of Tomball High School in Tomball ISD before 2011?",
        "answer": "It was the only high school in the district."
      }
    ]
  },
  {
    "content": "In 1908, the first classes in Tomball were held in the Woodmen of the World building.",
    "fact": True,
    "weight": 3,
    "qas": [
      {
        "question": "In which building were the first classes in Tomball held in 1908?",
        "answer": "The Woodmen of the World building."
      },
      {
        "question": "What was the role of the Woodmen of the World building for Tomball schools in 1908?",
        "answer": "It was the location of the first classes."
      }
    ]
  },
  {
    "content": "In 1908, the Woodmen of the World building, where classes were held in Tomball, was a one-room framed building.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What kind of building was the Woodmen of the World building where Tomball classes were held in 1908?",
        "answer": "A one-room framed building."
      },
      {
        "question": "What was the structure of the Woodmen of the World building when used for classes in 1908?",
        "answer": "A one-room framed building."
      }
    ]
  },
  {
    "content": "In 1908, the Woodmen of the World building educated nearly 15 students.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Approximately how many students were educated in the Woodmen of the World building in 1908?",
        "answer": "Nearly 15 students."
      },
      {
        "question": "What was the approximate student size when Tomball first opened classes in 1908?",
        "answer": "Nearly 15 students."
      }
    ]
  },
  {
    "content": "In 1910, a school was built on donated land on Cherry Street.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "On donated land along which street was a school built in 1910?",
        "answer": "Cherry Street."
      },
      {
        "question": "In which year was the school on Cherry Street built?",
        "answer": "1910."
      }
    ]
  },
  {
    "content": "The Cherry Street school built in 1910 was a two-story red brick schoolhouse.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What type of building was the Cherry Street school built in 1910?",
        "answer": "A two-story red brick schoolhouse."
      },
      {
        "question": "In which year was the two-story red brick schoolhouse on Cherry Street built?",
        "answer": "1910."
      }
    ]
  },
  {
    "content": "The two-story red brick schoolhouse on Cherry Street, built in 1910, was part of the Harris County School System.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which school system did the 1910 Cherry Street red brick schoolhouse belong to?",
        "answer": "Harris County School System."
      },
      {
        "question": "Which 1910 schoolhouse was part of the Harris County School System?",
        "answer": "The two-story red brick schoolhouse on Cherry Street."
      }
    ]
  },
  {
    "content": "The first floor of the 1910 two-story red brick schoolhouse on Cherry Street contained 2 classrooms.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "How many classrooms were on the first floor of the 1910 Cherry Street two-story red brick schoolhouse?",
        "answer": "2 classrooms."
      },
      {
        "question": "Which schoolhouse had 2 classrooms on its first floor?",
        "answer": "The 1910 two-story red brick schoolhouse on Cherry Street."
      }
    ]
  },
  {
    "content": "The second floor of the 1910 two-story red brick schoolhouse on Cherry Street had an auditorium.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What facility was on the second floor of the 1910 Cherry Street two-story red brick schoolhouse?",
        "answer": "An auditorium."
      },
      {
        "question": "Which schoolhouse had an auditorium on its second floor?",
        "answer": "The 1910 two-story red brick schoolhouse on Cherry Street."
      }
    ]
  },
  {
    "content": "By 1916, the Tomball school had 62 enrolled students.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "How many students attended the Tomball school by 1916?",
        "answer": "62 students."
      },
      {
        "question": "In which year did the student count reach 62 at the Tomball school?",
        "answer": "1916."
      }
    ]
  },
  {
    "content": "As of 1928, the Tomball community had 4 schools.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "How many schools were in the Tomball community as of 1928?",
        "answer": "4 schools."
      },
      {
        "question": "How many schools did the Tomball community have in 1928?",
        "answer": "4 schools."
      }
    ]
  },
  {
    "content": "By 1931, the Tomball school district employed 5 teachers.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "How many teachers did the Tomball school district employ by 1931?",
        "answer": "5 teachers."
      },
      {
        "question": "How many teachers worked in the Tomball school district in 1931?",
        "answer": "5 teachers."
      }
    ]
  },
  {
    "content": "In 1935, the original school building was demolished and replaced with a new buff brick building.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "When was the original school building demolished and replaced with a new one?",
        "answer": "1935."
      },
      {
        "question": "What type of building replaced the original schoolhouse in 1935?",
        "answer": "A buff brick building."
      }
    ]
  },
  {
    "content": "The new buff brick building constructed in 1935 housed grades 1–11.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which grades did the 1935 new buff brick building house?",
        "answer": "Grades 1 through 11."
      },
      {
        "question": "In which year was the building housing grades 1–11 constructed?",
        "answer": "1935."
      }
    ]
  },
  {
    "content": "In 1937 (two years after 1935), a petition was submitted to the Harris County School Board requesting that Tomball have its own school district.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "To which body was the petition requesting Tomball to have its own school district submitted?",
        "answer": "Harris County School Board."
      },
      {
        "question": "In which year was the petition requesting Tomball to have its own school district submitted?",
        "answer": "1937."
      }
    ]
  },
  {
    "content": "The Harris County School Board approved the petition for Tomball to establish its own school district.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which board approved the petition for Tomball to have its own school district?",
        "answer": "Harris County School Board."
      },
      {
        "question": "What petition did the Harris County School Board approve?",
        "answer": "The petition for Tomball to have its own school district."
      }
    ]
  },
  {
    "content": "After the Harris County School Board approved the petition, all funds and debts were transferred to the Tomball Independent School District.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "To which school district were all funds and debts transferred after the petition was approved?",
        "answer": "Tomball Independent School District."
      },
      {
        "question": "What did the Tomball Independent School District receive after the petition was approved?",
        "answer": "All funds and debts."
      }
    ]
  },
  {
    "content": "By 1938, the school on Cherry Street had grown to require a second campus.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "By which year did the Cherry Street school need a second campus?",
        "answer": "1938."
      },
      {
        "question": "Why was a second campus needed in 1938?",
        "answer": "The Cherry Street school had grown in size."
      }
    ]
  },
  {
    "content": "In 1938, the Tomball Independent School District built a new red brick high school in the 700 block of Main Street.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "In which block of Main Street was the new red brick high school built in 1938?",
        "answer": "The 700 block."
      },
      {
        "question": "Which school district built the new red brick high school on Main Street in 1938?",
        "answer": "Tomball Independent School District."
      }
    ]
  },
  {
    "content": "The new red brick high school built on Main Street in 1938 was equipped with a gymnasium.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What athletic facility was included in the 1938 Main Street new red brick high school?",
        "answer": "A gymnasium."
      },
      {
        "question": "Which high school, built on Main Street in 1938, was equipped with a gymnasium?",
        "answer": "The new red brick high school on Main Street."
      }
    ]
  },
  {
    "content": "The new red brick high school built on Main Street in 1938 was equipped with a swimming pool.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What facility was included in the 1938 Main Street new red brick high school?",
        "answer": "A swimming pool."
      },
      {
        "question": "Which high school, built in 1938, was equipped with a swimming pool?",
        "answer": "The new red brick high school on Main Street."
      }
    ]
  },
  {
    "content": "In 1961, the original high school on Main Street was destroyed by an electrical fire in the attic.",
    "fact": True,
    "weight": 3,
    "qas": [
      {
        "question": "Why was the original high school on Main Street destroyed in 1961?",
        "answer": "By an electrical fire in the attic."
      },
      {
        "question": "In which year was the original high school on Main Street destroyed?",
        "answer": "1961."
      }
    ]
  },
  {
    "content": "The Houston Fire Department responded to the 1961 fire at the original Main Street high school.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which department responded to the 1961 fire at the original Main Street high school?",
        "answer": "Houston Fire Department."
      },
      {
        "question": "What incident did the Houston Fire Department respond to in 1961?",
        "answer": "The fire at the original Main Street high school."
      }
    ]
  },
  {
    "content": "During the 1961 fire at the original Main Street high school, firefighters ran hoses to the swimming pool to draw water for extinguishing the fire.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "How did firefighters get water to put out the 1961 fire at the original Main Street high school?",
        "answer": "They ran hoses to the swimming pool to draw water."
      },
      {
        "question": "What water source did firefighters use during the 1961 Main Street high school fire?",
        "answer": "The swimming pool."
      }
    ]
  },
  {
    "content": "After the 1961 fire at the original Main Street high school, schools were closed for 4 days.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "How long were schools closed after the 1961 fire?",
        "answer": "4 days."
      },
      {
        "question": "Which incident led to schools being closed for 4 days?",
        "answer": "The 1961 fire at the original Main Street high school."
      }
    ]
  },
  {
    "content": "After the 1961 fire at the original Main Street high school, churches and other businesses in the community provided space for high school students to complete the school year.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which organizations provided space for high school students to finish the school year after the 1961 fire?",
        "answer": "Churches and other businesses in the community."
      },
      {
        "question": "What help did community churches and businesses provide after the 1961 fire?",
        "answer": "They provided space for high school students to complete the school year."
      }
    ]
  },
  {
    "content": "By 1974, students began attending the new Tomball High School located on Sandy Lane.",
    "fact": True,
    "weight": 3,
    "qas": [
      {
        "question": "When did students start attending the new Tomball High School on Sandy Lane?",
        "answer": "1974."
      },
      {
        "question": "On which road did students begin attending the new campus in 1974?",
        "answer": "Sandy Lane."
      }
    ]
  },
  {
    "content": "Sandy Lane was later renamed Quinn Road.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What was Sandy Lane later renamed to?",
        "answer": "Quinn Road."
      },
      {
        "question": "What was the former name of Quinn Road?",
        "answer": "Sandy Lane."
      }
    ]
  },
  {
    "content": "In 2000, voters approved a $98.4 million bond referendum that allowed the school district to renovate Tomball High School.",
    "fact": True,
    "weight": 3,
    "qas": [
      {
        "question": "What was the amount of the 2000 bond referendum approved to renovate Tomball High School?",
        "answer": "$98.4 million."
      },
      {
        "question": "What did the 2000 bond referendum allow the school district to do?",
        "answer": "Renovate Tomball High School."
      }
    ]
  },
  {
    "content": "In spring 2011, Tomball High School principal Gary Moss planned to retire.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "When did Tomball High School principal Gary Moss plan to retire?",
        "answer": "Spring 2011."
      },
      {
        "question": "Who was the Tomball High School principal planning to retire in spring 2011?",
        "answer": "Gary Moss."
      }
    ]
  },
  {
    "content": "In fall 2011, Greg Quinn was promoted from assistant principal of Tomball High School to principal.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which school did Greg Quinn become principal of in fall 2011?",
        "answer": "Tomball High School."
      },
      {
        "question": "What position did Greg Quinn hold at Tomball High School before becoming principal?",
        "answer": "Assistant principal."
      }
    ]
  },
  {
    "content": "Greg Quinn served as principal of Tomball High School from fall 2011 to spring 2017.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "When was Greg Quinn the principal of Tomball High School?",
        "answer": "From fall 2011 to spring 2017."
      },
      {
        "question": "Until when did Greg Quinn serve as principal of Tomball High School?",
        "answer": "Spring 2017."
      }
    ]
  },
  {
    "content": "Chris Scott is the current principal of Tomball High School.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Who is the current principal of Tomball High School?",
        "answer": "Chris Scott."
      },
      {
        "question": "Which school does Chris Scott serve as principal?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "Tomball High School was remodeled in 2005.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "When was Tomball High School remodeled?",
        "answer": "2005."
      },
      {
        "question": "Which school was remodeled in 2005?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "After the 2005 renovation, a cougar statue in front of Tomball High School was donated by an artist from Magnolia.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Who donated the cougar statue in front of Tomball High School?",
        "answer": "An artist from Magnolia."
      },
      {
        "question": "Which year's renovation is associated with the cougar statue in front of Tomball High School?",
        "answer": "2005."
      }
    ]
  },
  {
    "content": "In 2021, Tomball High School fundraised for a new cougar statue.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What did Tomball High School fundraise for in 2021?",
        "answer": "A new cougar statue."
      },
      {
        "question": "In which year did Tomball High School fundraise for a new cougar statue?",
        "answer": "2021."
      }
    ]
  },
  {
    "content": "In the 2018–2019 school year, the Texas Education Agency rated Tomball High School a B with a total score of 88/100.",
    "fact": True,
    "weight": 3,
    "qas": [
      {
        "question": "What was the total score given to Tomball High School by the Texas Education Agency in the 2018–2019 school year?",
        "answer": "88/100."
      },
      {
        "question": "What was the TEA rating for Tomball High School in the 2018–2019 school year?",
        "answer": "B."
      }
    ]
  },
  {
    "content": "In the 2018–2019 school year, the Texas Education Agency gave Tomball Memorial High School a total score of 93/100.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What was the TEA total score for Tomball Memorial High School in the 2018–2019 school year?",
        "answer": "93/100."
      },
      {
        "question": "Which school received a TEA total score of 93/100 in the 2018–2019 school year?",
        "answer": "Tomball Memorial High School."
      }
    ]
  },
  {
    "content": "In the 2018–2019 school year, Tomball High School received an A grade in the Student Achievement domain from the Texas Education Agency, with a score of 92.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What was the Student Achievement score for Tomball High School in the 2018–2019 school year?",
        "answer": "92."
      },
      {
        "question": "What grade did Tomball High School receive in the Student Achievement domain in the 2018–2019 school year?",
        "answer": "A."
      }
    ]
  },
  {
    "content": "In the 2018–2019 school year, Tomball High School received a B grade in the School Progress domain from the Texas Education Agency, with a score of 81.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What was the School Progress score for Tomball High School in the 2018–2019 school year?",
        "answer": "81."
      },
      {
        "question": "What grade did Tomball High School receive in the School Progress domain in the 2018–2019 school year?",
        "answer": "B."
      }
    ]
  },
  {
    "content": "In the 2018–2019 school year, Tomball High School received a C grade in the Closing the Gaps domain from the Texas Education Agency, with a score of 79.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What was the Closing the Gaps score for Tomball High School in the 2018–2019 school year?",
        "answer": "79."
      },
      {
        "question": "What grade did Tomball High School receive in the Closing the Gaps domain in the 2018–2019 school year?",
        "answer": "C."
      }
    ]
  },
  {
    "content": "In the 2018–2019 school year, Tomball High School received the distinction designation of Academic Achievement (Social Studies).",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which distinction designation did Tomball High School receive in the 2018–2019 school year?",
        "answer": "Academic Achievement (Social Studies)."
      },
      {
        "question": "Which subject is the Academic Achievement distinction for Tomball High School?",
        "answer": "Social Studies."
      }
    ]
  },
  {
    "content": "In the 2019–2020 school year, U.S. News awarded Tomball High School a score of 81.41/100.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What score did U.S. News give Tomball High School in the 2019–2020 school year?",
        "answer": "81.41/100."
      },
      {
        "question": "What was the U.S. News score for Tomball High School in the 2019–2020 school year?",
        "answer": "81.41/100."
      }
    ]
  },
  {
    "content": "In the 2019–2020 school year, Tomball High School ranked 275th among Texas high schools (U.S. News).",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What was Tomball High School's rank among Texas high schools (2019–2020, U.S. News)?",
        "answer": "275th."
      },
      {
        "question": "Where did U.S. News rank Tomball High School in Texas for the 2019–2020 school year?",
        "answer": "275th."
      }
    ]
  },
  {
    "content": "In the 2019–2020 school year, Tomball High School ranked 3,307th among U.S. high schools (U.S. News).",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What was Tomball High School's rank among U.S. high schools (2019–2020, U.S. News)?",
        "answer": "3,307th."
      },
      {
        "question": "Where did U.S. News rank Tomball High School nationally for the 2019–2020 school year?",
        "answer": "3,307th."
      }
    ]
  },
  {
    "content": "In 2020, 43% of students at Tomball High School took at least one AP (Advanced Placement) exam (U.S. News data).",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What percentage of Tomball High School students took at least one AP exam in 2020?",
        "answer": "43%."
      },
      {
        "question": "What share of Tomball High School students took at least one AP exam in 2020?",
        "answer": "43%."
      }
    ]
  },
  {
    "content": "In 2020, among Tomball High School students who took at least one AP exam, 68% passed at least one AP exam (U.S. News data).",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What percentage of Tomball High School students who took AP exams passed at least one in 2020?",
        "answer": "68%."
      },
      {
        "question": "What was the pass rate (at least one exam) for Tomball High School AP test-takers in 2020?",
        "answer": "68%."
      }
    ]
  },
  {
    "content": "In 2020, among Tomball High School students who took at least one AP exam, 32% did not pass any AP exams (U.S. News data).",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What percentage of Tomball High School AP test-takers did not pass any exams in 2020?",
        "answer": "32%."
      },
      {
        "question": "What share of Tomball High School students took AP exams but failed all of them in 2020?",
        "answer": "32%."
      }
    ]
  },
  {
    "content": "Tomball High School is a member of Region II, District 15-6A of the University Interscholastic League.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which Region and District of the UIL does Tomball High School belong to?",
        "answer": "Region II, District 15-6A."
      },
      {
        "question": "Which school is a member of UIL Region II, District 15-6A?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "In 2013, the Tomball High School baseball team won the Texas UIL 4A state championship with a record of 36–4.",
    "fact": True,
    "weight": 3,
    "qas": [
      {
        "question": "What championship did the Tomball High School baseball team win in 2013?",
        "answer": "Texas UIL 4A state championship."
      },
      {
        "question": "What was the record of the Tomball High School baseball team in 2013?",
        "answer": "36–4."
      }
    ]
  },
  {
    "content": "In 2024, the Tomball High School baseball team won the Texas UIL 6A state championship with a record of 41–3.",
    "fact": True,
    "weight": 3,
    "qas": [
      {
        "question": "What championship did the Tomball High School baseball team win in 2024?",
        "answer": "Texas UIL 6A state championship."
      },
      {
        "question": "What was the record of the Tomball High School baseball team in 2024?",
        "answer": "41–3."
      }
    ]
  },
  {
    "content": "In the 2024 Texas UIL 6A baseball season, C.J. Sampson, a sophomore starting pitcher for Tomball High School, had a record of 14–0.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What was the record of Tomball High School pitcher C.J. Sampson in 2024?",
        "answer": "14–0."
      },
      {
        "question": "Who posted a 14–0 record in the 2024 Texas UIL 6A baseball season?",
        "answer": "C.J. Sampson."
      }
    ]
  },
  {
    "content": "In the 2024 Texas UIL 6A baseball season, Karson Reeder, a sophomore starting pitcher for Tomball High School, had a record of 14–1.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What was the record of Tomball High School pitcher Karson Reeder in 2024?",
        "answer": "14–1."
      },
      {
        "question": "Who posted a 14–1 record in the 2024 Texas UIL 6A baseball season?",
        "answer": "Karson Reeder."
      }
    ]
  },
  {
    "content": "In the 2024 Texas UIL 6A baseball playoffs, C.J. Sampson of Tomball High School was named Most Valuable Player (MVP) of the semifinals.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Who was named semifinal MVP of the 2024 Texas UIL 6A baseball playoffs?",
        "answer": "C.J. Sampson."
      },
      {
        "question": "What honor did C.J. Sampson receive in the 2024 Texas UIL 6A baseball playoffs?",
        "answer": "Semifinals Most Valuable Player (MVP)."
      }
    ]
  },
  {
    "content": "In the 2024 Texas UIL 6A baseball playoffs, Karson Reeder of Tomball High School was named Most Valuable Player (MVP) of the championship game.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Who was named championship game MVP of the 2024 Texas UIL 6A baseball playoffs?",
        "answer": "Karson Reeder."
      },
      {
        "question": "What honor did Karson Reeder receive in the 2024 Texas UIL 6A baseball playoffs?",
        "answer": "Championship game Most Valuable Player (MVP)."
      }
    ]
  },
  {
    "content": "Cade Arrimbade, a senior catcher for Tomball High School, was named Texas Gatorade Player of the Year (year not specified in the text).",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which Tomball High School player was named Texas Gatorade Player of the Year?",
        "answer": "Cade Arrimbade."
      },
      {
        "question": "What honor did Cade Arrimbade receive?",
        "answer": "Texas Gatorade Player of the Year."
      }
    ]
  },
  {
    "content": "In the 2018–2019 school year, Tomball High School had a total enrollment of 2,054 students.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "How many students attended Tomball High School in the 2018–2019 school year?",
        "answer": "2,054 students."
      },
      {
        "question": "Which school year had a total enrollment of 2,054 students at Tomball High School?",
        "answer": "The 2018–2019 school year."
      }
    ]
  },
  {
    "content": "In the 2018–2019 school year, 4.5% of Tomball High School students were African American.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What percentage of Tomball High School students were African American in the 2018–2019 school year?",
        "answer": "4.5%."
      },
      {
        "question": "Which school year had 4.5% African American students at Tomball High School?",
        "answer": "The 2018–2019 school year."
      }
    ]
  },
  {
    "content": "In the 2018–2019 school year, 2.1% of Tomball High School students were Asian.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What percentage of Tomball High School students were Asian in the 2018–2019 school year?",
        "answer": "2.1%."
      },
      {
        "question": "Which school year had 2.1% Asian students at Tomball High School?",
        "answer": "The 2018–2019 school year."
      }
    ]
  },
  {
    "content": "In the 2018–2019 school year, 33.0% of Tomball High School students were Hispanic.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What percentage of Tomball High School students were Hispanic in the 2018–2019 school year?",
        "answer": "33.0%."
      },
      {
        "question": "Which school year had 33.0% Hispanic students at Tomball High School?",
        "answer": "The 2018–2019 school year."
      }
    ]
  },
  {
    "content": "In the 2018–2019 school year, 0.3% of Tomball High School students were American Indian.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What percentage of Tomball High School students were American Indian in the 2018–2019 school year?",
        "answer": "0.3%."
      },
      {
        "question": "Which school year had 0.3% American Indian students at Tomball High School?",
        "answer": "The 2018–2019 school year."
      }
    ]
  },
  {
    "content": "In the 2018–2019 school year, 57.1% of Tomball High School students were White.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What percentage of Tomball High School students were White in the 2018–2019 school year?",
        "answer": "57.1%."
      },
      {
        "question": "Which school year had 57.1% White students at Tomball High School?",
        "answer": "The 2018–2019 school year."
      }
    ]
  },
  {
    "content": "In the 2018–2019 school year, 3.0% of Tomball High School students were of two or more races.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What percentage of Tomball High School students were of two or more races in the 2018–2019 school year?",
        "answer": "3.0%."
      },
      {
        "question": "Which school year had 3.0% multiracial students at Tomball High School?",
        "answer": "The 2018–2019 school year."
      }
    ]
  },
  {
    "content": "In the 2018–2019 school year, 27.4% of Tomball High School students were economically disadvantaged.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What percentage of Tomball High School students were economically disadvantaged in the 2018–2019 school year?",
        "answer": "27.4%."
      },
      {
        "question": "Which school year had 27.4% economically disadvantaged students at Tomball High School?",
        "answer": "The 2018–2019 school year."
      }
    ]
  },
  {
    "content": "In the 2018–2019 school year, 3.9% of Tomball High School students were English Language Learners.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What percentage of Tomball High School students were English Language Learners in the 2018–2019 school year?",
        "answer": "3.9%."
      },
      {
        "question": "Which school year had 3.9% English Language Learners at Tomball High School?",
        "answer": "The 2018–2019 school year."
      }
    ]
  },
  {
    "content": "In the 2018–2019 school year, 8.8% of Tomball High School students received Special Education services.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "What percentage of Tomball High School students received Special Education services in the 2018–2019 school year?",
        "answer": "8.8%."
      },
      {
        "question": "Which school year had 8.8% of students receiving Special Education services at Tomball High School?",
        "answer": "The 2018–2019 school year."
      }
    ]
  },
  {
    "content": "Christopher S. Adams Jr. is a notable alumnus of Tomball High School; he is an author and a retired United States Air Force general.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which Tomball High School alumnus is an author and a retired U.S. Air Force general?",
        "answer": "Christopher S. Adams Jr."
      },
      {
        "question": "Which school counts Christopher S. Adams Jr. as a notable alumnus?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "Chad Brannon is a notable alumnus of Tomball High School; he is a television actor.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which Tomball High School alumnus is a television actor?",
        "answer": "Chad Brannon."
      },
      {
        "question": "What is Chad Brannon's connection to Tomball High School?",
        "answer": "He is a notable alumnus."
      }
    ]
  },
  {
    "content": "Josh Breaux is a notable alumnus of Tomball High School; he is a baseball player in the New York Yankees organization.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which Tomball High School alumnus plays baseball in the New York Yankees organization?",
        "answer": "Josh Breaux."
      },
      {
        "question": "Is Josh Breaux a notable alumnus of Tomball High School?",
        "answer": "Yes."
      }
    ]
  },
  {
    "content": "Drake Britton is a notable alumnus of Tomball High School; he is a baseball player for the Boston Red Sox.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which Tomball High School alumnus played for the Boston Red Sox?",
        "answer": "Drake Britton."
      },
      {
        "question": "Which school counts Drake Britton as a notable alumnus?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "Jimmy Butler, a basketball player, is a notable alumnus of Tomball High School.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which famous basketball player is an alumnus of Tomball High School?",
        "answer": "Jimmy Butler."
      },
      {
        "question": "Which high school counts Jimmy Butler as a notable alumnus?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "Karlie Hay is a notable alumnus of Tomball High School; she was Miss Teen USA 2016.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which Tomball High School alumna was Miss Teen USA 2016?",
        "answer": "Karlie Hay."
      },
      {
        "question": "What is Karlie Hay's connection to Tomball High School?",
        "answer": "She is a notable alumna."
      }
    ]
  },
  {
    "content": "Chris Herrmann is a notable alumnus of Tomball High School; he is a professional baseball player.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which Tomball High School alumnus is the baseball player Chris Herrmann?",
        "answer": "Chris Herrmann."
      },
      {
        "question": "Which school counts Chris Herrmann as a notable alumnus?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "Jimmy Needham is a notable alumnus of Tomball High School; he is a contemporary Christian musician.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which Tomball High School alumnus is a contemporary Christian musician?",
        "answer": "Jimmy Needham."
      },
      {
        "question": "Which high school counts Jimmy Needham as a notable alumnus?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "Jake Bates is a notable alumnus of Tomball High School; he is a placekicker for the Detroit Lions.",
    "fact": True,
    "weight": 2,
    "qas": [
      {
        "question": "Which Tomball High School alumnus is a placekicker for the Detroit Lions?",
        "answer": "Jake Bates."
      },
      {
        "question": "Which high school counts Jake Bates as a notable alumnus?",
        "answer": "Tomball High School."
      }
    ]
  },
  {
    "content": "The Wikipedia entry for \"Tomball High School\" includes a note in the \"History\" section stating that the section needs additional citations for verification.",
    "fact": False,
    "weight": 0,
    "qas": []
  }
]
        return ['']+[f'```json\n{json.dumps(solution_str, ensure_ascii=False, indent="  ")}\n```'], [{"document": "", "extra_info": {"domain": "test"}}]+[{"extra_info": {"domain": "shit"}, "document": "Coordinates : 30°6′40″N 95°37′43″W  /  30.11111°N 95.62861°W  / 30.11111; -95.62861 From Wikipedia, the free encyclopedia American public high school Tomball High School Location 30330 Quinn Road Tomball , Harris County , Texas 77375 United States Coordinates 30°6′40″N 95°37′43″W  /  30.11111°N 95.62861°W  / 30.11111; -95.62861 Information Type Public high school Motto The Pride of Texas Established 1908 (Current building: 1974) School district Tomball Independent School District Principal Kimberle McKinney Staff 162.88 (FTE) [ 1 ] Grades 9-12 Enrollment 3,078 (2023–2024) [ 1 ] Student to teacher ratio 18.90 [ 1 ] Colors Red & White Athletics conference University Interscholastic League Mascot Cougar Information 281-357-3220 Website School website Tomball High School is an accredited public high school in the city of Tomball, Texas . It was the only high school in the Tomball Independent School District before the construction of Tomball Memorial High School in 2011. History [ edit ] This section needs additional citations for verification . Please help improve this article by adding citations to reliable sources in this section. Unsourced material may be challenged and removed. ( October 2011 ) ( Learn how and when to remove this message ) In 1908, school was first held in Tomball in the Woodmen of the World building, which was a one-room framed building that educated nearly 15 students. The community soon made plans to build a school. Land on Cherry Street was donated, and by 1910 a school was built. The two-story red brick schoolhouse, which was part of the Harris County School System, consisted of two classrooms downstairs and an auditorium upstairs. By 1916, 62 students attended the school. As of 1928, the community grew and consisted of four schools. By 1931, the district employed five teachers. By 1935, the original schoolhouse was torn down and replaced with a new buff brick building that contained grades one through eleven. Two years later, a petition was presented to the Harris County School Board requesting that Tomball have its own school district. The petition was granted, which transferred all funds and debts to the Tomball Independent School District. The school on Cherry Street had grown so much that by 1938 a second campus was needed. Therefore, the district built a new red brick high school with a gymnasium and a swimming pool, which was located in the 700 block of Main Street. In 1961, the original high school on Main Street was destroyed after an electrical fire in the school's attic engulfed the building. The Houston Fire Department responded, where the fire fighters dropped a hose into the swimming pool to extinguish the flame. School was dismissed for four days. Churches and other businesses throughout the community offered their facilities so the high school students could complete the school year. By 1974, students began attending classes at the new Tomball High School on Sandy Lane, later renamed Quinn Road. Continued growth over the decades meant there was a need for additional classrooms. In 2000, voters approved a $98.4 million bond referendum that enabled the district to renovate Tomball High School. In 2011 principal Gary Moss was scheduled to retire in spring 2011, and Greg Quinn, an assistant principal, became principal. He served from Fall 2011 to Spring 2017. Chris Scott is the current principal. [ 2 ] The high school was remodeled in 2005 and a cougar statue in front of the school was donated by an artist from Magnolia . In 2021, the school raised funds for a new cougar statue. [ 3 ] Academics [ edit ] For the 2018-2019 school year, Tomball High School received a B grade from the Texas Education Agency , with an overall score of 88 out of 100. This is notably 5 points lower than their fellow Tomball ISD high school, Tomball Memorial High School , which received a 93 out of 100. Within the three performance domains, Tomball High School received an A grade for Student Achievement (score of 92), a B grade for School Progress (score of 81), and a C grade for Closing the Gaps (score of 79). For context, Tomball Memorial High School received an A grade for Student Achievement (score of 95, 3 points higher), an A grade for Student Progress (score of 92, 11 points higher), and a B grade for Closing the Gaps (score of 89, 10 points higher). Tomball High School received one distinction designation for Academic Achievement (in Social Studies), while failing to receive the Academic Achievement in Science, Mathematics, English Language Arts/Reading, and Postsecondary Readiness, while also failing to place in the top 25% in Comparative Academic Growth and Comparative Closing the Gaps. For context, Tomball Memorial High School received Academic Achievement marks in Social Studies, Mathematics, and English Language Arts/Reading, while successfully placing in the top 25% in Comparative Academic Growth and Comparative Closing the Gaps. [ 4 ] [ 5 ] For the 2019-2020 school year, U.S. News gave Tomball High School a score of an 81.41 out of 100. The school is ranked 275th among Texas high schools, and 3,307th nationally. According to U.S. News, 43% of the school's students took at least one AP (Advanced Placement) exam in 2020, and 68% of those students passed at least one. Inversely, 32% of Tomball High School students who took at least one AP exam did not pass any of their AP exams. These marks are lower than their district's average, yet higher than their state's average. [ 6 ] Athletics [ edit ] Tomball High School is a member of Region II, District 15-6A of the University Interscholastic League . In 2013, the Tomball High School baseball team finished the season with a 36-4 record and won the Texas UIL 4A State Championship. [ 7 ] In 2024, the Tomball High School baseball team finished the season with a 41-3 record and won the Texas UIL 6A State Championship. [ 8 ] Tomball's sophomore starting pitchers, C.J. Sampson (14-0) and Karson Reeder (14-1), were named MVPs of the semifinal game and championship game, respectively. Tomball's senior catcher, Cade Arrimbade, was named Texas Gatorade Player of the Year. Sport Boys Girls Highlights Baseball X - 4A State Champs: 2013 6A State Champs: 2024 6A Div. 2 State Runner Up: 2025 5A Region 3 Champs (State Semi-Finalist): 2015 Basketball X X 4A Girls State-Semifinalist: 1994 Cross-Country X X 4A Girls State Champ: Liz Shell (1991, 1992) Football X - Playoff Appearances: [ 9 ] 1946-50,57,67,71,83-90,94,2003,10,12,18,21-25 B Regional Champs: 1946, 1948, 1949, 1950 2A Region 3 Finalist: 1971 4A State Finalist: 1984, 1985 4A Region 3 Finalist: 1988 5A Div 2 Reg 2 Area Champs: 2010 6A Div 2 Region 2 Champs (State Semi-Finalist): 2021 6A Div 2 Reg 2 Area Champs: 2022, 2023 6A Div 2 Bi-District (15/16) Champs: 2024 Golf X X B Boys State Champ: Bob White (1954), Mike Higgins (1958) Soccer X X Softball - X 5A State Finalist: 1993, 2004 5A State Semi-Finalist: 1995, 2005, 2010 Swimming & Diving X X 4A Boys State Champ: Trevor Strong, 100 Breastroke (2014) Tennis X X 5A Boys Doubles State Champ: Bridges/Walley (2004) Track & Field X X 6A Girls State Champ: Naomi “Peach” Booker, 100m Hurdles and 300m Hurdles (2025) Volleyball - X Demographics [ edit ] In the 2018-2019 school year, there were 2,054 students. 4.5% were African American, 2.1% were Asian, 33.0% were Hispanic, 0.3% were American Indian, 57.1% were White, and 3.0% were two or more races. 27.4% of students were Economically Disadvantaged, 3.9% were English Language Learners, and 8.8% received Special Education services. [ 10 ] Notable alumni [ edit ] Christopher S. Adams Jr. , author and retired United States Air Force General [ 11 ] Chad Brannon , TV actor best known for his role in General Hospital [ 12 ] Josh Breaux , baseball player in the New York Yankees organization [ 13 ] Drake Britton , baseball player for the Boston Red Sox [ 14 ] Jimmy Butler , basketball player for the Chicago Bulls , Minnesota Timberwolves , Philadelphia 76ers , Miami Heat , and Golden State Warriors . [ 15 ] Robert Dugger , MLB pitcher Mike Eli , lead singer of the Eli Young Band [ 16 ] Karlie Hay , Miss Teen USA 2016 [ 17 ] Chris Herrmann , baseball player for the Arizona Diamondbacks , Minnesota Twins [ 18 ] Jordan Leslie , football player for Cleveland Browns Jimmy Needham , contemporary Christian musician. [ 19 ] Troy Patton , baseball player for the Baltimore Orioles [ 20 ] David Phelps , Christian vocalist [ 21 ] Gary Porter , professional football player and coach Kyle Russell , baseball player [ 20 ] Drew Svoboda , college football coach [ 22 ] Jake Bates , football placekicker for Detroit Lions [ 23 ] References [ edit ] ^ a b c \"TOMBALL H S\" . National Center for Education Statistics . Retrieved December 22, 2024 . ^ \"New principal named for THS, TMHS\" . The Potpourri (Tomball Edition) at the Houston Chronicle . April 21, 2010 . Retrieved December 27, 2016 . ^ Wedding, Paul (August 27, 2021). \"Tomball High School asking community, alumni for donations for new cougar statue\" . Houston Chronicle . Retrieved February 5, 2023 . ^ Overview: Tomball H S. Texas Education Agency. Retrieved 25 December 2019. ^ Overview: Tomball Memorial H S. Texas Education Agency. Retrieved 7 November 2020. ^ Tomball High School U.S. News. Retrieved 7 November 2020 ^ Verdejo, Angel (June 8, 2013). \"Tomball captures 4A baseball title\" . Houston Chronicle . Retrieved December 8, 2016 . ^ Tate, Rob (June 8, 2024). \"Tomball defeats Pearland to capture state baseball championship\" . Retrieved June 9, 2024 . ^ \"Football State Archives\" . www.uiltexas.org . University Interscholastic League Texas. ^ Profile: Tomball H S. Texas Education Agency. Retrieved 25 December 2019. ^ Jones, Allen (November 21, 2010). \"Tomball to celebrate 100 years with Dec. 2 birthday bash\" . ASP Westward . Retrieved May 26, 2014 . ^ Puckett, Sarah (November 21, 2010). \"Tomball STAGE to present 'Le Mis' \" . ASP Westward . Retrieved May 26, 2014 . ^ \"Josh Breaux Stats, Fantasy & News\" . MiLB.com . ^ Edes, Gordon (March 20, 2013). \"Drake Britton faces up to year in jail\" . ESPN.com . Retrieved May 26, 2014 . ^ Johnson, K.C (June 25, 2011). \"Butler knows family values\" . Chicago Tribune . Retrieved May 26, 2014 . ^ Schuman, Anna (February 25, 2009). \"Tomball HS grad nominated for top country music award\" . ASP Westward . Retrieved May 26, 2014 . ^ \"Miss Texas Teen USA 2016\" . missuniverse.com . Miss Universe Organization . Retrieved July 31, 2016 . ^ Berardino, Mike (April 7, 2014). \"Twinsights: Chris Herrmann explains his growing versatility (with video)\" . Pioneer-Press . Retrieved May 26, 2014 . ^ Matthews, Kent (January 13, 2011). \"Music: Jimmy Needham looks ahead and way back\" . Houston Chronicle . Retrieved May 26, 2014 . ^ a b Marrion, Jack (July 11, 2011). \"Tomball product Russell having All-Star season with Chattanooga\" . ASP Westward . Retrieved May 26, 2014 . ^ Giri, Nisha (November 21, 2010). \"Gospel Music Assoc[i]ation reveals nominations for Tomball graduate\" . ASP Westward . Retrieved May 26, 2014 . ^ \"Alumni Spotlight: Drew Svoboda (THS, '95)\" . tomballisd.net . March 9, 2023 . Retrieved December 9, 2024 . ^ \"Alumni Spotlight: Jake Bates (THS, '17)\" . www.tomballisd.net . February 7, 2025 . Retrieved September 15, 2025 . External links [ edit ] Texas portal Schools portal Tomball High School v t e Harris County, Texas schools Public high schools Aldine ISD Aldine High Aldine Ninth Grade Carver Benjamin O. Davis Eisenhower MacArthur Nimitz Victory Early College (Houston) Alief ISD Elsik Hastings Taylor Kerr (magnet) Early College (magnet) Clear Creek ISD Clear Brook Clear Lake Clear View Clear Horizons Early College Other schools are in Galveston County Cy-Fair ISD See template Galena Park ISD Galena Park High North Shore Goose Creek CISD Goose Creek Memorial Robert E. Lee Ross S. Sterling Impact Early College High School Houston ISD See template Humble ISD Atascocita Humble High Kingwood Kingwood Park Summer Creek Quest Early College (magnet) Katy ISD James E. Taylor Mayde Creek HS Morton Ranch HS Paetow Other schools are in Fort Bend County Klein ISD Klein High Klein Cain Klein Collins Klein Forest Klein Oak Vistas High School Program Pasadena ISD J. Frank Dobie Pasadena High Pasadena Memorial Sam Rayburn South Houston Dr. Kirk Lewis Career and Technical High School Spring ISD Andy Dekaney Spring High Westfield Carl Wunsche Sr. Early College Academy at Southridge Spring Branch ISD Memorial Northbrook Spring Woods Stratford Westchester Academy for Int'l Studies Tomball ISD Tomball High Tomball Memorial Other school districts Channelview ISD ( Channelview High ) Crosby ISD ( Crosby High ) Deer Park ISD ( Deer Park High ) Huffman ISD ( Hargrave ) La Porte ISD ( La Porte High ) Sheldon ISD ( C.E. King ) Waller ISD ( Waller High ) Charter schools Amigos Por Vida Friends For Life Aristoi Classical Academy Houston Heights KIPP (multiple campuses) Harmony Public Schools (multiple campuses) KIPP Texas (multiple campuses) Pro-Vision George I. Sanchez YES Prep (multiple campuses) Northline Yzaguirre School Former public schools Benji's Special Ed Acad Girls & Boys Prep Gulf Shores Acad Galaviz Charter Galena Park ISD Fidelity Manor HS North Forest ISD Forest Brook HS Elmore HS Smiley University of Houston Charter School Victory Preparatory Academy Independent schools Secular private high schools Awty Briarwood British Int'l Chinquapin Prep Houston Sudbury Kinkaid Mirus Monarch Post Oak Rainard River Oaks Academy School of the Woods St. John's Tenney Village Religious private high schools Al-Hadi Baytown Christian Beren Academy Christian Life Center Concordia Lutheran Covenant Prep Cristo Rey Jesuit Cypress Christian Darul Arqam North Emery/Weiner Episcopal Faith West Family Christian Frassati Catholic Grace Christian Houston Adventist Houston Christian Iman Academy Incarnate Word Academy Lutheran South Memorial Private Northland Christian Rosehill Christian Saint Agnes Saint John XXIII Second Baptist St. Francis Episcopal St. Pius X St. Thomas St. Thomas Episcopal Strake Jesuit Westbury Christian PK-8 only, HS closed St. Stephen's Episcopal Closed Lutheran North Mount Carmel Portions in the City of Stafford are served by the Stafford Municipal School District , which has its schools in Fort Bend County . Sections of Clear Creek ISD , Katy ISD , and Waller ISD extend into other counties; this template only lists schools in Harris County. Pearland ISD and Dayton ISD extend into Harris County but do not operate schools there. Authority control databases International ISNI VIAF National United States Geographic NCES Retrieved from \" https://en.wikipedia.org/w/index.php?title=Tomball_High_School&oldid=1324715282 \" Categories : Tomball Independent School District high schools 1908 establishments in Texas Educational institutions established in 1908 Public high schools in Texas Hidden categories: Pages using gadget WikiMiniAtlas Articles with short description Short description matches Wikidata Use American English from September 2025 All Wikipedia articles written in American English Use mdy dates from January 2025 Pages using infobox mapframe without shape links in Wikidata Pages using infobox school with a linked country Coordinates on Wikidata Articles needing additional references from October 2011 All articles needing additional references Pages using the Kartographer extension"}]

    def test_parse_fn(self): 
        batch_solution_str, _ = self.load_dataset()
        print(len(kd_rewrite_parse_solution_fn(batch_solution_str[0])))

    def test_key_function(self):
        batch_solution_str, batch_ground_truth = self.load_dataset()

        # task = KDRewriteCriticalJudge()

        # eval_inputs = [(" ".join([_["content"] for _ in kd_rewrite_parse_solution_fn(x)]), y["document"]) for x, y in zip(batch_solution_str, batch_ground_truth)]

        # async def main():
        #     results = await task.do_job(
        #         agent=UNITTEST_AGENT,
        #         batch_inputs=eval_inputs,
        #         max_concurrent_requests=512,
        #     )
        #     print(results)
        # aio.run(main())

        # task = KDRewriteAtomicityJudge()
        # task = KDRewriteIndependencyJudge()
        # task = KDRewriteFactAndWeightJudge()
        # task = KDRewriteQAQualityJudge()
        task = KDRewriteNLIJudge()
        # eval_inputs = [(_["content"], None) for _ in kd_rewrite_parse_solution_fn(batch_solution_str[0])]
        # eval_inputs = [(_, None) for _ in kd_rewrite_parse_solution_fn(batch_solution_str[0])]
        # eval_inputs = [(_, None) for _ in kd_rewrite_parse_solution_fn(batch_solution_str[0]) if len(_.get("qas"))]
        eval_inputs = [(_, batch_ground_truth[0]["document"]) for _ in kd_rewrite_parse_solution_fn(batch_solution_str[0]) if len(_.get("qas"))]


        async def main():
            results = await task.do_job(
                agent=UNITTEST_FAST_AGENT,
                batch_inputs=eval_inputs,
                max_concurrent_requests=8192,
            )
            print(results)
        aio.run(main())

    def test_compute_score(self):
        task = KDRewriteComputeScore(parse_solution_fn=kd_rewrite_parse_solution_fn, args=KD_REWRITE_DEFAULT_PARAMS)

        batch_solution_str, batch_ground_truth = self.load_dataset()

        async def main():
            # results = await task.full_refinement_judge(
            # results = await task.split_refinement_judge(
            #     [None] *
            #     len(batch_solution_str), batch_solution_str, batch_ground_truth,
            #     # eval_task=KDRewriteCriticalJudge(),
            #     # eval_task=KDRewriteAtomicityJudge(),
            #     # eval_task=KDRewriteIndependencyJudge(),
            #     # eval_task=KDRewriteNLIJudge(),
            #     # eval_task=KDRewriteFactAndWeightJudge(),
            #     eval_task=KDRewriteQAQualityJudge(),
            #     min_score=0.,
            #     max_score=1.0,
            #     eval_agent=task.medium_reason_verify_agent,
            #     score_type="bool",
            #     max_concurrent_requests=task.args['medium_reason_verify_agent']["max_concurrent_requests"],
            # )
            
            # judge_with_kill_criteria
            # judge_atomicity
            # judge_independency
            # judge_fact_and_weight
            # judge_nli

            results = await task._compute_score(
              [None] * len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
            print(results)
        aio.run(main())

class TestCLRewrite(unittest.TestCase):
    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_dataset("cl_rewrite", num=100, xml_cot=False)

        task = CLRewriteComputeScore(parse_solution_fn=cl_rewrite_parse_solution_fn, args=CL_REWRITE_DEFAULT_PARAMS)
        print(task.compute_score([None] * len(batch_solution_str), batch_solution_str, batch_ground_truth,))
        # batch_solution_str, batch_ground_truth = batch_solution_str[:2], batch_ground_truth[:2]
        # async def main():
        #     results = await task.rubric_verify(
        #       [None] * len(batch_solution_str), batch_solution_str, batch_ground_truth,
        #     )
        #     print(results)
        # aio.run(main())

class TestCLRewriteRubrics(unittest.TestCase):
    def load_dataset(self):
        solution = [
          "The response should include the section title 'Radical Christ 27 – The Decolonial King' and discuss colonization and US cultural influence as prominent, historical patterns.",
          "The response should include the section title 'Radical Christ 26 – Dying to find Paradise' and state the definition of the true, accessible paradise.",
          "The response should include the section title 'Radical Christ 25 – Nicodemus and the Great Reset', identifying 'The Great Reset' (WEF 2020) and connecting it to Anne Lamott’s concept of 'unplugging' to fix things.",
          "The response should describe Nicodemus' identity (including the Greek etymology of his name) and explain that his nighttime appearances symbolize the inner, spiritual nature of the required transformation.",
          "The response should define 'born from above' as a radical worldview transformation that often arises from dark moments or defeat, distinguishing it from spiritual superiority.",
          "The response should mention Nicodemus' extravagant offering of 100 pounds of spices, interpreting this act as a gesture of love that echoes Mary’s perfume offering.",
          "The response should include the section title 'Radical Christ 24 – Wrecking his Father’s House', using the Port Elizabeth illustration to introduce the Temple's commercial context and Jesus' angry response.",
          "The response should explain Jesus' 'tear down this temple' statement as referring to his body, and cite John 14 or 15 to conclude that God’s home is an inclusive, non-commercial place of 'multiple abidings'."
        ]
        return [f'<think>\nTHINK\n</think>\n\n```json\n{json.dumps(solution, ensure_ascii=False, indent="  ")}\n```'], [{"input": 'Enduring through horror requires clear\xa0meaning\n\n\nFacing and embracing our\xa0Demons\n\n\n\n\n\n\n\n\nMilarepa’s story feels much closer to the truth.\n\n\n\nEmbracing our deepest fears is often our salvation.\n\nThe Seduction of\xa0Unconsciousness\n\nPhotograph: Fayaz Aziz/Reuters\n\nLiving consciously is difficult and requires that we invest time and energy in the practice, and maintenance, of a clear mind.\nBecause awareness is costly, we all to easily lapse into unconsciousness.\n\nDuring the years of struggle against Apartheid one of the biggest challenges was to get our most oppressed citizens to believe that they actually were oppressed.\nOne constantly heard the protest from poor people of colour that the White Nationalist government was good, and looking after them, better than they could do so themselves.\nConscientization, was an important step in energising the majority of South Africans to say No to Apartheid.\n\nThe sudden resurgence of Taliban dominance in Afghanistan in recent weeks has shocked us all. How could this movement with its dreadful history of human-rights abuses, particularly against women, have surged into the power-void created by the withdrawal of American troops?\nI don’t pretend to know the whole dynamic of this complicated region of the world, and I distrust anyone who claims to. Single-factor explanations are at best simplistic and at worst just arrogant.\n\nI do wonder however, if part of the problem is that we in the West, while correctly addressing the suffering of the poor and vulnerable, don’t fall into the trap of assuming that our values and lifestyle are the generic panacea for all human ills?\n\nAfghanistan’s flip suggests that at least some part of the population agrees with the Taliban, or is the memory of the Red Toyota bakkies and the public beatings so strong that like a battered woman in an abusive marriage, Afghanis are acquiescing to power simply because to oppose it would bring more abuse and suffering? I don’t know.\n\nCould it even be possible that not everyone on the planet wants to live in a free-market democracy where gender equality, human and gay rights are the norm?\nListening last week, to some Americans interviewed at a Trump rally inside the US, it seems that even in the land of the brave and the free, libertarian values are not embraced by everyone. One person actually warned the Biden administration that as quickly as the Taliban resurged in Afghanistan, so the American Right could overthrow the US Government!\n\nConscious living, understanding more than one’s own point of view, embracing and championing those not of your “in group” or echo chamber; are all challenges demanding careful thought and skilful action.\nRational and free thinking is exhausting.\nPerhaps, more demanding than our overworked, locked down, narcissistic self-interest can afford?\n\nWhy not give in to fundamentalism and let others tell us what to think and how to live?\nDo we, like the numbed and exhausted Afhanis, allow the anti-intellectual “Talibans” in our own church communities, clubs and social media platforms to overrun our free thinking and return us to the oppression of that unquestioning hive-mind where free thought and free speech are treated with the same hostility as an educated, young, Afghani woman?\n\nThe future of civilization hangs by the slender thread of those willing to keep thinking even when their brains hurt!\n\nRadical Christ 28 – Good Friday Recipe: Roast leg of\xa0Scapegoat\n\n\n\n\n\n\n\n\nBut why animal sacrifice in the first place? \xa0Pourquoi?\n\n\n\n\n\n\n\n\n\n\nWhy in God’s name do that?\n\n\n<REDACTED PART>... ... </REDACTED PART>\n\n\nThat’s ultimate hospitality. Welcome home everyone!\nNo exclusive identity, nor tongue testing language skills required!\n\nRadical Christ 23 – The Transfiguration Pivot\xa0Point\n\nThe Transfiguration of Jesus is the turning point of Jesus’ ministry.\nFrom preaching, teaching and healing in Galilee, after the Transfiguration he heads south to Jerusalem to confront the oppressive and exploitative religious establishment there.\nThe transfiguring light of Mt Tabor will progressively dim to the darkness of a cold tomb where Jesus’ dead body will be sealed after the crucifixion.\nThe disciples want to stay on the mountain in the glow. Who wouldn’t? But Jesus insists they must go and confront the darkness.\n\n', "completion": 'The Radical Christ 27 – The Decolonial\xa0King\n\nColonization\xa0has been a hot topic for a while now. \xa0\n\n\n\n\n\n\n\n\nThe USA has been amazingly successful in propagating its culture globally.\xa0\n\n\n\n\n\nAnd none of this is new. \xa0\n\n\n\n\n\n\n\n\nRadical Christ 26 – Dying to find\xa0Paradise\n\nThat’s the true, accessible paradise.\n\nRadical Christ 25 – Nicodemus and the Great\xa0Reset\n\nThe Great Reset, was the theme of the World Economic Forum in January 2020.\nAt this meeting IMF director Kristalina Georgieva, listed three key aspects of a sustainable response to the current global crisis: green growth, smarter growth, and fairer growth. With the gap between rich and poor continuing to widen everywhere, the world could do with a reset.\n\nAnne Lamott – an American novelist, Christian, recovering alcoholic, single parent and political activist known as “The People’s Author”, has said, “Almost everything will work again if you unplug it for a few minutes, including you.”\nNicodemus would have agreed with the Great Reset, and with Anne Lamott.\n\nNicodemus who? You ask…\nAn interesting bible character, Nicodemus only appears in John’s gospel where he is introduced as a Pharisee and Jewish ruler. Nicodemus makes three appearances, all of them at night.\nHe comes to ask Jesus about a new way of living in John 3.\n\nHe advocates for Jesus with his fellow Pharisees, asking he be given a fair trial in chapter 7,\n\nand finally appears in John 19 when he joins Joseph of Arimathea in preparing the body of Jesus for burial and donating 100 pounds of myrrh and aloes for the process. The usual amount of spice required was 5 pounds, so the excessively generous gesture of twenty times the norm, demonstrates the love of Nicodemus, and echoes the extravagant love of Mary who “wasted” a pound of rare perfume to anoint the feet of Jesus in John 12.\n\nNicodemus’ Greek name means “victory to the common people” from nike = victory and demos = common people. In New Testament Greek usage demos is associated with the rabble. As in “democracy”?\nSo Nick, the representative of the victorious rabble, is recorded by John’s gospel, consulting Jesus about the Greatest Reset ever, the switching off and on of our very identities.\n\nJohn 3\n\nIn a discussion completely misunderstood by many Christians, Nicodemus, whose nighttime appearances symbolise the shadowy internal nature of The Greatest Reset; is informed by Jesus that no one can see the reign of God unless that person is “born from above”.\n\nJesus makes clear the imperative. A complete transformation of the way we look at the world, and behave in it, is required. Our natural survival drives and selfish interests, our power plays and perpetual domination of those weaker or poorer than ourselves, have to be reset in a radical transformation that is every bit as dramatic as being born afresh.\n\nThis is not some arrogant rank of spiritual superiority that “Born Again” has come to mean for some fundamentalists. It is rather a deep inner change process that happens in the depth of our beings, most often in the dark despairing nights of our souls.\nWe are ready for this rebirth when we know the taste of defeat and failure, and can advocate and attend to the crucified and broken ones as Nicodemus did for Jesus.\n\nSo don’t tell me you’re Born Again, show us all by the sacrificial service of your life.\n\nRadical Christ 24 – Wrecking his Father’s House (John\xa02:13-25)\n\nThey say, there’s no place like home. If only we knew what our home, Port Elizabeth was called! Once the Mayor and the Ministers have agreed, maybe the Mlungu’s will learn to pronounce Gqeberha?\n\nIt’s clearly karma for making second language English speakers struggle for decades with Fort Beaufort!\nYet, despite the name, there’s no place like home, as young lockdowners from all over the planet discovered as they returned to their parental nests.\n\nIf you are a Christian in the Anglo-Catholic tradition, this Sunday you will hear the gospel reading of how Jesus went to his Father’s House and wrecked the place!\nJohn 2:12-22 records that Jesus went to the temple to celebrate the passover, but found a lucrative business in operation. Jews from all over the known world dreamt of Passover, “next year in Jerusalem”, and like Muslims to the Haj, or Christians to the Camino de Santiago, they pilgrimaged to Jerusalem.\n\nThe business of the temple had two profit centres – foreign exchange, to convert the pilgrims’ currency into Tyrian shekels (the only currency permitted in the temple, because it had no engraved image on it), and the second business arm – the sale of bulls, sheep and pigeons (for worshippers on a tight budget).\nIt is estimated that in Jesus’ day the entire economy of Jerusalem was based on the sacrificial throughput of the temple.\n\nFinding forex stalls set up in the only space where non-Jews (Gentiles) were allowed to stand during rituals so enraged the inclusive Jesus, he chased the livestock out of the court and upended the workstations of the money changers.\n“Do not make my fathers house a place of business”, he shouted. “Father’s house” is the conventional translation of “Oikon tou Patros” and “place of business” translates “emporiou” (the English emporium), so “shopping mall”, would a be a fair translation.\n\nJesus had clearly lost faith in the temple, which is why when interrogated by the Jews about his permit to do what he did, he said “Tear down this temple and I will build it up in three days”. John records how the disciples later realised Jesus meant his body and not the building, which suggests the home of God is no longer a building. The temple is wherever our heart is.\nThere’s no way to make money from that, Jesus’ priority was people not profits, so religious businessmen killed him.\n\nInterestingly the next time Jesus speaks about his parent’s house it’s in John 14, where he says “In my Father’s house there are many rooms”. Again a little digging reveals that the phrase “many rooms or monai pollai” in Greek can also read “multiple abidings”.\nSo it’s not a building, and not a business, neither an exclusive, mono-cultural, single room; but a diverse, multiversal home of tolerance and hospitality.\nJesus also says in the very next chapter 15, “meinate en emoi”, Greek for, “Stay here with me.”'}]

    def test_parse_fn(self): 
        batch_solution_str, _ = self.load_dataset()
        print(cl_rewrite_rubrics_parse_solution_fn(batch_solution_str[0]))

    def test_compute_score(self):
        task = CLRewriteRubricsComputeScore(parse_solution_fn=cl_rewrite_rubrics_parse_solution_fn, args=CL_REWRITE_RUBRICS_DEFAULT_PARAMS)

        # task = CLRewriteRubricsOverlapJudge()
        # task = CLRewriteSingleRubricJudge()
        # task = CLRewriteSingleRubricRecallJudge()
        # task = CLRewriteSingleRubricPrecisionJudge()
        # batch_solution_str, batch_ground_truth = self.load_dataset()

        # eval_inputs = [(json.dumps(cl_rewrite_rubrics_parse_solution_fn(batch_solution_str[0]), ensure_ascii=False, indent="  "), batch_ground_truth[0])]
        # eval_inputs = [(_, None) for _ in cl_rewrite_rubrics_parse_solution_fn(batch_solution_str[0])]
        # eval_inputs = [(_, batch_ground_truth[0]["completion"]) for _ in cl_rewrite_rubrics_parse_solution_fn(batch_solution_str[0])]
        # eval_inputs = [(_, batch_ground_truth[0]["input"]) for _ in cl_rewrite_rubrics_parse_solution_fn(batch_solution_str[0])]


        # async def main():
        #     results = await task.do_job(
        #         # agent=UNITTEST_FAST_AGENT,
        #         agent=UNITTEST_AGENT,
        #         batch_inputs=eval_inputs,
        #         max_concurrent_requests=16,
        #     )
        #     print(results)
        # aio.run(main())

        batch_solution_str, batch_ground_truth = self.load_dataset()
        async def main():
            results = await task._compute_score(
              [None] * len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
            print(results)
        aio.run(main())

if __name__ == '__main__':
    pass
