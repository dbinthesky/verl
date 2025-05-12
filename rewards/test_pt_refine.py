import os
import re
import json
import string
import random
import unittest
from tqdm import tqdm
import pandas as pd
import asyncio as aio
from pt_refine import (
    contain_chinese,
    pretrain_postprocess,
    parse_doc_wo_notes,
    parse_doc_w_notes,
    parse_doc_wo_notes_and_tags,
    parse_solution_fn,
    parse_solution_fn,
    MainBodyRecall,
    LengthDiffPenalty,
    NotesFormatReward,
    NotesDocumentRepetitionPenalty,
    NotesIntraRepetitionReward,
    NotesDispersionReward,
    LanguageConsistencyReward,
    QwQLongCoTPretrainRefineComputeScore,
    CoTEnhanceComputeScore,
    qwq_longcot_pretrain_refine_compute_score_valid,
    qwq_longcot_pretrain_refine_compute_score_train,
)


def batchify(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def random_generate_doc(doc_size=500):
    all_characters = string.ascii_letters + string.digits + " "
    n = 8
    doc = []
    for i in range(doc_size):
        doc.append(''.join(random.choice(all_characters) for _ in range(n)))
    return " ".join(doc)


def load_pretrain_refinement(num=100):
    filename = "/cpfs01/shared/llm_ddd/tongjian/verl/rewards/pt_refine.json"
    batch_solution_str, batch_ground_truth = [], []

    with open(filename, "rt") as f:
        data = json.load(f)
    batch_solution_str, batch_ground_truth = data["batch_solution_str"], data["batch_ground_truth"]

    def tag_modify(s):
        output = s.replace("<chain-of-thought>",
                           "<think>").replace("</chain-of-thought>", "</think>")
        output = output.replace("[Note]", "[EXPLANATION]").replace(
            "[/Note]", f"[/EXPLANATION]\n\n[CONCLUSION]{random_generate_doc(10)}[/CONCLUSION]")
        return output
    batch_solution_str = [tag_modify(_) for _ in batch_solution_str]
    for _ in batch_ground_truth:
        _["lang_code"] = "zh" if contain_chinese(_["ground_truth"]) else "en"

    return batch_solution_str, batch_ground_truth


def load_pretrain_cot_enhance():
    def get_notes_and_conclusions(s: str):
        try:
            notes = re.findall(
                r'\[EXPLANATION\].*?\[/EXPLANATION\]\n*\[CONCLUSION\].*?\[/CONCLUSION\]', s, re.DOTALL)
            return notes
        except Exception as err:
            return []

    def get_conclusion(s):
        return re.findall(r'\[CONCLUSION\](.*?)\[/CONCLUSION\]', s, re.DOTALL)[0].strip()

    def get_question(s):
        if "提问：" in s and "一步步思考：" in s:
            return s[s.index("提问：")+len("提问："):s.index("一步步思考：")].strip()
        if "Question:" in s and "Think Step by Step:" in s:
            return s[s.index("Question:")+len("提问："):s.index("Think Step by Step:")].strip()
        return None

    input_data = []
    with open("/cpfs01/shared/llm_ddd/tongjian/pretrain/pretrain_refine_e2e_cot_enhance_train.jsonl", "rt") as f:
        for line in f:
            example = json.loads(line)
            input_data.append(example)

    random.shuffle(input_data)
    batch_solution_str = []
    batch_ground_truth = []

    TEMPLATE = """
    任务说明: 
下面是一篇文档，文档中包含一些注释，注释的格式为“[EXPLANATION]***[/EXPLANATION][CONCLUSION]***[/CONCLUSION]”
其中[EXPLANATION]***[/EXPLANATION]内的内容以“自问自答”的形式进行组织
- 对于中文文档，格式为“提问：*** 一步步思考：***”
- 对于英文文档，格式为“Question: *** Think Step by Step: ***”

现在我需要你帮我把思考过程（“一步步思考：”之后的内容，或者“Think Step by Step: ”）进行改进、完善，步骤清晰、逻辑严密。

优秀的思考过程应该满足下面的条件
### **一、逻辑基础**  
1. **概念清晰**：明确问题边界、核心定义及已知条件，避免模糊假设。  
2. **推理严密**：论证符合逻辑规则（归纳/演绎/类比），结论可被前提支撑，无明显逻辑漏洞。  


### **二、分析深度**  
3. **本质挖掘**：超越表面现象，追溯问题根源（如底层机制、核心矛盾）。  
4. **抽象具象**：能在具体案例与抽象规律间双向转化（归纳共性/演绎落地）。  


### **三、思维广度**  
5. **多维视角**：纳入不同立场（用户/竞品/执行者）、学科（技术/经济/心理）、时间（短期/长期）维度。  
6. **要素关联**：识别关键变量及其因果关系、互动机制，避免孤立分析。  


### **四、创新突破**  
7. **质疑假设**：挑战固有认知或行业惯例，避免惯性思维。  
8. **方案拓展**：生成多种备选方案（最优/次优/风险解），接纳非传统路径。  


### **五、自我校准**  
9. **证据依赖**：基于数据/事实/逻辑判断，主动验证反例，拒绝主观臆断。  
10. **动态调整**：随信息更新（数据/反馈/环境）及时修正结论，承认局限性。  


[RAW DOCUMENT] 
```
{document}
```





其中注释部分我帮你摘取出来
[NOTE]
```
{note}
```

下面请你完善注释的思考步骤。格式和[NOTE]部分的格式完全一致，注意，问题和结论部分不要做修改，需要一字不改地完整保留。
[OUTPUT]

"""

    for example in input_data:
        prompt = example["self_improvement"]["prompt"]
        prompt = prompt[prompt.index("[NOTE]"):]
        raw_notes = get_notes_and_conclusions(prompt)

        normalized_notes = {}
        for note in raw_notes:
            question = get_question(note)
            conclusion = get_conclusion(note)
            if question is None or conclusion is None:
                continue
            normalized_notes[(question, conclusion)] = note
        if len(raw_notes) != len(normalized_notes) or len(raw_notes) == 0:
            continue

        chosen = random.choice(example["self_improvement"]["responses"])[
            "response"]
        prompt = example["self_improvement"]["prompt"]
        raw_document = re.findall(
            r'\[RAW DOCUMENT\] \n```(.*?)```', prompt, re.DOTALL)[0].strip()
        raw_notes = get_notes_and_conclusions(
            prompt[prompt.index("其中注释部分我帮你摘取出来\n[NOTE]"):])

        batch_solution_str.append(
            chosen
        )
        batch_ground_truth.append(
            {
                "notes": raw_notes,
                "judges": [TEMPLATE.format(note=note, document=raw_document) for note in raw_notes]
            }
        )
        if len(batch_ground_truth) == 512:
            break

    return batch_solution_str, batch_ground_truth


class TestPretrainRefine(unittest.TestCase):
    def test_main_body_recall(self):
        batch_solution_str, batch_ground_truth = load_pretrain_refinement(
            num=100)
        recall = MainBodyRecall(
            postprocess_solution_fn=parse_doc_wo_notes_and_tags)
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(recall.get_penalty_or_reward(
                solution_str, ground_truth))

    def test_language_consistency_reward(self):
        batch_solution_str, batch_ground_truth = load_pretrain_refinement(
            num=100)
        penalty = LanguageConsistencyReward(
            postprocess_solution_fn=parse_solution_fn)
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty.get_penalty_or_reward(
                solution_str, ground_truth))

    def test_length_diff_penalty(self):
        batch_solution_str, batch_ground_truth = load_pretrain_refinement(
            num=100)
        penalty = LengthDiffPenalty(
            postprocess_solution_fn=parse_doc_wo_notes_and_tags)
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty.get_penalty_or_reward(
                solution_str, ground_truth))

    def test_notes_format_reward(self):
        batch_solution_str, batch_ground_truth = load_pretrain_refinement(
            num=100)
        penalty = NotesFormatReward(
            postprocess_solution_fn=parse_doc_w_notes)
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty.get_penalty_or_reward(
                solution_str, ground_truth))

    def test_notes_repetition_penalty(self):
        batch_solution_str, batch_ground_truth = load_pretrain_refinement(
            num=100)
        penalty = NotesDocumentRepetitionPenalty(
            postprocess_solution_fn=parse_doc_w_notes)
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty.get_penalty_or_reward(
                solution_str, ground_truth))

    def test_notes_intra_repetition_reward(self):
        batch_solution_str, batch_ground_truth = load_pretrain_refinement(
            num=100)
        penalty = NotesIntraRepetitionReward(
            postprocess_solution_fn=parse_doc_w_notes)
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty.get_penalty_or_reward(
                solution_str, ground_truth, "en"))

    def test_notes_dispersion_penalty(self):
        batch_solution_str, batch_ground_truth = load_pretrain_refinement(
            num=100)
        penalty = NotesDispersionReward(
            postprocess_solution_fn=parse_doc_w_notes)
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty.get_penalty_or_reward(
                solution_str, ground_truth))

    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_pretrain_refinement(
            num=100)
        task = QwQLongCoTPretrainRefineComputeScore(split="valid")
        qwq_longcot_pretrain_refine_compute_score_valid(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth
        )

        print("[Finish]")

    def test_get_revise_rm_rewards(self):
        async def main():
            batch_solution_str, batch_ground_truth = load_pretrain_refinement(
                num=100)
            task = QwQLongCoTPretrainRefineComputeScore(split="valid")
            results = await task.get_single_question_judge_rm_rewards(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth
            )
            # results = await task.get_revise_rm_rewards(
            #     [None] *
            #     len(batch_solution_str), batch_solution_str, batch_ground_truth
            # )
            print(results)
            print(len(results))
        aio.run(main())


class TestCoTEnhance(unittest.TestCase):
    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_pretrain_cot_enhance()
        task = CoTEnhanceComputeScore(split="valid")
        task.compute_score(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth
        )


if __name__ == '__main__':
    unittest.main()
