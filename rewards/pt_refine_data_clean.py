import os
import re
import json
import string
import random
import unittest
from tqdm import tqdm
import pandas as pd
from pt_refine import (
    contain_chinese,
    pretrain_postprocess,
    parse_doc_wo_notes,
    parse_doc_w_notes,
    parse_solution_fn,
    parse_doc_wo_notes_and_tags,
    MainBodyRecall,
    LengthDiffPenalty,
    NotesFormatReward,
    NotesRepetitionPenalty,
    LanguageConsistencyReward,
    get_notes_and_conclusions,
    get_notes
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


def random_generate_doc():
    all_characters = string.ascii_letters + string.digits + " "
    n = 8
    doc_size = 500
    doc = []
    for i in range(doc_size):
        doc.append(''.join(random.choice(all_characters) for _ in range(n)))
    return " ".join(doc)


def main(input_filename, output_filename, index, total):
    recall = MainBodyRecall(
        postprocess_solution_fn=parse_doc_wo_notes_and_tags)
    len_diff_penalty = LengthDiffPenalty(
        postprocess_solution_fn=parse_doc_wo_notes)
    format_reward = NotesFormatReward(
        postprocess_solution_fn=parse_doc_w_notes)
    rep_penalty = NotesRepetitionPenalty(
        postprocess_solution_fn=parse_doc_w_notes)
    lang_consist = LanguageConsistencyReward(
        postprocess_solution_fn=parse_solution_fn)

    with open(f'{output_filename}_{index}', "wt") as g:
        for filename in tqdm(os.listdir(input_filename)):
            file_num = int(filename.split(".")[0])
            # if file_num % total != int(index):
            #     continue

            filename = os.path.join(input_filename, filename)
            candidates = []
            with open(filename, "rt") as f:
                for line in f:
                    example = json.loads(line)
                    parsed = parse_solution_fn(example["output"])
                    if parsed is None:
                        continue
                    _, document = parsed
                    gt = example["prompt"]
                    if "# Hint: " in gt:
                        gt = gt[gt.index(
                            "[RAW DOCUMENT]")+len("[RAW DOCUMENT]"):gt.index("# Hint: ")].strip()
                    elif "# 提示" in gt:
                        gt = gt[gt.index(
                            "[RAW DOCUMENT]")+len("[RAW DOCUMENT]"):gt.index("# 提示")].strip()
                    if "[RAW DOCUMENT]" in gt:
                        gt = gt[gt.index("[RAW DOCUMENT]") +
                                len("[RAW DOCUMENT]"):].strip()
                    output = {}
                    output["lang_code"] = "zh" if contain_chinese(gt) else "en"

                    lang_consist_score = lang_consist.get_penalty_or_reward(
                        example["output"], {"ground_truth": gt})

                    if output["lang_code"] == "zh" and lang_consist_score < 0.8:
                        continue
                    if output["lang_code"] == "en" and lang_consist_score < 0.4:
                        continue

                    notes = get_notes(example["output"])
                    note_contents = re.findall(
                        r'\[EXPLANATION\](.*?)\[/EXPLANATION\]', document, re.DOTALL)
                    prohibit_kw = (
                        "[EXPLANATION]", "[/EXPLANATION]", "[CONCLUSION]", "[/CONCLUSION]"
                    )
                    if any(any(kw in _.upper() for kw in prohibit_kw) for _ in note_contents):
                        continue

                    notes_w_coclusions = get_notes_and_conclusions(
                        example["output"])

                    notes_w_coclusions = [
                        _ for _ in notes_w_coclusions if "...[原文内容].." not in _]

                    if len(notes) != len(notes_w_coclusions) or len(notes_w_coclusions) <= 1:
                        continue

                    rep_penalty_score = rep_penalty.get_penalty_or_reward(
                        example["output"], {"ground_truth": gt})
                    if rep_penalty_score < -0.1:
                        continue

                    # 检测参考文献是否被删除
                    if "bibliography" in document or "# References" in document or "# 参考文献" in document:
                        continue
                    if len_diff_penalty.get_penalty_or_reward(
                            example["output"], {"ground_truth": gt}) < -0.08:
                        continue

                    flag = False
                    if all(("Question:" in _) and ("Think Step by Step") for _ in notes) or all(("提问：" in _) and ("一步步思考") for _ in notes):
                        flag = True
                    if not flag:
                        continue

                    notes = get_notes_and_conclusions(example["output"])

                    if len(notes) == 0:
                        continue
                    if output["lang_code"] == "zh":
                        if not all(("提问" in _ and "一步步思考" in _) for _ in notes):
                            continue
                    else:
                        if not all(("Question" in _ and "Think Step by Step" in _) for _ in notes):
                            continue

                    text_recall = recall.get_penalty_or_reward(
                        example["output"], {"ground_truth": gt})
                    if text_recall < 0.80:
                        continue

                    output["response"] = example["output"]
                    prompt = example["prompt"]

                    prompt = prompt[prompt.index(
                        "<s><|im_start|>user")+len("<s><|im_start|>user"):prompt.index("<|im_end|>\n<|im_start|>assistant")].strip()
                    if "# 提示：" in prompt:
                        prompt = prompt[:prompt.index(
                            "# 提示：")].strip()
                    elif "# Hint: " in prompt:
                        prompt = prompt[:prompt.index("# Hint: ")].strip()
                    prompt = prompt.replace("现在请开始你的任务，对下面的数据增加注解，注意不要改变原文的内容，你需要先一步步思考，把你的思考过程放在<chain-of-thought> </chain-of-thought>里面，再输出**增加注解后**的语料在<doc> </doc>部分。注意回答语言和文档语言保持一致。\nNow, please start your task, add annotations to the following data. Pay attention not to change the original content. You need to think step by step first, put your thinking process within <chain-of-thought> </chain-of-thought>, then output the annotated corpus in the <doc> </doc> part. Pay attention to keeping the answering language consistent with the document language.", "")
                    output["prompt"] = prompt
                    candidates.append((output, len(notes_w_coclusions)))
            if len(candidates) > 0:
                candidates = sorted(candidates, key=lambda x: x[1])
                chosen = candidates[-1]
                print(f'{chosen[0]["lang_code"]} notes num={chosen[1]}')
                g.write(
                    f'{json.dumps(chosen[0], ensure_ascii=False)}\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='简单位置参数示例')
    parser.add_argument("--input", "-i", help='请输入名称')
    parser.add_argument("--output", "-o", help='请输入名称')
    parser.add_argument("--index", "-n", help='请输入名称')
    parser.add_argument("--total", "-t", default=20)

    args = parser.parse_args()

    main(args.input, args.output, args.index, args.total)
