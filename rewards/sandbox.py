import re
import jieba
from abc import abstractmethod
from sacremoses import MosesTokenizer, MosesDetokenizer

en_mt = MosesTokenizer(lang='en')

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# BASE
# ------------------------------------------------------------------------------------------------------------------------------------------------------


class PenaltyOrReward(object):
    @abstractmethod
    def get_penalty_or_reward(self, solution_str, ground_truth, lang_code=None):
        raise NotImplementedError


def contain_chinese(string):
    try:
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        if re.search(pattern, string):
            return True
        return False
    except Exception as err:
        return False


def postprocess_solution(solution_str):
    if "<|im_end|>" in solution_str:
        return solution_str[:solution_str.index("<|im_end|>")]
    return solution_str


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# BASE
# ------------------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 沙盒问题合成（一阶段）
# ------------------------------------------------------------------------------------------------------------------------------------------------------


def fabricate_qa_postprocess_solution_fn(solution_str: str):
    solution_str = postprocess_solution(solution_str)
    try:
        thought = re.findall(r'<think>.*</think>',
                                solution_str, re.DOTALL)[0]
    except Exception as err:
        return None
    try:
        conclusion = re.findall(r'<question>(.*)</question>',
                                solution_str, re.DOTALL)[0]
    except Exception as err:
        return None

    if ("<question>" in conclusion) or ("</question>" in conclusion):
        return None

    return conclusion.strip()


class LengthDiffPenalty(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 penalty_base=-0.1):
        self.postprocess_solution_fn = postprocess_solution_fn
        self.penalty_base = penalty_base

    def get_penalty_or_reward(self, solution_str, ground_truth):
        solution_str = self.postprocess_solution_fn(solution_str)
        if solution_str is None:
            return 0.

        gt = ground_truth["ground_truth"]
        lang_code = ground_truth["lang_code"]

        if lang_code == "en":
            gt_tokenized = en_mt.tokenize(gt.lower())
            sol_tokenized = en_mt.tokenize(solution_str.lower())
        elif lang_code == "zh":
            gt_tokenized = list(jieba.cut(gt))
            sol_tokenized = list(jieba.cut(solution_str))

        return self.penalty_base * min(abs(len(sol_tokenized)-len(gt_tokenized)) / len(gt_tokenized), 10.)


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 沙盒问题合成（一阶段）
# ------------------------------------------------------------------------------------------------------------------------------------------------------
