import re
import math
import jieba
import random
import aiohttp
import asyncio
import requests
import numpy as np
from functools import partial
from asyncio import Semaphore
from abc import abstractmethod
from typing import Dict, Callable, List
from collections import defaultdict
from tqdm import tqdm as tqdm_nonasync
from rouge_score import rouge_scorer
from sacremoses import MosesTokenizer, MosesDetokenizer

en_mt = MosesTokenizer(lang='en')


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# BASE
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# RM_URLS = [
#     "http://10.130.1.101:32436",
#     "http://10.130.1.101:32877",
#     "http://10.130.1.101:25601",
#     "http://10.130.1.101:32976",
#     "http://10.130.2.53:28863",
#     "http://10.130.2.53:33706",
#     "http://10.130.2.53:30696",
#     "http://10.130.2.53:29722"
# ]


RM_URLS = [
    "http://10.130.1.125:27356",
    "http://10.130.1.125:29254",
    "http://10.130.1.125:33221",
    "http://10.130.1.125:33121",
    "http://10.130.1.125:30667",
    "http://10.130.1.125:34695",
]


DEFAULT_PARSE_FAILURE_REWARD = -2.


class PenaltyOrReward(object):
    @abstractmethod
    def get_penalty_or_reward(self, solution_str, ground_truth, lang_code=None):
        raise NotImplementedError


def batchify(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


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


async def rm_request_with_retry(urls, data, max_retries=3, retry_delay=5, suffix="/reward"):
    retries = 0
    while retries < max_retries:
        try:
            url = random.choice(urls)
            async with aiohttp.ClientSession() as session:
                async with session.post(f'{url}{suffix}', json=data, timeout=aiohttp.ClientTimeout(total=3000)) as response:
                    response.raise_for_status()
                    return await response.json()
        except (aiohttp.ClientError, aiohttp.ClientResponseError) as e:
            print(f"请求(数据总量={len(data)})失败，错误信息: {e}，重试第 {retries + 1} 次...")
            retries += 1
            if retries < max_retries:
                await asyncio.sleep(retry_delay)
    print("达到最大重试次数，请求失败。")
    return None


async def compute_rm_score(
        urls: List[str],
        batch_solution_str,
        batch_ground_truth,
        postprocess_solution_fn,
        parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD,
        judge_prompt_key="ground_truth",
        desc=""
):
    input_datas = []
    rewards = {}

    for i, (solution_str, ground_truth) in enumerate(zip(batch_solution_str, batch_ground_truth)):
        solution_str = postprocess_solution_fn(solution_str)
        if solution_str is None:
            rewards[i] = parse_result_failure_score
            continue
        if ground_truth is None:
            rewards[i] = parse_result_failure_score
            continue

        input_data = {
            "prompt": ground_truth[judge_prompt_key], "response": solution_str, "id": i
        }
        input_datas.append(input_data)

    if len(input_datas) > 0:
        for batch in tqdm_nonasync(batchify(input_datas, n=32), desc=f'[RM{desc}][{urls}] batchify inference (batch=32)'):
            output_datas = await rm_request_with_retry(urls, batch)
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
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# BASE
# ------------------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 基于规则的数据后处理
# ------------------------------------------------------------------------------------------------------------------------------------------------------
PUNCTS = [
    ".", ",", '\\', "*", ".................", ")", "(", "/", ':', '|', '`', "-", '【', '】', "•", ";"
]

STOP_WORDS = ['--', '?', '“', '”', '》', '－－', 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually', 'after', 'afterwards', 'again',
              'against', "ain't", 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst',
              'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate',
              'appropriate', 'are', "aren't", 'around', 'as', "a's", 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'be',
              'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'believe', 'below', 'beside', 'besides',
              'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'came', 'can', 'cannot', 'cant', "can't", 'cause', 'causes', 'certain', 'certainly',
              'changes', 'clearly', "c'mon", 'co', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains',
              'corresponding', 'could', "couldn't", 'course', "c's", 'currently', 'definitely', 'described', 'despite', 'did', "didn't", 'different', 'do', 'does',
              "doesn't", 'doing', 'done', "don't", 'down', 'downwards', 'during', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely',
              'especially', 'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'far',
              'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'four', 'from', 'further', 'furthermore',
              'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'had', "hadn't", 'happens', 'hardly', 'has',
              "hasn't", 'have', "haven't", 'having', 'he', 'hello', 'help', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', "here's", 'hereupon', 'hers',
              'herself', "he's", 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', "i'd", 'ie', 'if', 'ignored', "i'll", "i'm",
              'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', "isn't",
              'it', "it'd", "it'll", 'its', "it's", 'itself', "i've", 'just', 'keep', 'keeps', 'kept', 'know', 'known', 'knows', 'last', 'lately', 'later', 'latter',
              'latterly', 'least', 'less', 'lest', 'let', "let's", 'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks', 'ltd', 'mainly', 'many', 'may',
              'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'name', 'namely', 'nd',
              'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non', 'none', 'noone',
              'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one',
              'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own',
              'particular', 'particularly', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provides', 'que', 'quite',
              'qv', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 'said', 'same',
              'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible',
              'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she', 'should', "shouldn't", 'since', 'six', 'so', 'some', 'somebody', 'somehow',
              'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub',
              'such', 'sup', 'sure', 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', 'thats', "that's", 'the', 'their', 'theirs',
              'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'theres', "there's", 'thereupon', 'these', 'they',
              "they'd", "they'll", "they're", "they've", 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout',
              'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', "t's", 'twice', 'two', 'un', 'under',
              'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'value', 'various',
              'very', 'via', 'viz', 'vs', 'want', 'wants', 'was', "wasn't", 'way', 'we', "we'd", 'welcome', 'well', "we'll", 'went', 'were', "we're", "weren't",
              "we've", 'what', 'whatever', "what's", 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', "where's", 'whereupon',
              'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', "who's", 'whose', 'why', 'will', 'willing', 'wish', 'with',
              'within', 'without', 'wonder', "won't", 'would', "wouldn't",
              'yes', 'yet', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've", 'zero', 'zt', 'ZT', 'zz', 'ZZ', '一', '一下',
              '一些', '一切', '一则', '一天', '一定', '一方面', '一旦', '一时', '一来', '一样', '一次', '一片', '一直', '一致', '一般', '一起', '一边', '一面', '万一', '上下',
              '上升', '上去', '上来', '上述', '上面', '下列', '下去', '下来', '下面', '不一', '不久', '不仅', '不会', '不但', '不光', '不单', '不变', '不只', '不可', '不同',
              '不够', '不如', '不得', '不怕', '不惟', '不成', '不拘', '不敢', '不断', '不是', '不比', '不然', '不特', '不独', '不管', '不能', '不要', '不论', '不足', '不过',
              '不问', '与', '与其', '与否', '与此同时', '专门', '且', '两者', '严格', '严重', '个', '个人', '个别', '中小', '中间', '丰富', '临', '为', '为主', '为了', '为什么',
              '为什麽', '为何', '为着', '主张', '主要', '举行', '乃', '乃至', '么', '之', '之一', '之前', '之后', '之後', '之所以', '之类', '乌乎', '乎', '乘', '也', '也好',
              '也是', '也罢', '了', '了解', '争取', '于', '于是', '于是乎', '云云', '互相', '产生', '人们', '人家', '什么', '什么样', '什麽', '今后', '今天', '今年', '今後',
              '仍然', '从', '从事', '从而', '他', '他人', '他们', '他的', '代替', '以', '以上', '以下', '以为', '以便', '以免', '以前', '以及', '以后', '以外', '以後', '以来',
              '以至', '以至于', '以致', '们', '任', '任何', '任凭', '任务', '企图', '伟大', '似乎', '似的', '但', '但是', '何', '何况', '何处', '何时', '作为', '你', '你们',
              '你的', '使得', '使用', '例如', '依', '依照', '依靠', '促进', '保持', '俺', '俺们', '倘', '倘使', '倘或', '倘然', '倘若', '假使', '假如', '假若', '做到', '像',
              '允许', '充分', '先后', '先後', '先生', '全部', '全面', '兮', '共同', '关于', '其', '其一', '其中', '其二', '其他', '其余', '其它', '其实', '其次', '具体',
              '具体地说', '具体说来', '具有', '再者', '再说', '冒', '冲', '决定', '况且', '准备', '几', '几乎', '几时', '凭', '凭借', '出去', '出来', '出现', '分别', '则', '别',
              '别的', '别说', '到', '前后', '前者', '前进', '前面', '加之', '加以', '加入', '加强', '十分', '即', '即令', '即使', '即便', '即或', '即若', '却不', '原来', '又',
              '及', '及其', '及时', '及至', '双方', '反之', '反应', '反映', '反过来', '反过来说', '取得', '受到', '变成', '另', '另一方面', '另外', '只是', '只有', '只要', '只限',
              '叫', '叫做', '召开', '叮咚', '可', '可以', '可是', '可能', '可见', '各', '各个', '各人', '各位', '各地', '各种', '各级', '各自', '合理', '同', '同一', '同时',
              '同样', '后来', '后面', '向', '向着', '吓', '吗', '否则', '吧', '吧哒', '吱', '呀', '呃', '呕', '呗', '呜', '呜呼', '呢', '周围', '呵', '呸', '呼哧', '咋',
              '和', '咚', '咦', '咱', '咱们', '咳', '哇', '哈', '哈哈', '哉', '哎', '哎呀', '哎哟', '哗', '哟', '哦', '哩', '哪', '哪个', '哪些', '哪儿', '哪天', '哪年',
              '哪怕', '哪样', '哪边', '哪里', '哼', '哼唷', '唉', '啊', '啐', '啥', '啦', '啪达', '喂', '喏', '喔唷', '嗡嗡', '嗬', '嗯', '嗳', '嘎', '嘎登', '嘘', '嘛',
              '嘻', '嘿', '因', '因为', '因此', '因而', '固然', '在', '在下', '地', '坚决', '坚持', '基本', '处理', '复杂', '多', '多少', '多数', '多次', '大力', '大多数',
              '大大', '大家', '大批', '大约', '大量', '失去', '她', '她们', '她的', '好的', '好象', '如', '如上所述', '如下', '如何', '如其', '如果', '如此', '如若', '存在',
              '宁', '宁可', '宁愿', '宁肯', '它', '它们', '它们的', '它的', '安全', '完全', '完成', '实现', '实际', '宣布', '容易', '密切', '对', '对于', '对应', '将', '少数',
              '尔后', '尚且', '尤其', '就', '就是', '就是说', '尽', '尽管', '属于', '岂但', '左右', '巨大', '巩固', '己', '已经', '帮助', '常常', '并', '并不', '并不是', '并且',
              '并没有', '广大', '广泛', '应当', '应用', '应该', '开外', '开始', '开展', '引起', '强烈', '强调', '归', '当', '当前', '当时', '当然', '当着', '形成', '彻底', '彼',
              '彼此', '往', '往往', '待', '後来', '後面', '得', '得出', '得到', '心里', '必然', '必要', '必须', '怎', '怎么', '怎么办', '怎么样', '怎样', '怎麽', '总之', '总是',
              '总的来看', '总的来说', '总的说来', '总结', '总而言之', '恰恰相反', '您', '意思', '愿意', '慢说', '成为', '我', '我们', '我的', '或', '或是', '或者', '战斗', '所',
              '所以', '所有', '所谓', '打', '扩大', '把', '抑或', '拿', '按', '按照', '换句话说', '换言之', '据', '掌握', '接着', '接著', '故', '故此', '整个', '方便', '方面',
              '旁人', '无宁', '无法', '无论', '既', '既是', '既然', '时候', '明显', '明确', '是', '是否', '是的', '显然', '显著', '普通', '普遍', '更加', '曾经', '替', '最后',
              '最大', '最好', '最後', '最近', '最高', '有', '有些', '有关', '有利', '有力', '有所', '有效', '有时', '有点', '有的', '有着', '有著', '望', '朝', '朝着', '本',
              '本着', '来', '来着', '极了', '构成', '果然', '果真', '某', '某个', '某些', '根据', '根本', '欢迎', '正在', '正如', '正常', '此', '此外', '此时', '此间',
              '毋宁', '每', '每个', '每天', '每年', '每当', '比', '比如', '比方', '比较', '毫不', '没有', '沿', '沿着', '注意', '深入', '清楚', '满足', '漫说', '焉',
              '然则', '然后', '然後', '然而', '照', '照着', '特别是', '特殊', '特点', '现代', '现在', '甚么', '甚而', '甚至', '用', '由', '由于', '由此可见', '的', '的话', '目前',
              '直到', '直接', '相似', '相信', '相反', '相同', '相对', '相对而言', '相应', '相当', '相等', '省得', '看出', '看到', '看来', '看看', '看见', '真是', '真正', '着', '着呢',
              '矣', '知道', '确定', '离', '积极', '移动', '突出', '突然', '立即', '第', '等', '等等', '管', '紧接着', '纵', '纵令', '纵使', '纵然', '练习', '组成', '经', '经常',
              '经过', '结合', '结果', '给', '绝对', '继续', '继而', '维持', '综上所述', '罢了', '考虑', '者', '而', '而且', '而况', '而外', '而已', '而是', '而言', '联系', '能',
              '能否', '能够', '腾', '自', '自个儿', '自从', '自各儿', '自家', '自己', '自身', '至', '至于', '良好', '若', '若是', '若非', '范围', '莫若', '获得', '虽', '虽则',
              '虽然', '虽说', '行为', '行动', '表明', '表示', '被', '要', '要不', '要不是', '要不然', '要么', '要是', '要求', '规定', '觉得', '认为', '认真', '认识', '让',
              '许多', '论', '设使', '设若', '该', '说明', '诸位', '谁', '谁知', '赶', '起', '起来', '起见', '趁', '趁着', '越是', '跟', '转动', '转变', '转贴', '较', '较之',
              '边', '达到', '迅速', '过', '过去', '过来', '运用', '还是', '还有', '这', '这个', '这么', '这么些', '这么样', '这么点儿', '这些', '这会儿', '这儿', '这就是说',
              '这时', '这样', '这点', '这种', '这边', '这里', '这麽', '进入', '进步', '进而', '进行', '连', '连同', '适应', '适当', '适用', '逐步', '逐渐', '通常', '通过',
              '造成', '遇到', '遭到', '避免', '那', '那个', '那么', '那么些', '那么样', '那些', '那会儿', '那儿', '那时', '那样', '那边', '那里', '那麽', '部分', '鄙人',
              '采取', '里面', '重大', '重新', '重要', '鉴于', '问题', '防止', '阿', '附近', '限制', '除', '除了', '除此之外', '除非', '随', '随着', '随著', '集中', '需要',
              '非但', '非常', '非徒', '靠', '顺', '顺着', '首先', '高兴', '是不是', '说说', '```latex']


def remove_latex_format_fn(text):
    # 移除LaTeX命令
    text = re.sub(r'\\[a-zA-Z]+(\{[^\}]+\})?', '', text)
    # 移除LaTeX环境
    text = re.sub(
        r'\\begin\{[a-zA-Z]+\}(.*?)\\end\{[a-zA-Z]+\}', r'\1', text, flags=re.DOTALL)
    # 移除LaTeX特殊字符
    text = re.sub(r'[\$#%&_{}]', '', text)
    return text


def remove_identifiers_fn(text):
    # 去除参考文献部分（thebibliography环境）
    text = re.sub(
        r'\\begin{thebibliography}{99}(.*?)\\end{thebibliography}', '', text, flags=re.DOTALL)
    # 去除 \bibliographystyle 和 \bibliography 相关行
    text = re.sub(
        r'\\bibliographystyle\{.*?\}\n\\bibliography\{.*?\}\n', '', text)

    # 去除本地电子打印件 ID
    text = re.sub(r'Local EPrints ID: \d+', '', text)
    # 去除 URI
    text = re.sub(
        r'URI: http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # 去除 DOI
    text = re.sub(r'doi:\d+\.\d+/[a-zA-Z0-9.-]+', '', text)
    # 去除 ISSN
    text = re.sub(r'ISSN: \d{4}-\d{4}', '', text)
    # 去除 PURE UUID
    text = re.sub(
        r'PURE UUID: [0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', '', text)
    # 去除 ORCID
    text = re.sub(
        r'orcid.org/[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X]', '', text)

    # 去除美国电话号码 (xxx) xxx-xxxx 或 xxx-xxx-xxxx
    text = re.sub(r'\(\d{3}\) \d{3}-\d{4}|\d{3}-\d{3}-\d{4}', '', text)

    # 去除中国电话号码 11 位数字
    text = re.sub(r'1\d{10}', '', text)

    # 去除多余的空白字符
    text = re.sub(r'\s+', ' ', text).strip()

    # 去除 HTML 标签的正则表达式
    text = re.sub(r'<[^>]*>', '', text)

    return text


def pretrain_postprocess(
        s,
        lang_code=None,
        remove_latex_format=True,
        remove_identifiers=True,
        remove_stop_words=True,
        remove_punct=True,
        return_str=True):

    if lang_code is None:
        lang_code = "zh" if contain_chinese(s) else "en"
    # LaTeX格式清理
    if remove_latex_format:
        s = remove_latex_format_fn(s)

    # 清楚标记符(URI、DOI)
    if remove_identifiers:
        s = remove_identifiers_fn(s)

    if lang_code == "en":
        tokenized_text = en_mt.tokenize(s.lower())
    elif lang_code == "zh":
        tokenized_text = list(jieba.cut(s))
    else:
        raise NotImplementedError

    if remove_stop_words:
        tokenized_text = [_ for _ in tokenized_text if _ not in STOP_WORDS]
    if remove_punct:
        tokenized_text = [_ for _ in tokenized_text if _ not in PUNCTS + [" "]]

    if return_str:
        return " ".join(tokenized_text)
    else:
        return tokenized_text


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 基于规则的数据后处理
# ------------------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 预训练数据治理
# ------------------------------------------------------------------------------------------------------------------------------------------------------

def parse_solution_fn(solution_str: str):
    solution_str = postprocess_solution(solution_str)
    try:
        thought = re.findall(r'<think>.*</think>',
                             solution_str, re.DOTALL)[0]
    except Exception as err:
        return None
    try:
        document = re.findall(r'<doc>(.*)</doc>',
                              solution_str, re.DOTALL)[0].strip()
    except Exception as err:
        return None

    if any(_ in document for _ in ("<think>", "</think>", "<doc>", "</doc>")):
        return None
    return thought, document


def parse_doc_w_notes(solution_str: str):
    result = parse_solution_fn(solution_str)
    if result is None:
        return None
    thought, document = result

    return document


def parse_doc_wo_notes(solution_str: str):
    result = parse_solution_fn(solution_str)
    if result is None:
        return None
    thought, document = result

    document = re.sub(r'\[EXPLANATION\][\s\S]*?\[/EXPLANATION\]',
                      "", document, flags=re.DOTALL)
    return document


def parse_doc_wo_notes_and_tags(solution_str: str):
    document = parse_doc_wo_notes(solution_str)
    if document is None:
        return None
    return document.replace("[CONCLUSION]", "").replace("[/CONCLUSION]", "")


def get_thought(solution_str: str):
    result = parse_solution_fn(solution_str)
    if result is None:
        return f"<FORMAT CORRUPT>"

    thought, document = result
    return thought


def get_notes(solution_str: str):
    result = parse_solution_fn(solution_str)
    if result is None:
        return []
    thought, document = result

    try:
        notes = re.findall(
            r'\[EXPLANATION\].*?\[/EXPLANATION\]', document, re.DOTALL)
        return notes
    except Exception as err:
        return []


def get_notes_and_conclusions(solution_str: str):
    result = parse_solution_fn(solution_str)
    if result is None:
        return []
    thought, document = result

    try:
        notes = re.findall(
            r'\[EXPLANATION\].*?\[/EXPLANATION\]\n*\[CONCLUSION\].*?\[/CONCLUSION\]', document, re.DOTALL)

        uniq_notes = []
        uniq_conclusions = set()
        for note in notes:
            if "一步步思考：" in note:
                uniq_key = note[:note.index("一步步思考：")].strip()
            elif "Think Step by Step" in note:
                uniq_key = note[:note.index("Think Step by Step")].strip()
            else:
                uniq_key = note
            if uniq_key not in uniq_conclusions:
                uniq_conclusions.add(uniq_key)
                uniq_notes.append(note)
        return notes
    except Exception as err:
        return []


class MainBodyRecall(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 parse_result_failure_score=0.,
                 high_range=0.85,
                 middle_range=0.6,
                 low_range_penalty=-0.1
                 ):
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2'], use_stemmer=True)
        self.parse_result_failure_score = parse_result_failure_score
        self.postprocess_solution_fn = postprocess_solution_fn

        self.high_range = high_range
        self.middle_range = middle_range
        self.low_range_penalty = low_range_penalty

    def get_penalty_or_reward(self, solution_str, ground_truth, lang_code=None):
        try:
            solution_str = self.postprocess_solution_fn(solution_str)
            if solution_str is None:
                return self.parse_result_failure_score

            gt = ground_truth["ground_truth"]
            if lang_code is None:
                if contain_chinese(gt):
                    lang_code = "zh"
                else:
                    lang_code = "en"

            gt_tokenized = pretrain_postprocess(gt, lang_code=lang_code)
            sl_tokenized = pretrain_postprocess(
                solution_str, lang_code=lang_code)

            score = self.scorer.score(gt_tokenized, sl_tokenized)

            rouge_recall = (score["rouge1"].fmeasure +
                            score["rouge2"].fmeasure) / 2.0

            # 分段函数打分
            if rouge_recall >= self.high_range:
                return 1.0
            elif rouge_recall >= self.middle_range:
                return rouge_recall
            else:
                return rouge_recall - self.low_range_penalty

        except Exception as err:
            print(f'[ERROR] {err}')
            return self.parse_result_failure_score


class LanguageConsistencyReward(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 penalty_base=0.8,
                 ):
        self.postprocess_solution_fn = postprocess_solution_fn
        self.penalty_base = penalty_base

    def get_penalty_or_reward(self, solution_str, ground_truth, lang_code=None):
        solution_str = self.postprocess_solution_fn(solution_str)
        if solution_str is None:
            return 0.

        gt = ground_truth["ground_truth"]
        if lang_code is None:
            if contain_chinese(gt):
                lang_code = "zh"
            else:
                lang_code = "en"
        thought, document = solution_str
        reward = 0.0
        if lang_code == "en":
            if not contain_chinese(thought):
                reward += self.penalty_base / 2
        else:
            if contain_chinese(thought):
                reward += self.penalty_base / 2

        explanation = re.findall(
            r'\[EXPLANATION\](.*?)\[/EXPLANATION\]', document, re.DOTALL)
        if len(explanation) != 0:
            if lang_code == "zh":
                consist = len(
                    [_ for _ in explanation if contain_chinese(_)]) / len(explanation)
            else:
                consist = len(
                    [_ for _ in explanation if not contain_chinese(_)]) / len(explanation)
            reward += consist * self.penalty_base / 2
        return reward


class LengthDiffPenalty(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 penalty_base=-0.8,
                 mode="lt"
                 ):
        self.postprocess_solution_fn = postprocess_solution_fn
        self.penalty_base = penalty_base
        self.mode = mode

    def get_penalty_or_reward(self, solution_str, ground_truth, lang_code=None):
        solution_str = self.postprocess_solution_fn(solution_str)
        if solution_str is None:
            return 0.

        gt = ground_truth["ground_truth"]
        if lang_code is None:
            if contain_chinese(gt):
                lang_code = "zh"
            else:
                lang_code = "en"

        gt_tokenized = pretrain_postprocess(
            gt, lang_code=lang_code, return_str=False)
        sl_tokenized = pretrain_postprocess(
            solution_str, lang_code=lang_code, return_str=False)

        gt_token_size = len(gt_tokenized)
        sol_token_size = len(sl_tokenized)

        if self.mode == "lt":
            return self.penalty_base * min(max((gt_token_size-sol_token_size), 0) / gt_token_size, 20.)
        elif self.mode == "gt":
            return self.penalty_base * min(max((sol_token_size-gt_token_size), 0) / gt_token_size, 20.)
        elif self.mode == "both":
            return self.penalty_base * min(abs(sol_token_size-gt_token_size) / gt_token_size, 20.)


class NotesDispersionReward(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 ):
        self.postprocess_solution_fn = postprocess_solution_fn

    def dedup_notes(self, notes_w_conclusions):
        dedup = {}
        for note in notes_w_conclusions:
            key = note[note.index(
                "[EXPLANATION]")+len("[EXPLANATION]"):note.index("[/EXPLANATION]")].strip()
            dedup[key] = note
        return list(dedup.values())

    def get_penalty_or_reward(self, solution_str, ground_truth, lang_code=None):
        solution_str = self.postprocess_solution_fn(solution_str)
        if solution_str is None:
            return 0.

        base_score = 0.0
        notes = re.findall(
            r'\[EXPLANATION\].*?\[/EXPLANATION\]\n*\[CONCLUSION\].*?\[/CONCLUSION\]', solution_str, re.DOTALL)
        locations = [
            solution_str.index(_) for _ in notes
        ]
        if len(locations) == 0:
            return -1.0
        cv = np.std(locations) / np.mean(locations)
        return cv


class NotesIntraRepetitionReward(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 ):
        self.postprocess_solution_fn = postprocess_solution_fn
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2'], use_stemmer=True)

    def get_penalty_or_reward(self, solution_str, ground_truth, lang_code=None):
        solution_str = self.postprocess_solution_fn(solution_str)
        if solution_str is None:
            return 0.

        base_score = 0.0
        notes = re.findall(
            r'\[EXPLANATION\](.*?)\[/EXPLANATION\]', solution_str, re.DOTALL)

        def extract_question(s):
            if "Think Step by Step:" in s and "Question:" in s:
                s = s[s.index("Question:") + len("Question:")
                              :s.index("Think Step by Step:")]
            if "一步步思考：" in s and "提问：" in s:
                s = s[s.index("提问：") + len("提问："):s.index("一步步思考：")]
            return s.strip()

        notes = [extract_question(_) for _ in notes]
        recalls = []
        score = 0.0
        try:
            for i, a in enumerate(notes):
                b = "\n".join([_ for j, _ in enumerate(notes) if j != i])
                if lang_code == "en":
                    a = " ".join(en_mt.tokenize(a.lower()))
                    b = " ".join(en_mt.tokenize(b.lower()))
                elif lang_code == "zh":
                    a = " ".join(list(jieba.cut(a)))
                    b = " ".join(list(jieba.cut(b)))

                rouge_recall = self.scorer.score(a, b)["rouge2"].recall
                recalls.append(rouge_recall)
            score = -min(np.mean(recalls), 1.0) * len(recalls)
        except Exception as err:
            pass
        if math.isnan(score):
            return 0.0
        return max(score, -20.)


class NotesFormatReward(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 max_reward=0.2,
                 step_reward=0.01,
                 max_steps=20,
                 max_penalty=-2.0
                 ):
        self.postprocess_solution_fn = postprocess_solution_fn
        self.max_reward = max_reward
        self.step_reward = step_reward
        self.max_steps = max_steps
        self.max_penalty = max_penalty

    def dedup_notes(self, notes_w_conclusions):
        dedup = {}
        for note in notes_w_conclusions:
            key = note[note.index(
                "[EXPLANATION]")+len("[EXPLANATION]"):note.index("[/EXPLANATION]")].strip()
            dedup[key] = note
        return list(dedup.values())

    def get_penalty_or_reward(self, solution_str, ground_truth, lang_code=None):
        solution_str = self.postprocess_solution_fn(solution_str)
        if solution_str is None:
            return 0.

        base_score = 0.0

        if lang_code is None:
            if contain_chinese(gt):
                lang_code = "zh"
            else:
                lang_code = "en"

        # [EXPLANATION][/EXPLANATION]闭合
        wo_notes = re.sub(r'\[EXPLANATION\][\s\S]*?\[/EXPLANATION\]',
                          "", solution_str, flags=re.DOTALL)
        if any(_ in wo_notes.upper() for _ in ("[EXPLANATION]", "[/EXPLANATION]")):
            base_score += self.max_penalty

        notes = re.findall(
            r'\[EXPLANATION\](.*?)\[/EXPLANATION\]', solution_str, re.DOTALL)
        prohibit_kw = (
            "[EXPLANATION]", "[/EXPLANATION]", "[CONCLUSION]", "[/CONCLUSION]"
        )
        if any(any(kw in _.upper() for kw in prohibit_kw) for _ in notes):
            base_score += self.max_penalty

        # 思考过程奖励
        try:
            loose_follow = re.findall(
                r'\[EXPLANATION\].*?\[/EXPLANATION\]\n*\[CONCLUSION\].*?\[/CONCLUSION\]', solution_str, re.DOTALL)
            if len(loose_follow) != len(notes):
                return base_score

            loose_follow = self.dedup_notes(loose_follow)

            if lang_code == "zh":
                strict_follow = [_ for _ in loose_follow if (
                    "提问：" in _ and "一步步思考：" in _)]
            else:
                strict_follow = [_ for _ in loose_follow if (
                    "Question:" in _ and "Think Step by Step:" in _)]
            score = min(len(loose_follow), self.max_steps) * self.step_reward/2 + \
                min(len(strict_follow), self.max_steps) * self.step_reward/2
            return base_score + min(score, self.max_reward)
        except Exception as err:
            return base_score


class NotesDocumentRepetitionPenalty(PenaltyOrReward):
    """ Coef建议设置多少呢？ =0.5
    """

    def __init__(self,
                 postprocess_solution_fn,
                 ):
        self.postprocess_solution_fn = postprocess_solution_fn
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge2', 'rougeL'], use_stemmer=True)

    def get_penalty_or_reward(self, solution_str, ground_truth, lang_code=None):
        solution_str = self.postprocess_solution_fn(solution_str)
        if solution_str is None:
            return 0.

        gt = ground_truth["ground_truth"]
        if lang_code is None:
            if contain_chinese(gt):
                lang_code = "zh"
            else:
                lang_code = "en"

        def normalize(s):
            s = s.replace("[EXPLANATION]", "").replace(
                "[/EXPLANATION]", "").strip()
            s = s.replace("Q:", "").replace("Think:", "").strip()
            s = s.replace("Question:", "").replace(
                "Think Step by Step:", "").strip()
            s = s.replace("提问：", "").replace("一步步思考：", "").strip()
            return s

        notes_w_conclusions = re.findall(
            r'\[EXPLANATION\](.*?)\[/EXPLANATION\]\n*\[CONCLUSION\](.*?)\[/CONCLUSION\]', solution_str, re.DOTALL)
        if len(notes_w_conclusions) == 0:
            return -1.0

        explanations = "\n".join([normalize(_[0])
                                 for _ in notes_w_conclusions])
        conclusions = "\n".join([normalize(_[1]) for _ in notes_w_conclusions])

        if lang_code == "en":
            explanation_tokens = " ".join(en_mt.tokenize(explanations.lower()))
            conclusion_tokens = " ".join(en_mt.tokenize(conclusions.lower()))
            gt_tokens = " ".join(en_mt.tokenize(gt.lower()))
        elif lang_code == "zh":
            explanation_tokens = " ".join(list(jieba.cut(explanations)))
            conclusion_tokens = " ".join(list(jieba.cut(conclusions)))
            gt_tokens = " ".join(list(jieba.cut(gt)))

        rouge_recall1 = self.scorer.score(explanation_tokens, conclusion_tokens)[
            "rouge2"].recall
        rouge_recall2 = self.scorer.score(explanation_tokens, gt_tokens)[
            "rouge2"].recall

        rouge_recall = max(rouge_recall1, rouge_recall2)
        penalty = 0.
        if rouge_recall < 0.05:
            penalty = 0.
        else:
            penalty = -rouge_recall
        return penalty


class QwQLongCoTPretrainRefineComputeScore(object):
    JUDGE_CRITERIA_WO_NOTES_ZH = """### **大模型数据治理评价标准（Criteria）**

#### 一、内容纯净度
- **违规内容彻底清除**：明确识别并彻底删除色情暗示、赌博诱导、广告营销（含链接/二维码/品牌硬广）、政治敏感（如涉政言论、敏感事件）、仇恨言论、暴力描述、医疗文档中的“包治百病”“神医”等违规内容。提供具体的关键词列表，如“赌博”、“色情”、“政治敏感”等。
- **格式噪声**：标准化格式，去除连续空格（超过2个）、多余换行符（超过1行），修正过度标点。具体示例：连续空格“  ”、多余换行符“\n\n\n”。
- **内容噪声**：删除与上下文无关的孤立短句（如“同上”“如题”“啊啊啊”等）、无意义语气词堆砌。具体示例：“同上”在某些情况下可以保留，如表格中的重复内容。
- **学习噪声**：删除ISBN、网址、论文引用文献、DOI、ISSN、ORCID等学术标识符；删除时间信息、网址等对内容理解无关的信息，清除不可恢复的多模态内容（如图片、表格）。明确哪些元数据需要删除，哪些需要保留，如时间信息在某些情况下需要保留。

#### 二、语义修复有效性
- **基础规范**：修正拼写语法错误，统一标点符号、大小写、特殊符号（如全角半角转换、火星文/颜文字过滤）。具体示例：“接受”与“接收”的区别，标点符号的全角半角转换。
- **语义优化**：结合上下文合理补全不完整句子，合并重复表意。具体示例：“由于……因此……”的结构。
- **逻辑增强**：明确指代，调整语序，补充逻辑连接词（如“因此”、“然而”、“他”等）。具体示例：常见的逻辑连接词和指代示例。
- **质量提升**：消除机翻痕迹，修复逻辑断裂，修正术语翻译错误、文化差异错误。具体示例：“翻译腔”、“文化背景差异”等。

#### 三、信息完备性
- **信息保留**：除需要删除、改写外的其他信息完整保留，特别是时间信息在某些情况下需要保留。明确哪些信息是必须保留的，哪些信息是可以删除的，提供具体的判断标准。
- **最小干预**：仅修正明确错误，不改变原文主要内容，明确哪些修改是必要的，哪些是不必要的。具体示例：拼写错误必须修正，但某些语法错误可以忽略。

#### 四、格式规范性
- **规范段落间距、表格格式**：统一段落间距（如1.5倍行距），确保表格对齐方式一致。具体示例：1.5倍行距的具体设置方法。
- **确保Markdown、代码块、LaTeX等技术格式正确**：检查并修复Markdown、代码块、LaTeX等技术格式，确保其正确无误。具体示例：列表项格式混乱、链接格式错误的具体修复方法。

#### 五、语言一致性
- **语种统一**：全文语种一致，代码注释与代码语种匹配，处理多语言文档时确保语种统一。具体示例：如何处理中英文混合的文档。
- **风格匹配**：保持与原文一致的正式度和专业术语使用，明确不同风格的具体定义和匹配方法。具体示例：正式度和专业术语的具体使用方法。

#### 六、可读性
- **文档可读性**：确保文档在治理后仍然易于阅读和理解，避免冗长复杂的句子结构，保持段落清晰。具体示例：如何避免冗长复杂的句子结构，如何保持段落清晰。

#### 七、附加要求
- **数据隐私**：确保处理过程中不泄露个人隐私信息，如姓名、地址、电话号码等。具体示例：姓名、地址、电话号码等。
- **数据合规**：确保处理后的数据符合相关法律法规和行业标准。具体示例：符合《个人信息保护法》等。
"""

    JUDGE_CRITERIA_WO_NOTES_EN = """### Criteria for Governance of Large Model Data

#### I. Content Purity
- **Thorough Removal of Illegal Content**: Clearly identify and completely delete content such as pornographic hints, gambling inducements, advertising and marketing (including links, QR codes, and hard brand advertisements), politically sensitive information (such as remarks related to politics and sensitive events), hate speech, violent descriptions, and illegal content like "curing all diseases" and "miracle doctors" in medical documents. Provide a specific list of keywords, such as "gambling", "pornography", "politically sensitive", etc.
- **Format Noise**: Standardize the format, remove consecutive spaces (more than 2), redundant line breaks (more than 1 line), and correct excessive punctuation. Specific examples: Consecutive spaces "  ", redundant line breaks "\n\n\n".
- **Content Noise**: Delete isolated short sentences irrelevant to the context (such as "the same as above", "as in the question", "ahhhh", etc.) and meaningless piles of interjections. Specific examples: "The same as above" can be retained in some cases, such as repeated content in a table.
- **Learning Noise**: Delete academic identifiers such as ISBN, website URLs, cited literature in papers, DOI, ORCID, etc.; delete information that is irrelevant to content understanding, such as time information and website URLs, and remove unrecoverable multimodal content (such as pictures and tables). Clearly define which metadata needs to be deleted and which needs to be retained. For example, time information may need to be retained in some cases.

#### II. Effectiveness of Semantic Repair
- **Basic Specification**: Correct spelling and grammar errors, unify punctuation marks, case, and special symbols (such as conversion between full-width and half-width characters, filtering of strange characters and emoticons). Specific examples: The difference between "accept" and "receive", conversion between full-width and half-width punctuation marks.
- **Semantic Optimization**: Reasonably complete incomplete sentences in combination with the context, and merge repetitive expressions. Specific examples: The structure of "due to... therefore...".
- **Logical Enhancement**: Clearly define references, adjust word order, and supplement logical connectives (such as "therefore", "however", "he", etc.). Specific examples: Common examples of logical connectives and references.
- **Quality Improvement**: Eliminate the traces of machine translation, repair logical breaks, and correct translation errors of terms and errors caused by cultural differences. Specific examples: "Translationese", "cultural background differences", etc.

#### III. Information Completeness
- **Information Retention**: Completely retain all information except for the content that needs to be deleted or rewritten. In particular, time information may need to provide specific judgment criteria on which information must be retained and which can be deleted. Specific examples: Spelling errors must be corrected, but some grammar errors can be ignored.

#### IV. Format Specification
- **Standardize Paragraph Spacing and Table Format**: Unify the paragraph spacing (such as 1.5-line spacing), and ensure consistent alignment of tables. Specific examples: The specific method for setting 1.5-line spacing.
- **Ensure the Correctness of Technical Formats such as Markdown, Code Blocks, and LaTeX**: Check and repair technical formats such as Markdown, code blocks, and LaTeX to ensure their correctness. Specific examples: Specific repair methods for chaotic list item formats and incorrect link formats.

#### V. Language Consistency
- **Language Unity**: Ensure consistent language throughout the document. The language of code comments should match the code language. When dealing with multi-language documents, ensure language unity. Specific examples: How to deal with documents that mix Chinese and English.
- **Style Matching**: Maintain the same formality and use of professional terms as the original text, and clearly define the specific definitions and matching methods of different styles. Specific examples: The specific usage methods of formality and professional terms.

#### VI. Readability
- **Document Readability**: Ensure that the document is still easy to read and understand after governance, avoid long and complex sentence structures, and keep paragraphs clear. Specific examples: How to avoid long and complex sentence structures and how to keep paragraphs clear.

#### VII. Additional Requirements
- **Data Privacy**: Ensure that personal privacy information, such as names, addresses, and phone numbers, is not leaked during the processing. Specific examples: Names, addresses, phone numbers, etc.
- **Data Compliance**: Ensure that the processed data complies with relevant laws, regulations, and industry standards. Specific examples: Comply with laws such as the Personal Information Protection Law.
"""

    JUDGE_CRITERIA_SINGLE_QUESTION_ZH = """### 高质量提问评价标准

#### 一、核心评价维度与评分细则

以下从五个维度评估提问质量，每个维度按0-4分打分（4分为最高）：

1. **相关性**
   - **0分**：问题与文档核心内容无关，涉及背景知识外的话题（如询问作者学术背景、文档格式等）。
   - **1分**：问题指向边缘细节，但对理解核心内容有辅助作用（如追问术语定义，有助于理解后续内容）。
   - **2分**：问题涉及次要逻辑环节，如询问具体公式推导步骤，但未指出其中假设漏洞。
   - **3分**：问题精准定位关键知识点或逻辑断层，并明确指出矛盾点。
   - **4分**：问题直接指向文档中理解晦涩或过于简略的部分，且明确指出未解释清楚的内容，并能引导读者进一步思考。

2. **逻辑深度**
   - **0分**：停留在事实复述或表面现象。
   - **1分**：基于单一因果关系提问，未涉及多因素关联。
   - **2分**：追问方法或过程，并能指出潜在的假设。
   - **3分**：涉及知识原理或潜在假设，并能指出假设的合理性。
   - **4分**：追问知识体系的底层逻辑或潜在风险，涉及多因素关联，并能引导读者进行系统性思考。

3. **引导性**
   - **0分**：封闭性问题，答案为“是/否”或单一事实。
   - **1分**：问题仅需单一解释。
   - **2分**：问题隐含步骤提示，并能引导读者思考。
   - **3分**：问题明确引导逻辑链条。
   - **4分**：问题构建系统性思考框架，并能引导读者进行多角度思考，且能提供具体思考路径。

4. **批判性视角**
   - **0分**：无质疑，仅请求解释或复述内容。
   - **1分**：表面质疑，未具体指出漏洞。
   - **2分**：指出方法局限性或现实矛盾，并能提出改进建议。
   - **3分**：质疑原文假设或逻辑漏洞，并能提出具体质疑点。
   - **4分**：探索替代方案或逆向思考，提供具体替代方法，并能引导读者进行深入探讨。

5. **具体性**
   - **0分**：问题过于宽泛，难以具体回答。
   - **1分**：问题有一定的具体性，但仍有较大的回答空间。
   - **2分**：问题具体明确，指向明确的内容。
   - **3分**：问题不仅具体明确，还包含背景信息，便于回答。
   - **4分**：问题具体明确，包含详细的背景信息，并能引导读者进行深入思考。

#### 二、综合评分计算方法

1. **权重分配**：五个维度权重均等，各占20%。
2. **得分计算**：总分 = （相关性得分 + 逻辑深度得分 + 引导性得分 + 批判性视角得分 + 具体性得分） × 0.2，满分4分。
   - **示例**：
     - 提问“若数据存在时空相关性，原文的平稳性假设失效，此时应如何修正模型？”
     - 相关性4分（指向假设漏洞），逻辑深度3分（涉及理论适用），引导性3分（隐含修正步骤），批判性视角4分（探索替代方案），具体性3分（具体明确且包含背景信息）。
     - 总分 = (4+3+3+4+3)×0.2 = 3.4分，属于“逻辑较深且具有批判意识的高质量提问”。

### 说明

- **相关性**：增加对问题是否能引导读者深入理解文档核心内容的评估。
- **逻辑深度**：增加对问题是否能引导读者进行多角度思考的评估。
- **引导性**：增加对问题是否能引导读者进行系统性思考的评估。
- **批判性视角**：增加对问题是否能引导读者进行逆向思考的评估。
- **具体性**：增加对问题具体性和明确性的评估，确保问题能够明确指出未解释清楚的内容。
"""

    JUDGE_CRITERIA_SINGLE_QUESTION_EN = """### High-quality Question Evaluation Criteria

#### I. Core Evaluation Dimensions and Scoring Rules

The quality of questions is evaluated from the following five dimensions, with each dimension scored from 0 to 4 points (4 points being the highest).

1. **Relevance**
   - **0 points**: The question is unrelated to the core content of the document and involves topics outside of the background knowledge (such as asking about the author's academic background, document format, etc.).
   - **1 point**: The question points to marginal details but has an auxiliary role in understanding the core content (such as asking for the definition of a term, which helps in understanding the subsequent content).
   - **2 points**: The question involves secondary logical links, such as asking for the derivation steps of a specific formula, but does not point out the loopholes in the assumptions.
   - **3 points**: The question precisely locates key knowledge points or logical discontinuities and clearly points out contradictions.
   - **4 points**: The question directly points to the parts of the document that are difficult to understand or too brief, clearly points out the content that is not explained clearly, and can guide the reader to think further.

2. **Logical Depth**
   - **0 points**: Stays at the level of fact repetition or surface phenomena.
   - **1 point**: Asks questions based on a single causal relationship without involving the correlation of multiple factors.
   - **2 points**: Asks about the method or process and can point out potential assumptions.
   - **3 points**: Involves knowledge principles or potential assumptions and can point out the rationality of the assumptions.
   - **4 points**: Asks about the underlying logic or potential risks of the knowledge system, involves the correlation of multiple factors, and can guide the reader to think systematically.

3. **Guidedness**
   - **0 points**: Closed-ended question with an answer of "yes/no" or a single fact.
   - **1 point**: The question only requires a single explanation.
   - **2 points**: The question implies step hints and can guide the reader to think.
   - **3 points**: The question clearly guides the logical chain.
   - **4 points**: The question constructs a systematic thinking framework, can guide the reader to think from multiple angles, and can provide a specific thinking path.

4. **Critical Perspective**
   - **0 points**: No questioning, only requests for explanation or repetition of content.
   - **1 point**: Superficial questioning without specifically pointing out loopholes.
   - **2 points**: Points out the limitations of the method or real-world contradictions and can propose improvement suggestions.
   - **3 points**: Questions the assumptions or logical loopholes in the original text and can put forward specific points of doubt.
   - **4 points**: Explores alternative solutions or thinks in reverse, provides specific alternative methods, and can guide the reader to conduct in-depth discussions.

5. **Specificity**
   - **0 points**: The question is too broad to be answered specifically.
   - **1 point**: The question has some specificity but still leaves a large space for answering.
   - **2 points**: The question is specific and clear, pointing to clear content.
   - **3 points**: The question is not only specific and clear but also includes background information, making it easy to answer.
   - **4 points**: The question is specific and clear, includes detailed background information, and can guide the reader to think deeply.

#### II. Comprehensive Scoring Calculation Method

1. **Weight Allocation**: The five dimensions have equal weight, each accounting for 20%.
2. **Score Calculation**: Total score = (Relevance score + Logical depth score + Guidedness score + Critical perspective score + Specificity score) × 0.2, with a full score of 4 points.
   - **Example**:
     - Question: "If there is spatiotemporal correlation in the data and the stationarity assumption in the original text fails, how should the model be revised?"
     - Relevance: 4 points (points to the assumption loophole), Logical depth: 3 points (involves the application of theory), Guidedness: 3 points (implies the steps of revision), Critical perspective: 4 points (explores alternative solutions), Specificity: 3 points (specific and clear, includes background information).
     - Total score = (4 + 3 + 3 + 4 + 3)×0.2 = 3.4 points, which belongs to "a high-quality question with relatively deep logic and critical awareness".

### Explanation

- **Relevance**: Adds an assessment of whether the question can guide the reader to deeply understand the core content of the document.
- **Logical Depth**: Adds an assessment of whether the question can guide the reader to think from multiple angles.
- **Guidedness**: Adds an assessment of whether the question can guide the reader to think systematically.
- **Critical Perspective**: Adds an assessment of whether the question can guide the reader to think in reverse.
- **Specificity**: Adds an assessment of the specificity and clarity of the question to ensure that the question can clearly point out the content that is not explained clearly.
"""

    JUDGE_CRITERIA_QUESTION_DIVERSITY_ZH = """### **问题多样性评价标准**：

### 一、问题类型多样性
1. **高阶思维类问题**：
   - **跨域整合**：结合多学科知识提出问题，例如“结合法律和经济学视角，分析知识产权保护的有效性”。
   - **批判性思考**：质疑既有结论，提出研究设计反思的问题，例如“分析文献中样本偏差对研究结果的影响”。
   - **创新性拓展**：提出假设推演与知识延伸的问题，例如“探讨重力常数变化对航天模型的影响”。

2. **应用分析类问题**：
   - **场景适配**：提出特定场景适用性分析的问题，例如“分析高并发场景下的技术选型”。
   - **风险评估**：提出潜在问题预判与局限性分析的问题，例如“分析自动驾驶技术的瓶颈”。
   - **优劣比较**：提出方案对比与适用情境区分的问题，例如“比较不同研究方法的优劣”。

3. **基础认知类问题**：
   - **原理推导**：提出逻辑链条构建的问题，例如“从麦克斯韦方程组推导电磁波方程”。
   - **方法解析**：提出技术路线分解的问题，例如“解析层次分析法权重计算”。
   - **概念阐释**：提出基础定义解读的问题，例如“解释机器学习中的过拟合”。
   - **具体应用**：针对文档中的简略或晦涩部分，提出具体问题，例如“‘本改进工艺试验为提高甘油产品质量提供了一个较好的方法’，具体改进了哪些工艺参数？”。

### 二、问题视角多样性
1. **利益相关者视角**：
   - **决策者视角**：提出政策影响与资源配置的问题，例如“碳关税对企业创新的激励”。
   - **研究者视角**：提出研究方法优化的问题，例如“统计模型选择的依据”。
   - **使用者视角**：提出用户体验与产品适配的问题，例如“APP界面设计原理”。

2. **场景维度多元性**：
   - **实践应用**：提出技术开发、商业决策等现实场景的问题，例如“芯片产业的机遇与挑战”。
   - **学术研究**：提出科研全流程的问题，例如“文献差异原因分析”。
   - **教育教学**：提出教学专属的问题，例如“试题设计思路分析”。

3. **学科覆盖广度**：
   - **交叉学科**：提出跨学科融合的问题，例如“认知心理学与多媒体教学”。
   - **人文社科**：提出法学、经济学等领域的问题，例如“历史赋税制度的经济逻辑”。
   - **自然科学**：提出基础学科与应用技术的问题，例如“柯西不等式向量形式证明”。

### 三、分析深度多样性
1. **定性与定量分析融合度**：
   - **定量建模**：提出数据统计与数学建模的问题，例如“使用Logistic回归进行风险评估”。
   - **定性描述**：提出现象归纳与特征阐释的问题，例如“分析小说中的主题思想”。

2. **正向推导与逆向思考**：
   - **正向推导**：从已知条件出发，推导出结论，例如“从益气活血法的疗效出发，推导其作用机制”。
   - **逆向思考**：从结果出发，反推原因，例如“分析自动驾驶技术的瓶颈，反推其技术难题”。

3. **单维分析与系统分析**：
   - **单维分析**：从单一角度分析问题，例如“从单个药物的作用机制出发，分析其在益气活血法中的贡献”。
   - **系统分析**：从多个角度综合分析问题，例如“分析益气活血法在不同地区和不同人群中的适用性”。

### 四、思维路径多样性
1. **归纳与演绎**：
   - **归纳**：从具体案例中归纳出一般规律，例如“从历史案例中归纳出取消协议的常见原因”。
   - **演绎**：从一般规律推导出具体结论，例如“从物理原理出发，推导引射喷管的性能”。

2. **发散与收敛**：
   - **发散**：从不同角度探讨问题，例如“从不同角度探讨取消协议的可能后果”。
   - **收敛**：将不同角度的分析综合成一个结论，例如“综合各方意见，提出最优解决方案”。

3. **线性与非线性思维**：
   - **线性思维**：按步骤顺序分析问题，例如“分析自动驾驶技术的渐进过程”。
   - **非线性思维**：从非线性角度分析问题，例如“分析自动驾驶技术的突变
"""
    JUDGE_CRITERIA_QUESTION_DIVERSITY_EN = """### Problem Diversity Evaluation Criteria:

### I. Problem Type Diversity
1. **High-order Thinking Problems**:
   - **Cross-domain Integration**: Propose questions that combine knowledge from multiple disciplines. For example, "Analyze the effectiveness of intellectual property protection from the perspectives of law and economics."
   - **Critical Thinking**: Raise questions that challenge existing conclusions and involve reflections on research designs. For example, "Analyze the impact of sample bias in the literature on research results."
   - **Innovative Expansion**: Put forward questions that involve hypothesis deduction and knowledge extension. For example, "Explore the impact of changes in the gravitational constant on aerospace models."

2. **Application and Analysis Problems**:
   - **Scenario Adaptability**: Pose questions about the analysis of applicability in specific scenarios. For example, "Analyze the technology selection in high-concurrency scenarios."
   - **Risk Assessment**: Raise questions about predicting potential problems and analyzing limitations. For example, "Analyze the bottlenecks of autonomous driving technology."
   - **Advantages and Disadvantages Comparison**: Propose questions about comparing different solutions and distinguishing applicable situations. For example, "Compare the advantages and disadvantages of different research methods."

3. **Basic Cognitive Problems**:
   - **Principle Deduction**: Put forward questions about constructing logical chains. For example, "Deduce the electromagnetic wave equation from Maxwell's equations."
   - **Method Analysis**: Raise questions about decomposing technical routes. For example, "Analyze the weight calculation in the Analytic Hierarchy Process."
   - **Concept Interpretation**: Pose questions about interpreting basic definitions. For example, "Explain overfitting in machine learning."
   - **Specific Application**: Raise specific questions regarding the concise or obscure parts in documents. For example, "In the statement 'This improved process experiment provides a good method for improving the quality of glycerol products', which process parameters are specifically improved?"

### II. Problem Perspective Diversity
1. **Stakeholder Perspectives**:
   - **Decision-maker Perspective**: Raise questions about the impact of policies and resource allocation. For example, "The incentive of carbon tariffs for corporate innovation."
   - **Researcher Perspective**: Pose questions about optimizing research methods. For example, "The basis for choosing a statistical model."
   - **User Perspective**: Raise questions about user experience and product adaptation. For example, "The design principles of an APP interface."

2. **Diversity of Scenario Dimensions**:
   - **Practical Application**: Propose questions related to real-world scenarios such as technology development and business decision-making. For example, "The opportunities and challenges in the chip industry."
   - **Academic Research**: Raise questions covering the entire process of scientific research. For example, "Analysis of the reasons for differences in literature."
   - **Education and Teaching**: Pose questions specific to teaching. For example, "Analysis of the design ideas of test questions."

3. **Breadth of Subject Coverage**:
   - **Interdisciplinary**: Raise questions that integrate multiple disciplines. For example, "Cognitive psychology and multimedia teaching."
   - **Humanities and Social Sciences**: Pose questions in fields such as law and economics. For example, "The economic logic of historical tax systems."
   - **Natural Sciences**: Raise questions in basic disciplines and applied technologies. For example, "Proof of the vector form of the Cauchy inequality."

### III. Diversity of Analysis Depth
1. **Integration of Qualitative and Quantitative Analysis**:
   - **Quantitative Modeling**: Raise questions about data statistics and mathematical modeling. For example, "Use Logistic regression for risk assessment."
   - **Qualitative Description**: Put forward questions about summarizing phenomena and explaining characteristics. For example, "Analyze the theme in a novel."

2. **Forward Deduction and Reverse Thinking**:
   - **Forward Deduction**: Starting from known conditions, derive conclusions. For example, "Starting from the curative effect of the method of replenishing qi and promoting blood circulation, deduce its mechanism of action."
   - **Reverse Thinking**: Starting from the results, infer the causes. For example, "Analyze the bottlenecks of autonomous driving technology and infer its technical difficulties."

3. **Single-dimensional Analysis and Systematic Analysis**:
   - **Single-dimensional Analysis**: Analyze problems from a single perspective. For example, "Starting from the mechanism of action of a single drug, analyze its contribution to the method of replenishing qi and promoting blood circulation."
   - **Systematic Analysis**: Analyze problems comprehensively from multiple angles. For example, "Analyze the applicability of the method of replenishing qi and promoting blood circulation in different regions and among different groups of people."

### IV. Diversity of Thinking Paths
1. **Induction and Deduction**:
   - **Induction**: Summarize general laws from specific cases. For example, "Summarize the common reasons for canceling agreements from historical cases."
   - **Deduction**: Derive specific conclusions from general laws. For example, "Starting from physical principles, deduce the performance of an ejector nozzle."

2. **Divergent and Convergent Thinking**:
   - **Divergent**: Explore problems from different angles. For example, "Explore the possible consequences of canceling agreements from different angles."
   - **Convergent**: Synthesize analyses from different angles into a conclusion. For example, "Propose the optimal solution by integrating various opinions."

3. **Linear and Non-linear Thinking**:
   - **Linear Thinking**: Analyze problems step by step. For example, "Analyze the progressive process of autonomous driving technology."
   - **Non-linear Thinking**: Analyze problems from a non-linear perspective. For example, "Analyze the sudden changes in autonomous driving technology."

"""

    def __init__(self,
                 split="train",
                 parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD):
        self.split = split
        self.parse_result_failure_score = parse_result_failure_score

        # FIXME
        self.recall = MainBodyRecall(
            postprocess_solution_fn=parse_doc_wo_notes_and_tags)
        self.len_diff = LengthDiffPenalty(
            postprocess_solution_fn=parse_doc_wo_notes_and_tags)
        self.note_format = NotesFormatReward(
            postprocess_solution_fn=parse_doc_w_notes)
        self.note_rep = NotesDocumentRepetitionPenalty(
            postprocess_solution_fn=parse_doc_w_notes)
        self.lang_consist = LanguageConsistencyReward(
            postprocess_solution_fn=parse_solution_fn)
        self.note_dispersion = NotesDispersionReward(
            postprocess_solution_fn=parse_doc_w_notes
        )
        self.note_intra_rep = NotesIntraRepetitionReward(
            postprocess_solution_fn=parse_doc_w_notes
        )

    def get_penalties(self) -> Dict[str, Callable]:
        return {
            "TextRecall": self.recall.get_penalty_or_reward,
            "LengthDiff": self.len_diff.get_penalty_or_reward,
            "NoteFormat": self.note_format.get_penalty_or_reward,
            "NoteRep": self.note_rep.get_penalty_or_reward,
            "LangConsistency": self.lang_consist.get_penalty_or_reward,
            "NoteDispersion": self.note_dispersion.get_penalty_or_reward,
            "NoteIntraRepetition": self.note_intra_rep.get_penalty_or_reward
        }

    def get_penalty_coef(self):
        return {
            "TextRecall": 1.0,
            "LengthDiff": 1.0,
            "NoteFormat": 1.0,
            "NoteRep": 0.5,
            "LangConsistency": 1.0,
            "NoteDispersion": 1.0,
            "NoteIntraRepetition": 0.00
        }

    async def get_revise_rm_rewards(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            urls=RM_URLS):
        """
            评价除去处思考过程后的改写内容
        """
        refine_judges = []

        for _ in batch_ground_truth:
            lang_code = _["lang_code"]
            if lang_code == "zh":
                judge_template = self.JUDGE_CRITERIA_WO_NOTES_ZH
            else:
                judge_template = self.JUDGE_CRITERIA_WO_NOTES_EN
            refine_judges.append({
                "ground_truth": f'你是一名专精于大模型数据改写的治理专家。目标是给定一篇从网页爬取或者PDF解析出来的文档，改写成一篇优质的大语言模型预训练语料。\n\n[Raw Corpus]\n{_["ground_truth"]}\n\n\n# 评价标准\n{judge_template}'
            })

        tasks = []
        n = len(urls)

        for i, batch in enumerate(batchify(zip(refine_judges, batch_solution_str), n=64)):
            refine_judge = [_[0] for _ in batch]
            mini_batch_solution_str = [_[1] for _ in batch]
            tasks.append(
                compute_rm_score(
                    batch_solution_str=mini_batch_solution_str,
                    batch_ground_truth=refine_judge,
                    postprocess_solution_fn=parse_doc_wo_notes_and_tags,
                    parse_result_failure_score=self.parse_result_failure_score,
                    desc="-revise",
                    urls=[urls[i % n]]
                )
            )

        results = await self.run_tasks_in_queues(tasks, n=n)

        rewards = []
        for _ in results:
            rewards.extend(_)

        return rewards

    def normalize_question(self, note):
        if "提问：" in note and "一步步思考：" in note:
            question = note[note.index("提问："):note.index(
                "一步步思考：")].strip()
        elif "Question:" in note and "Think Step by Step:" in note:
            question = note[note.index("Question:"):note.index(
                "Think Step by Step:")].strip()
        else:
            question = re.findall(
                r'\[EXPLANATION\](.*?)\[/EXPLANATION\]', note, re.DOTALL)[0].strip()
        conclusion = re.findall(
            r'\[CONCLUSION\](.*?)\[/CONCLUSION\]', note, re.DOTALL)[0].strip()
        return question.strip(), conclusion.strip()

    async def process_queue(self, queue, semaphore):
        """处理单个队列，确保队列内任务串行执行"""
        async with semaphore:  # 限制并发队列数量
            results = []
            for task in queue:
                result = await task
                results.append(result)
            return results

    async def run_tasks_in_queues(self, tasks, n):
        """将任务分成n个队列并行执行"""
        # 创建n个队列
        queues = [[] for _ in range(n)]

        # 平均分配任务到各个队列
        for i, task in enumerate(tasks):
            queues[i % n].append(task)

        # 创建信号量限制并发队列数量
        semaphore = Semaphore(n)

        # 并行处理所有队列
        queue_results = await asyncio.gather(
            *[self.process_queue(queue, semaphore) for queue in queues]
        )

        # 展平结果列表（保持原始顺序）
        flattened_results = []
        for i in range(len(tasks)):
            queue_idx = i % n
            task_idx = i // n
            flattened_results.append(queue_results[queue_idx][task_idx])

        return flattened_results

    async def get_single_question_judge_rm_rewards(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            urls=RM_URLS,
            default_penalty=-1.0,
            reward_rectifier_value=0.
    ):
        """
            评价单条提问质量
        """
        indices = []
        print(batch_solution_str)

        for i, (_gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            lang_code = _gt["lang_code"]
            if lang_code == "zh":
                judge_template = self.JUDGE_CRITERIA_SINGLE_QUESTION_ZH
            else:
                judge_template = self.JUDGE_CRITERIA_SINGLE_QUESTION_EN

            notes = get_notes(sol)
            notes_w_coclusions = get_notes_and_conclusions(sol)
            if len(notes) != len(notes_w_coclusions):
                continue
            if len(notes_w_coclusions) == 0:
                continue

            questions = [self.normalize_question(
                _) for _ in notes_w_coclusions]

            if lang_code == "zh":
                questions = [
                    f'- 原文中需要进行提问的部分： \n"{_[1]}"\n- 提问：\n"{_[0]}"' for _ in questions]
                judge_prompt = f'任务：针对文档中理解晦涩、过于简略的部分进行提问。\n\n[Raw Corpus]\n{_gt["ground_truth"]}\n\n\n# 评价标准\n{judge_template}'
            else:
                questions = [
                    f'- Identify the parts in the original text that need to be questioned.\n"{_[1]}"\n- The question raised.\n"{_[0]}"' for _ in questions]
                judge_prompt = f'Task: Ask questions about the obscure and overly brief parts in the document.\n\n[Raw Corpus]\n{_gt["ground_truth"]}\n\n\n# Judge Criteria\n{judge_template}'

            for question in questions:
                addition_judges.append({"ground_truth": judge_prompt})
                new_batch_solution_strs.append(question)
            sizes.append(len(questions))

            indices.append(i)

        tasks = []
        n = len(urls)

        for i, batch in enumerate(batchify(zip(addition_judges, new_batch_solution_strs), n=64)):
            addition_judge = [_[0] for _ in batch]
            new_batch_solution_str = [_[1] for _ in batch]
            tasks.append(
                compute_rm_score(
                    batch_solution_str=new_batch_solution_str,
                    batch_ground_truth=addition_judge,
                    postprocess_solution_fn=lambda x: x,
                    parse_result_failure_score=self.parse_result_failure_score,
                    desc="-single_question_judge",
                    urls=[urls[i % n]]
                )
            )

        results = await self.run_tasks_in_queues(tasks, n=n)

        rewards = []
        for _ in results:
            rewards.extend(_)
        rewards_group = []
        for size in sizes:
            rewards_group.append(rewards[:size])
            rewards = rewards[size:]

        full_rewards = []
        for i in range(len(batch_solution_str)):
            if i in indices:
                full_rewards.append(rewards_group[indices.index(i)])
            else:
                full_rewards.append([default_penalty])
        return full_rewards

    async def get_question_diversity_rm_rewards(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            urls=RM_URLS,
            default_penalty=-0.1,
    ):
        """
            整体评价提问的多样性
        """
        addition_judges = []
        new_batch_solution_strs = []
        indices = []

        for i, (_gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            lang_code = _gt["lang_code"]
            if lang_code == "zh":
                judge_template = self.JUDGE_CRITERIA_QUESTION_DIVERSITY_ZH
            else:
                judge_template = self.JUDGE_CRITERIA_QUESTION_DIVERSITY_EN

            notes = get_notes(sol)
            notes_w_coclusions = get_notes_and_conclusions(sol)
            if len(notes) != len(notes_w_coclusions):
                continue

            if len(notes_w_coclusions) == 0:
                continue

            questions = [self.normalize_question(
                _) for _ in notes_w_coclusions]

            if lang_code == "zh":
                questions = [
                    f'- 原文中需要进行提问的部分： \n"{_[1]}"\n- 提问：\n"{_[0]}"' for _ in questions]
                judge_prompt = f'任务：针对文档中理解晦涩、过于简略的部分进行提问。\n\n[Raw Corpus]\n{_gt["ground_truth"]}\n\n\n# 评价标准\n{judge_template}'
            else:
                questions = [
                    f'- Identify the parts in the original text that need to be questioned.\n"{_[1]}"\n- The question raised.\n"{_[0]}"' for _ in questions]
                judge_prompt = f'Task: Ask questions about the obscure and overly brief parts in the document.\n\n[Raw Corpus]\n{_gt["ground_truth"]}\n\n\n# Judge Criteria\n{judge_template}'

            addition_judges.append({
                "ground_truth": judge_prompt
            })
            indices.append(i)
            new_batch_solution_strs.append("\n\n".join(questions))

        tasks = []
        n = len(urls)

        for i, batch in enumerate(batchify(zip(addition_judges, new_batch_solution_strs), n=64)):
            addition_judge = [_[0] for _ in batch]
            new_batch_solution_str = [_[1] for _ in batch]
            tasks.append(
                compute_rm_score(
                    batch_solution_str=new_batch_solution_str,
                    batch_ground_truth=addition_judge,
                    postprocess_solution_fn=lambda x: x,
                    parse_result_failure_score=self.parse_result_failure_score,
                    desc="-question_diversity_judge",
                    urls=[urls[i % n]]
                )
            )

        results = await self.run_tasks_in_queues(tasks, n=n)
        rewards = []
        for _ in results:
            rewards.extend(_)

        full_rewards = []
        for i in range(len(batch_solution_str)):
            if i in indices:
                full_rewards.append(rewards[indices.index(i)])
            else:
                full_rewards.append(default_penalty)
        return full_rewards

    async def get_rm_rewards(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth):
        revise_scores = await self.get_revise_rm_rewards(
            batch_data_sources, batch_solution_str, batch_ground_truth)

        single_question_scores = await self.get_single_question_judge_rm_rewards(
            batch_data_sources, batch_solution_str, batch_ground_truth
        )
        question_diversity_scores = await self.get_question_diversity_rm_rewards(
            batch_data_sources, batch_solution_str, batch_ground_truth
        )

        rewards_union = [0.0] * len(batch_data_sources)
        rewards_split = []
        for i in range(len(batch_data_sources)):
            rewards_split.append(
                [revise_scores[i], single_question_scores[i], question_diversity_scores[i]])

        for i in range(len(batch_data_sources)):
            # TODO: 参数化
            rewards_union[i] += revise_scores[i] * 2.0 + np.sum(
                [_ + 0.5 * question_diversity_scores[i] for _ in single_question_scores[i]])
        return rewards_union, rewards_split

    def compute_score(self,
                      batch_data_sources,
                      batch_solution_str,
                      batch_ground_truth,
                      ):
        async def main():
            return await self._compute_score(batch_data_sources, batch_solution_str, batch_ground_truth)
        return asyncio.run(main())

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             ):

        penalty = defaultdict(dict)
        for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
            for key, fn in self.get_penalties().items():
                penalty[key][i] = fn(
                    solution_str, ground_truth, lang_code=ground_truth["lang_code"])
        base_rewards, base_rewards_split = await self.get_rm_rewards(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )

        final_results = []
        for i in range(len(batch_solution_str)):
            penalty_log_str = []
            _reward = base_rewards[i]

            for name, _penalty in penalty.items():
                if i in _penalty:
                    _reward += _penalty[i] * self.get_penalty_coef()[name]
                    try:
                        penalty_log_str.append(
                            f'{name}={_penalty[i]:.3f}*{self.get_penalty_coef()[name]}')
                    except Exception as _:
                        pass
            final_results.append(_reward)
            thought = get_thought(batch_solution_str[i])

            notes_summary = self.get_notes_summary(batch_solution_str[i])

            _revise, _single_q, _diversity = base_rewards_split[i]
            if self.split == "valid" or (self.split == "train" and random.random() < 0.01):
                log = True
                log_flag = "[VALID]" if self.split == "valid" else "[TRAIN]"
            else:
                log = False

            if log:
                print(
                    f"--------------------------------{log_flag}--------------------------------")
                print(
                    f"【Thought】({len(thought)})`{repr(self.clip_string(thought))}`")
                print(
                    f'【Refine】({batch_ground_truth[i]["lang_code"]})({self.get_document_len(batch_solution_str[i])})`{self.log_solution(batch_solution_str[i])}`')
                print(
                    f'【Raw】({batch_ground_truth[i]["lang_code"]})({len(batch_ground_truth[i]["ground_truth"])})``{self.log_ground_truth(batch_ground_truth[i])}`')
                print(
                    f'[Final Reward]={_reward:.3f}|RM_UNION={base_rewards[i]:.3f}|RM_REVISE={_revise:.2f}|{"|".join(penalty_log_str)}[{self.get_penalty_coef()}]\n')
                for j, note in enumerate(notes_summary):
                    print(
                        f'\t【新增注释{j}】({f"{_single_q[j]:.3f}" if j < len(_single_q) else "<not_found>"}+(0.5*{_diversity:.3f})){repr(note)}')
        return final_results

    def get_notes_summary(self, solution):
        notes_and_conclusions = get_notes_and_conclusions(solution)
        return notes_and_conclusions

    def log_ground_truth(self, ground_truth):
        return repr(self.clip_string(ground_truth["ground_truth"]))

    def log_solution(self, solution):
        norm = parse_doc_w_notes(solution)
        if norm is None:
            return repr(self.clip_string(solution))
        return repr(self.clip_string(norm))

    def get_document_len(self, solution):
        norm = parse_doc_w_notes(solution)
        if norm is None:
            return 0
        return len(norm)

    def clip_string(self, s: str):
        if len(s) > 1500:
            return f'{s[:700]}... [省略] ...{s[-800:]}'
        return s


_qwq_longcot_pretrain_refine_compute_score_train = QwQLongCoTPretrainRefineComputeScore(
    split="train")
_qwq_longcot_pretrain_refine_compute_score_valid = QwQLongCoTPretrainRefineComputeScore(
    split="valid")
qwq_longcot_pretrain_refine_compute_score_train = _qwq_longcot_pretrain_refine_compute_score_train.compute_score
qwq_longcot_pretrain_refine_compute_score_valid = _qwq_longcot_pretrain_refine_compute_score_valid.compute_score
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 预训练数据治理
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 思维过程优化
# ------------------------------------------------------------------------------------------------------------------------------------------------------


class CoTEnhanceComputeScore(QwQLongCoTPretrainRefineComputeScore):
    JUDGE_CRITERIA_COT_ZH = """
## 思维过程评价标准体系

一、逻辑基础
（一）概念清晰（1）
明确界定问题边界、核心概念定义及已知条件，通过具体示例阐释抽象概念，避免模糊假设与语义歧义。操作要点：
采用 "属 + 种差" 法定义核心概念
列举 3 个以上典型 / 非典型案例区分概念外延
标注概念适用的前提条件
（二）推理严密（2）
论证过程严格遵循逻辑规则（归纳 / 演绎 / 类比），清晰呈现每一步推导的逻辑链条，确保论据与结论之间存在必然或合理的推导关系。操作要点：
明示推理类型（如 "采用不完全归纳法，基于 N 个样本数据得出 X 结论"）
标注逻辑规则依据（如 "符合三段论推理规则：大前提 A + 小前提 B→结论 C"）
建立逻辑校验清单（如检查是否存在偷换概念、以偏概全等谬误）
二、分析深度
（一）本质挖掘（3）
超越表面现象，通过追溯问题的底层机制、核心矛盾或本质属性，揭示事物发展的内在规律。操作要点：
运用 5Why 法追问问题根源（连续追问至少 3 层）
建立因果关系模型（如绘制鱼骨图 / 思维导图分析根本原因）
引用基础理论解释（如运用马斯洛需求层次理论分析用户行为动机）
（二）抽象具象（4）
实现抽象理论与具体案例的双向转化，既能从具体实践中提炼普遍规律，也能将抽象原理应用于特定场景。操作要点：
建立 "原理 - 案例" 映射表（如将 SWOT 分析模型对应至某企业战略规划案例）
采用 "假设 - 验证" 循环（从抽象理论出发设计实验方案，通过具体数据验证理论适用性）
制作类比说明（用日常生活案例解释专业理论，如用 "水库蓄水" 类比企业现金流管理）
三、思维广度
（一）多维视角（5）
纳入多维度分析框架，包括不同利益相关者立场（用户 / 竞品 / 决策者 / 执行者）、学科视角（技术 / 经济 / 心理 / 社会）、时间维度（短期 / 中期 / 长期）及空间维度（地域 / 文化 / 行业）。操作要点：
建立利益相关者矩阵（标注各主体的核心诉求与冲突点）
运用 PESTEL 模型进行宏观环境分析（政治 / 经济 / 社会 / 技术 / 环境 / 法律）
制作跨学科理论应用对照表（如在教育领域同时应用认知心理学和统计学理论）
（二）要素关联（6）
识别系统内关键变量及其因果关系、互动机制与反馈回路，构建要素关联模型，避免孤立分析单一因素。操作要点：
绘制因果关系图（标注正 / 负向影响及作用强度）
运用系统动力学方法分析动态反馈（如建立 "价格 - 需求 - 供给" 循环模型）
识别关键临界点（标注变量间影响的阈值条件，如 "当用户增长率超过 X% 时，网络效应开始显现"）
四、创新突破
（一）质疑假设（7）
对固有认知、行业惯例或前提假设进行批判性检验，探讨不同文化语境、技术条件或价值体系下的适用性差异。操作要点：
建立假设清单（列举分析过程中依赖的所有显性 / 隐性假设）
进行反事实推演（如 "假设不存在 XX 限制条件，问题解决方案会发生哪些变化"）
开展跨文化比较（对比中西方用户在相似场景下的行为差异及背后的文化因素）
（二）方案拓展（8）
生成多元化解决方案，包括最优解、次优解、风险解及创新解，系统评估各方案的优缺点、适用场景及实施成本。操作要点：
采用头脑风暴法生成至少 5 种备选方案
建立方案评估矩阵（从可行性 / 收益性 / 风险性等维度进行量化评分）
设计 "反常规" 方案（如在资源受限场景下提出颠覆性解决方案，突破传统思维框架）
五、自我校准
（一）证据依赖（9）
决策过程基于客观数据、事实依据或逻辑推理，主动检索相关研究成果、统计数据及实践案例，建立反例验证机制。操作要点：
建立证据库（分类存储学术论文、行业报告、实验数据等）
采用双盲验证法（由独立第三方对证据的真实性和相关性进行审核）
实施证伪检验（主动寻找与结论相悖的证据，评估其对结论的影响程度）
（二）动态调整（10）
建立结论更新机制，根据新信息（数据反馈、环境变化、逻辑漏洞）及时修正分析过程与最终结论，明确标注调整依据与修订轨迹。操作要点：
设定信息更新频率（如每周 / 月进行一次数据迭代）
建立版本控制体系（记录每次调整的时间、原因及具体修改内容）
设计容错区间（明确在多大程度的信息变化范围内需要启动结论修正流程）
六、实际应用
（一）操作性强（11）
将分析结论转化为可执行的步骤方案，明确每个环节的责任主体、时间节点、资源需求及操作指南。操作要点：
制定 WBS 工作分解结构（将方案分解为可执行的最小任务单元）
编制操作手册（包含流程图、工具清单、风险预案等附件）
进行试点验证（选择典型场景进行小范围测试，收集执行反馈并优化方案）
（二）可验证性（12）
建立清晰的结论检验标准与方法，确保分析过程和结果可重复验证，支持第三方独立评估。操作要点：
设定量化指标（如 "方案实施 3 个月后，目标用户留存率提升 X%"）
设计对照实验（通过 A/B 测试验证不同方案的实际效果）
建立审计追踪机制（记录关键数据来源、分析步骤及决策依据，确保过程可回溯）
七、沟通表达
（一）清晰表达（13）
采用结构化表达形式（如金字塔原理），确保逻辑层次分明、语言准确规范，避免歧义性表述。操作要点：
运用 "结论先行 - 论据支撑 - 细节补充" 的表达框架
建立术语表（统一关键概念的定义与表述方式）
进行歧义性检测（通过第三方盲审识别可能引起误解的表述）
（二）有效沟通（14）
根据受众特点（专业背景、认知水平、利益关注点）调整沟通策略，确保信息传递的准确性与接受度。操作要点：
制作多版本汇报材料（如面向技术团队的专业版 vs 面向管理层的精简版）
运用可视化工具（图表、模型、案例视频等辅助理解复杂逻辑）
设计互动环节（通过问答、研讨等形式确认受众理解程度）
八、反思总结
（一）自我反思（15）
定期对思考过程进行系统性复盘，识别逻辑漏洞、方法缺陷或认知偏差，提出针对性改进措施。操作要点：
建立反思日志（记录每次分析过程中的成功经验与失败教训）
运用 SWOT 法分析自身思维优势与不足
实施 360 度反馈（收集同行、用户等多主体对思考过程的评价意见）
（二）总结提升（16）
将碎片化经验转化为系统化知识体系，建立可复用的分析框架、工具模板及方法论，持续优化思维流程。操作要点：
开发标准化分析工具包（包含常用模型、数据模板、评估清单等）
构建知识图谱（梳理不同领域分析方法的适用场景及相互关联）
设计能力提升计划（针对薄弱环节制定学习路径与实践方案）
"""

    def __init__(self,
                 split="train",
                 parse_result_failure_score=-2.0):
        self.split = split
        self.parse_result_failure_score = parse_result_failure_score

    def compute_score(self,
                      batch_data_sources,
                      batch_solution_str,
                      batch_ground_truth,
                      ):
        async def main():
            return await self._compute_score(batch_data_sources, batch_solution_str, batch_ground_truth)
        return asyncio.run(main())

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             ):

        rewards = await self.get_rm_rewards(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )
        final_results = []
        for i in range(len(batch_solution_str)):
            _reward = rewards[i]
            final_results.append(np.mean(_reward))

            if self.split == "valid" or (self.split == "train" and random.random() < 0.05):
                log = True
                log_flag = "[VALID]" if self.split == "valid" else "[TRAIN]"
            else:
                log = False

            notes_summary = self.get_notes_summary(
                batch_ground_truth[i], batch_solution_str[i])

            if log:
                print(
                    f"================================{log_flag}================================")
                print(
                    f'[Final Reward]={np.mean(_reward):.3f}\n')
                for j, (raw, refine) in enumerate(notes_summary):
                    score = f'{_reward[j]:.3f}' if j < len(
                        _reward) else "<not_found>"
                    print(
                        f'\t【注释{j}修改前】{repr(self.clip_string(raw))}')
                    print(
                        f'\t【注释{j}修改后】({score}){repr(self.clip_string(refine))}\n')
                    print(
                        '----------------------------------------------------------------')
        return final_results

    def get_notes_summary(self, ground_truth, solution):
        notes = ground_truth["notes"]
        raw = {}
        for note in notes:
            question, conclusion = self.get_question(
                note), self.get_conclusion(note)
            raw[(question, conclusion)] = note

        refined = {}
        refinements = self.get_notes_and_conclusions(solution)
        for refine in refinements:
            question, conclusion = self.get_question(
                refine), self.get_conclusion(refine)
            if (question, conclusion) in raw:
                refined[(question, conclusion)] = refine

        summary = []
        for note in notes:
            question, conclusion = self.get_question(
                note), self.get_conclusion(note)
            summary.append((
                raw[(question, conclusion)], refined.get(
                    (question, conclusion), "<corrupt>")
            ))
        return summary

    def get_conclusion(self, s):
        return re.findall(r'\[CONCLUSION\](.*?)\[/CONCLUSION\]', s, re.DOTALL)[0].strip()

    def get_question(self, s):
        if "提问：" in s and "一步步思考：" in s:
            return s[s.index("提问：")+len("提问："):s.index("一步步思考：")].strip()
        if "Question:" in s and "Think Step by Step:" in s:
            return s[s.index("Question:")+len("提问："):s.index("Think Step by Step:")].strip()
        return None

    def get_notes_and_conclusions(self, s: str):
        try:
            notes = re.findall(
                r'\[EXPLANATION\].*?\[/EXPLANATION\]\n*\[CONCLUSION\].*?\[/CONCLUSION\]', s, re.DOTALL)
            return notes
        except Exception as err:
            return []

    async def get_rm_rewards(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            urls=RM_URLS):
        """
            评价除去处思考过程后的改写内容
        """
        mapper = {}
        cot_judges = []
        new_batch_solution_strs = []

        format_corrupt = []

        for i, (_gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            judge_template = self.JUDGE_CRITERIA_COT_ZH

            raw_notes = _gt["notes"]
            judge_prompts = _gt["judges"]

            # 格式检查
            try:
                content = sol[sol.index("[NOTE]\n```")+len("[NOTE]\n```"):]
                content = content[:content.index("```")]
                otherpart = re.sub(
                    r'\[EXPLANATION\][\s\S]*?\[/EXPLANATION\]\n*\[CONCLUSION\][\s\S]*?\[/CONCLUSION\]', "", content, re.DOTALL).strip()

                if len(otherpart) != 0:
                    format_corrupt.append(i)
                    continue
            except Exception as err:
                format_corrupt.append(i)
                continue

            refine_notes = self.get_notes_and_conclusions(sol)

            for j, note in enumerate(raw_notes):
                uniq_key = (self.get_question(note), self.get_conclusion(note))

                matched = None
                for refine in refine_notes:
                    refine_key = (self.get_question(refine),
                                  self.get_conclusion(refine))
                    if any(kw in refine for kw in ("概念清晰", "推理严密", "本质挖掘")):
                        break
                    if refine_key == uniq_key:
                        matched = refine
                        break

                if matched:
                    judge_prompt = f'{judge_prompts[j]}\n\n\n# 评价标准\n{judge_template}'
                    cot_judges.append({"ground_truth": judge_prompt})
                    new_batch_solution_strs.append(matched)
                    mapper[(i, j)] = len(new_batch_solution_strs) - 1
                else:
                    continue

        tasks = []
        n = len(urls)

        for i, batch in enumerate(batchify(zip(cot_judges, new_batch_solution_strs), n=64)):
            batch_judges = [_[0] for _ in batch]
            new_batch_solution_str = [_[1] for _ in batch]
            tasks.append(
                compute_rm_score(
                    batch_solution_str=new_batch_solution_str,
                    batch_ground_truth=batch_judges,
                    postprocess_solution_fn=lambda x: x,
                    parse_result_failure_score=-0.2,
                    desc="-cot_judge",
                    urls=[urls[i % n]]
                )
            )

        results = await self.run_tasks_in_queues(tasks, n=n)

        rewards = []
        for _ in results:
            rewards.extend(_)

        full_rewards = []
        for i, (_gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            if i in format_corrupt:
                full_rewards.append([self.parse_result_failure_score])
                continue

            raw_notes = _gt["notes"]
            full_rewards.append([])
            assert len(raw_notes) > 0
            for j, note in enumerate(raw_notes):
                if (i, j) in mapper:
                    full_rewards[-1].append(rewards[mapper[(i, j)]])
                else:
                    full_rewards[-1].append(-0.2)

        return full_rewards


_cot_enhance_compute_score_train = CoTEnhanceComputeScore(
    split="train", parse_result_failure_score=-10.0)
_cot_enhance_compute_score_valid = CoTEnhanceComputeScore(
    split="valid", parse_result_failure_score=-10.0)
cot_enhance_compute_score_train = _cot_enhance_compute_score_train.compute_score
cot_enhance_compute_score_valid = _cot_enhance_compute_score_valid.compute_score

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 思维过程优化
# ------------------------------------------------------------------------------------------------------------------------------------------------------
