import re
import jieba
import random
import aiohttp
import asyncio
import requests
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

RM_URLS = [
    "http://10.130.1.95:34604",
    "http://10.130.1.95:34580",
    "http://10.130.1.95:31929",
    "http://10.130.1.95:31400"
]
# RM_URLS = [
#     "http://10.130.1.95:27337",
#     "http://10.130.1.95:25950",
#     "http://10.130.1.95:26240",
#     "http://10.130.1.95:34377"
# ]
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


async def rm_request_with_retry(urls, data, max_retries=3, retry_delay=1, suffix="/reward"):
    retries = 0
    while retries < max_retries:
        try:
            url = random.choice(urls)
            async with aiohttp.ClientSession() as session:
                async with session.post(f'{url}{suffix}', json=data, timeout=aiohttp.ClientTimeout(total=1200)) as response:
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
        for batch in tqdm_nonasync(batchify(input_datas, n=128), desc=f'[RM{desc}][{urls}] batchify inference (batch=64)'):
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
        return notes
    except Exception as err:
        return []


class MainBodyRecall(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 parse_result_failure_score=0.,
                 high_range=0.75,
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

            rouge_recall = (score["rouge1"].recall +
                            score["rouge2"].recall) / 2.0

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
            pass
            # FIXME
            # if contain_chinese(thought):
            #     reward += self.penalty_base / 2

        explanation = re.findall(
            r'\[EXPLANATION\](.*?)\[/EXPLANATION\]', document, re.DOTALL)
        if len(explanation) != 0:
            if lang_code == "zh":
                consist = 0
                # FIXME
                # consist = len(
                #     [_ for _ in explanation if contain_chinese(_)]) / len(explanation)
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


class NotesFormatReward(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 max_reward=0.1,
                 step_reward=0.01,
                 max_steps=10,
                 ):
        self.postprocess_solution_fn = postprocess_solution_fn
        self.max_reward = max_reward
        self.step_reward = step_reward
        self.max_steps = max_steps

    def get_penalty_or_reward(self, solution_str, ground_truth, lang_code=None):
        solution_str = self.postprocess_solution_fn(solution_str)
        if solution_str is None:
            return 0.

        base_score = 0.0

        # [EXPLANATION][/EXPLANATION]闭合
        wo_notes = re.sub(r'\[EXPLANATION\][\s\S]*?\[/EXPLANATION\]',
                          "", solution_str, flags=re.DOTALL)
        if any(_ in wo_notes for _ in ("[EXPLANATION]", "[/EXPLANATION]")):
            base_score -= 0.5

        # 思考过程奖励
        try:
            loose_follow = re.findall(
                r'\[EXPLANATION\].*?\[/EXPLANATION\]\n*\[CONCLUSION\].*?\[/CONCLUSION\]', solution_str, re.DOTALL)
            strict_follow = [_ for _ in loose_follow if (
                ("Question:" in _ and "Think Step by Step:" in _) or ("提问：" in _ and "一步步思考：" in _))]
            score = min(len(loose_follow), self.max_steps) * self.step_reward/2 + \
                min(len(strict_follow), self.max_steps) * self.step_reward/2
            return base_score + min(score, self.max_reward)
        except Exception as err:
            return base_score


class NotesRepetitionPenalty(PenaltyOrReward):
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

        def normalize(s):
            s = s.replace("[EXPLANATION]", "").replace(
                "[/EXPLANATION]", "").strip()
            s = s.replace("Q:", "").replace("Think:", "").strip()
            s = s.replace("Question:", "").replace(
                "Think Step by Step:", "").strip()
            s = s.replace("提问：", "").replace("一步步思考：", "").strip()
            return s

        notes = re.findall(
            r'\[EXPLANATION\].*?\[/EXPLANATION\]', solution_str, re.DOTALL)
        notes_str = "\n".join([normalize(_) for _ in notes])
        if len(notes) == 0:
            return 0.

        gt = ground_truth["ground_truth"]

        score = self.scorer.score(gt, notes_str)
        rouge_recall = score["rouge2"].recall

        penalty = 0.
        if rouge_recall < 0.05:
            penalty = 0.
        else:
            penalty = -rouge_recall
        return penalty


class QwQLongCoTPretrainRefineComputeScore(object):
    JUDGE_CRITERIA_WO_NOTES = """
以下是深度整合 **内容删除治理、内容改写治理** 后的 **完整大模型数据治理评价标准（Criteria）**：


### **一、内容纯净度 **
- **违规内容彻底清除**：色情暗示、赌博诱导、广告营销（含链接/二维码/品牌硬广）、政治敏感、仇恨言论、暴力描述等显性/隐性违规内容、医疗文档禁“包治百病”“神医”，教育文档禁“考试答案”“内部渠道”，法律文档禁“套路贷”“虚假诉讼”暗示。
- **格式噪声**：标准化格式，去除乱码，修正过度标点。
- **内容噪声**：重复内容去重、与上下文无关的孤立短句、无意义语气词堆砌
- **学习噪声**：删除 ISBN、网址、论文引用文献、DOI、ISSN、ORCID 等学术标识符；删除ISBN、时间信息、网址等对内容理解无关的信息，清除不可恢复的多模态内容


### **二、语义修复有效性**
核心目标：最小干预修复问题，完整保留核心语义
1. **基础规范**：修正拼写语法错误，统一标点，规范技术格式
2. **语义优化**：补全不完整句子，合并重复表意
3. **逻辑增强**：明确指代，调整语序，补充逻辑连接词
4. **质量提升**：消除机翻痕迹，修复逻辑断裂


## 三、信息完备性
核心目标：确保原文有效信息完整，避免不必要修改
1. **信息保留**：除需要删除、改写外的其他信息完整保留
2. **最小干预**：仅修正明确错误，不改变原文主要内容


## 四、格式规范性
核心目标：统一治理后文档格式，确保技术元素正确
1. **规范段落间距、表格格式**
2. **确保Markdown、代码块、LaTeX等技术格式正确**


## 五、语言一致性
核心目标：保持原文语言风格和语种统一
1. **语种统一**：全文语种一致，代码注释与代码语种匹配
2. **风格匹配**：保持与原文一致的正式度和专业术语使用
"""

    JUDGE_CRITERIA_W_NOTES = """
## 内容新增治理之“思考过程”专项评价标准

### 一、结构规范性
1. **标签使用准确性**
   - 正确使用标记：
     - 非代码文本使用“[EXPLANATION]...[/EXPLANATION]”，代码场景在注释区域添加
     - 紧接着思考过程“[EXPLANATION]...[/EXPLANATION]”之后，使用“[CONCLUSION]...[/CONCLUSION]”把原文中内容作为结论部分。
   - 标签内是否严格遵循英文“Question: *** Think Step by Step: ***”的自问自答格式中文“提问：*** 一步步思考：***”；禁止出现无设问的纯叙述性思考

2. **位置合理性**
   - 思考过程是否出现在需要解释的内容**之前**，确保问题导向性
   - 避免在无关位置插入思考（如在结论后补充问题，或在段落中间突兀插入不相关思考）
   - 提问、思考与原文内容（[EXPLANATION]...[/EXPLANATION][CONCLUSION]...[/CONCLUSION]）需要构成“问题-思考-结论”的直接映射关系

### 二、内容价值性
#### 1. **信息增量有效性**
- **优质特征**：
  - 包含原文未明确写出的 **背景知识、原理依据、潜在假设、风险分析或应用场景**。
  - 逻辑链条延伸而非表面复述，体现 **“为何如此”“如何推导”**。
  - 避免同义转换，需引入 **跨学科关联、前沿动态或实际案例**。
- **低效特征**：
  - 仅对原文进行 **同义替换或简单概括**。
  - 无实质新信息，如空泛表述“这是重要研究方向”“对行业有帮助”，未说明具体价值或原理。
  - 直接复制原文结论，未补充 **推导过程或隐性逻辑**。

#### 2. **问题导向精准性**
- **优质特征**：
  - 提问聚焦 **“核心矛盾”或“认知盲区”**，如“为何选择X方法而非Y方法？”“实验数据中的异常值如何处理？”，直指逻辑薄弱点。
  - 问题具体且有指向性，避免宽泛或无意义设问（例如：“如何优化算法时间复杂度？”而非“算法有什么用？”）。
  - 提问与原文结论形成 **“问题-答案”闭环**，思考内容需完整回应问题并提供深层解释。
- **低效特征**：
  - 问题表面化，仅复述原文内容。
  - 问题模糊笼统，如“如何理解该理论？”“说明标准的重要性”，未明确具体思考维度。
  - 提问与原文结论无关，或无法通过思考过程推导出结论。

#### 3. **批判性思维体现**
- **优质特征**：
  - **多维度分析**：引入对比（如“X方法vs.Y方法的优劣”）、假设（如“若改变实验条件，结果将如何？”）、风险评估（如“该模型在长尾数据下的潜在偏差”）。
  - **挖掘隐含条件**：主动识别原文未明示的前提或逻辑漏洞。
  - **提出替代方案**：针对多解问题探索其他路径。
- **低效特征**：
  - 单向度解释，仅陈述“是什么”或“有效”，未分析“为什么有效”或“局限性”（例如：“该方法可行”，未说明适用边界）。
  - 直接接受原文结论，未质疑潜在假设。
  - 缺乏对比或风险意识，如忽略“不同场景下方法效果差异”“数据偏差对结论的影响”。

#### 4. **知识衔接深度**
- **优质特征**：
  - 补全 **逻辑断层**：将原文隐含的推导步骤显性化（例如：数学证明中补充“辅助线构造利用等腰三角形对称性”的几何原理）。
  - 建立 **跨维度关联**：连接单一知识点与学科底层原理、实际应用或前沿研究（例如：将“分子生物学研究”与“疾病诊断工具开发”结合，说明技术转化逻辑）。
  - 分层拆解复杂问题，体现“从原理到应用”的链条（例如：解释“代码实现”时，先说明算法思想，再拆解变量功能和边界条件处理）。
- **低效特征**：
  - 浅层次关联，仅罗列概念或步骤（例如：“研究包含A、B、C方向”，未说明各方向的内在联系）。
  - 碎片化陈述，缺乏因果推导（例如：“实验需多次测量”，未解释“误差分布→数据处理”的科学依据）。
  - 未衔接底层原理，如直接使用专业术语而不解释（例如：提及“熵增”但未定义“熵”的物理意义）。
"""

    JUDGE_CRITERIA_THINKING_QUALITY = """
## 思考过程质量评价标准

#### **一、逻辑结构**  
1. **链条连贯性**  
   - 步骤是否按“问题识别→要素分析→逻辑推导”线性展开，无断裂或循环重复。  
2. **分层拆解度**  
   - 复杂问题是否拆解为可操作的**分阶子步骤**，每阶任务清晰可追溯。  


#### **二、问题拆解能力**  
3. **要素精准度**  
   - 是否准确定位问题的**核心驱动变量**）、约束条件或假设。  
4. **因果逻辑性**  
   - 是否从“现象描述”深入到“机制分析”，再延伸至“影响推导”，形成“是什么→为什么→如何影响”的递进链条。  


#### **三、专业深度与实用性**  
5. **理论嵌入性**  
   - 是否调用领域核心理论或技术原理支撑分析，避免停留在表面描述。  
6. **现实适配性**  
   - 是否兼顾理论理想化假设与实际约束。  


#### **四、过程完整性**  
7. **目标聚焦性**  
   - 所有分析步骤是否紧密围绕初始问题核心。  
8. **步骤详实度**  
   - 关键环节是否包含**具体操作细节**，避免笼统概括。  


#### **五、表达规范**  
9. **术语准确性**  
   - 专业术语使用是否精准，无概念混淆。  
10. **表述简洁性**  
    - 是否用“首先/其次”“由于/因此”等逻辑连接词串联步骤，**避免同义重复或冗长描述*。  


### **核心特征**  
- **无冗余性**：思考过程与结论指向明确，步骤间无重复论证，尤其是思考过程和结论无重复内容展开。  
- **缜密分阶性**：复杂问题必按**可操作的子步骤展开**，每阶逻辑严密、细节完整。  
- **因果闭环性**：从问题提出到结论形成形成完整链条，关键推导环节无逻辑断层。  


### 优化说明  
1. **规避重复**：通过“链条连贯性”“目标聚焦性”等条目，明确要求步骤围绕核心问题单向推进，杜绝同一概念在不同维度重复描述。  
2. **强化缜密性**：新增“分层拆解度”“步骤详实度”，强调复杂问题必须拆解为**可执行的分阶任务**，每个环节需包含具体操作细节，避免笼统表述。
"""
    JUDGE_CRITERIA_QUESTION_QUALITY = """
    ## 高质量提问评价标准

1. **相关性**  
   - 问题必须直接指向原文待解释的关键知识点、逻辑断层或核心结论，避免无关发散。  
   - 能精准定位内容中的核心矛盾、未明确假设或潜在逻辑跳步。  

2. **多样性**  
   - 问题视角多样性，需要覆盖尽可能更多的场景和提问类型；问题提问视角千万不要单一；下面是一些不同提问类型的例子

    #### **1. 数学证明思路补全**  
    提问：如何证明柯西不等式的向量形式？  

    #### **2. 算法复杂度分析**  
    提问：如何推导快速排序算法的平均时间复杂度？  

    #### **3. 物理定律推导**  
    提问：如何从开普勒行星运动定律推导出万有引力定律？  

    ### **4. 教育场景**  
    提问：这道微积分题的出题思路是什么？为什么要这么出题  

    #### **5. 实验题能力要求分析**  
    提问：本研究为何选择 XX 群体作为研究对象？分组时采用 XX 标准的依据是什么？技术路线中关键步骤的顺序安排基于哪些方法论原则？

    #### **6. 语文阅读理解题考点设置**  
    提问：小说阅读题为何反复强调“老钟摆”的意象？  

    #### **7. 方法学合理性论证**
    提问：为何选择 XX 实验方法而非其他技术？所采用的统计工具在本研究中的适用性及潜在局限是什么？数据收集方法如何平衡效度与信度？

    #### **8. 数据解读深度拓展**
    提问：如何从现有数据推断出 XX 生物学 / 医学机制？数据中 XX 异常趋势可能反映了哪些未被揭示的潜在规律？

    #### **9. 跨文献研究对比分析**
    提问：本研究结论与 XX 文献的差异，可能由哪些样本选择、实验条件或理论假设的不同导致？如何解释文献中相互矛盾的研究发现？

    #### **10. 研究局限性与改进方向**
    提问：研究设计中样本量不足或数据维度单一可能导致哪些结论偏差？未来研究可从哪些技术手段或研究视角进行优化？
    
    #### **11. 结论推导严谨性补充**
    提问：从数据统计结果到核心结论的推导过程中，需要补充哪些逻辑验证步骤以避免过度推断？
    
    #### **12. 技术文档场景**
    提问：API 接口中参数类型校验、长度限制和格式规范的设计，主要基于哪些安全性、兼容性和易用性考量？

    #### **13. 算法参数调优逻辑**
    提问：在机器学习模型训练中，选择 XX 学习率、批次大小和正则化系数的决策依据是什么？如何通过交叉验证平衡模型的偏差与方差？

    #### **14. 古代政策文本解读**
    提问：史书中记载的 XX 朝代赋税制度（如两税法、一条鞭法），其征税对象和税率调整的背后反映了哪些经济结构变化和统治逻辑？

    #### **15. 外交文书语义辨析**
    提问：近代 XX 条约（如《南京条约》《马关条约》）中 “利益均沾”“最惠国待遇” 等模糊表述，如何体现当时列强的权力博弈和外交策略？

    #### **16. 法律文书场景**
    提问：商业合同中违约责任条款为何区分 “轻微违约”“重大违约” 并设置不同赔偿标准？条款中 “不可抗力” 范围的列举式规定有何法律实务意义？

    #### **17. 法律条文歧义消解**
    提问：刑法中 “情节严重”“数额较大” 等弹性条款，司法实践中主要依据哪些司法解释和指导性案例进行具体认定？
    
    #### **18. 翻译文本场景**
    提问：中医术语 “经络”“气血” 在英译时，为何采用音译加注释而非字面直译？这种处理如何兼顾文化独特性与国际学术理解？

    #### **19. 法律文本术语校准**
    提问：国际公约中 “Intellectual Property” 译为 “知识产权” 而非 “知识财产”，主要基于哪些法律概念的精准性和行业惯例考量？
    
    #### **20. 法律文本术语校准**
    提问：正式商务邮件为何采用 “结论先行 — 细节支撑 — 行动诉求” 的结构？这种布局如何提升收件人的信息处理效率？

    #### **21. 跨部门沟通话术设计**
    提问：在跨部门协作中，“问题描述 — 影响分析 — 解决方案” 的沟通模板如何减少信息误差并推动快速共识？

3. **逻辑深度**  
   - 围绕“为什么”“如何”“基于什么假设”等深层逻辑展开，而非停留在表面事实。  
   - 体现对知识原理、方法论或潜在风险的追问。  

4. **引导性**  
   - 问题需自然引出“步骤性思考”，如通过“需要哪些前提条件？”“分几步验证？”等表述，为后续解释提供逻辑框架。  
   - 避免封闭性问题（如“是对的吗？”），应鼓励展开式推导（如“从实验数据到结论的归纳过程中，可能存在哪些推导漏洞？”）。  

5. **批判性视角（挖掘深层问题）**  
   - 主动质疑原文假设、方法局限性或逻辑漏洞。  
   - 探索替代方案或逆向思考。  
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
        self.note_rep = NotesRepetitionPenalty(
            postprocess_solution_fn=parse_doc_w_notes)
        self.lang_consist = LanguageConsistencyReward(
            postprocess_solution_fn=parse_solution_fn)

    def get_penalties(self) -> Dict[str, Callable]:
        return {
            "TextRecall": self.recall.get_penalty_or_reward,
            "LengthDiff": self.len_diff.get_penalty_or_reward,
            "NoteFormat": self.note_format.get_penalty_or_reward,
            "NoteRep": self.note_rep.get_penalty_or_reward,
            "LangConsistency": self.lang_consist.get_penalty_or_reward
        }

    def get_penalty_coef(self):
        return {
            "TextRecall": 1.0,
            "LengthDiff": 1.0,
            "NoteFormat": 1.0,
            "NoteRep": 0.7,
            "LangConsistency": 1.0
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
            refine_judges.append({
                "ground_truth": f'你是一名专精于大模型数据改写的治理专家。目标是给定一篇从网页爬取或者PDF解析出来的文档，改写成一篇优质的大语言模型预训练语料。\n\n[Raw Corpus]\n{_["ground_truth"]}\n\n\n# 评价标准\n{self.JUDGE_CRITERIA_WO_NOTES}'
            })
        rewards = await compute_rm_score(
            urls=urls,
            batch_solution_str=batch_solution_str,
            batch_ground_truth=refine_judges,
            postprocess_solution_fn=parse_doc_wo_notes_and_tags,
            parse_result_failure_score=self.parse_result_failure_score,
            desc="-judge_wo_notes"
        )
        return rewards

    async def get_notes_mix_rm_rewards(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            urls=RM_URLS,
            default_penalty=-0.1
    ):
        """
            评价整体改写后的内容（思考过程+原文）
        """
        addition_judge = []
        new_batch_solution_str = []
        indices = []
        for i, (_, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            notes = get_notes(sol)
            notes_w_coclusions = get_notes_and_conclusions(sol)
            if len(notes) != len(notes_w_coclusions):
                continue
            if len(notes) == 0:
                continue
            addition_judge.append({
                "ground_truth": f'你是一名专精于大模型数据改写的治理专家。目标是给定一篇从网页爬取或者PDF解析出来的文档，改写成一篇优质的大语言模型预训练语料。目标是给定一篇从网页爬取或者PDF解析出来的文档增加注释（思考过程）。好的新增思考过程应当满足下面的标准\n\n# 评价标准\n{self.JUDGE_CRITERIA_W_NOTES}'
            })
            indices.append(i)
            new_batch_solution_str.append(sol)

        rewards = await compute_rm_score(
            urls=urls,
            batch_solution_str=new_batch_solution_str,
            batch_ground_truth=addition_judge,
            postprocess_solution_fn=parse_doc_w_notes,
            parse_result_failure_score=self.parse_result_failure_score,
            desc="-judge_w_notes"
        )
        full_rewards = []
        for i in range(len(batch_solution_str)):
            if i in indices:
                full_rewards.append(rewards[indices.index(i)])
            else:
                full_rewards.append(default_penalty)
        return full_rewards

    async def get_thinking_rm_rewards(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            urls=RM_URLS,
            default_penalty=-0.1
    ):
        """
            单独评价思考过程
        """
        addition_judge = []
        new_batch_solution_str = []
        indices = []
        for i, (_, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            notes = get_notes(sol)
            notes_w_coclusions = get_notes_and_conclusions(sol)
            if len(notes) != len(notes_w_coclusions):
                continue

            if len(notes_w_coclusions) == 0:
                continue

            addition_judge.append({
                "ground_truth": f'你是一个擅长提问的思考者。你需要提出高质量的问题并回答。\n\n# 评价标准\n{self.JUDGE_CRITERIA_QUESTION_QUALITY}'
            })
            indices.append(i)
            new_batch_solution_str.append("\n\n\n".join(notes_w_coclusions))
        rewards = await compute_rm_score(
            urls=urls,
            batch_solution_str=new_batch_solution_str,
            batch_ground_truth=addition_judge,
            postprocess_solution_fn=lambda x: x,
            parse_result_failure_score=self.parse_result_failure_score,
            desc="-judge_thinking"
        )
        full_rewards = []
        for i in range(len(batch_solution_str)):
            if i in indices:
                full_rewards.append(rewards[indices.index(i)])
            else:
                full_rewards.append(default_penalty)
        return full_rewards

    async def get_question_rm_rewards(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            urls=RM_URLS,
            default_penalty=-0.1
    ):
        """
            单独评价提问质量
        """
        addition_judge = []
        new_batch_solution_str = []
        indices = []
        for i, (_, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            notes = get_notes(sol)
            notes_w_coclusions = get_notes_and_conclusions(sol)
            if len(notes) != len(notes_w_coclusions):
                continue
            if len(notes_w_coclusions) == 0:
                continue

            addition_judge.append({
                "ground_truth": f'你是一个擅长提问的思考者。你需要提出高质量的问题并回答。\n\n# 评价标准\n{self.JUDGE_CRITERIA_QUESTION_QUALITY}'
            })
            indices.append(i)
            new_batch_solution_str.append("\n\n\n".join(notes_w_coclusions))
        rewards = await compute_rm_score(
            urls=urls,
            batch_solution_str=new_batch_solution_str,
            batch_ground_truth=addition_judge,
            postprocess_solution_fn=lambda x: x,
            parse_result_failure_score=self.parse_result_failure_score,
            desc="-judge_questioning"
        )
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
        tasks = [
            self.get_revise_rm_rewards(
                batch_data_sources, batch_solution_str, batch_ground_truth, urls=[RM_URLS[0]]),
            self.get_notes_mix_rm_rewards(
                batch_data_sources, batch_solution_str, batch_ground_truth, urls=[RM_URLS[1]]),
            self.get_thinking_rm_rewards(
                batch_data_sources, batch_solution_str, batch_ground_truth, urls=[RM_URLS[2]]),
            self.get_question_rm_rewards(
                batch_data_sources, batch_solution_str, batch_ground_truth, urls=[RM_URLS[3]])
        ]
        results = await asyncio.gather(*tasks)
        rewards_union = [0.0] * len(batch_data_sources)
        rewards_split = []
        for i in range(len(batch_data_sources)):
            rewards_split.append([0.0] * len(tasks))

        for j, result in enumerate(results):
            for i, reward in enumerate(result):
                rewards_union[i] += reward
                rewards_split[i][j] = reward

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
                penalty[key][i] = fn(solution_str, ground_truth)
        base_rewards, base_rewards_split = await self.get_rm_rewards(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth
        )

        final_results = []
        for i in range(len(batch_solution_str)):
            penalty_log_str = []
            _reward = base_rewards[i]

            for name, _penalty in penalty.items():
                if i in _penalty:
                    _reward += _penalty[i] * self.get_penalty_coef()[name]
                    try:
                        penalty_log_str.append(f'{name}={_penalty[i]:.3f}')
                    except Exception as _:
                        pass

            final_results.append(_reward)
            thought = get_thought(batch_solution_str[i])

            notes_summary = self.get_notes_summary(batch_solution_str[i])

            if self.split == "valid":
                print(
                    f"--------------------------------[VALID]--------------------------------")
                print(
                    f"【Thought】({len(thought)})`{repr(self.clip_string(thought))}`")
                print(
                    f"【Refine】({self.get_document_len(batch_solution_str[i])})`{self.log_solution(batch_solution_str[i])}`")
                print(
                    f'【Raw】({len(batch_ground_truth[i]["ground_truth"])})``{self.log_ground_truth(batch_ground_truth[i])}`')
                print(
                    f'[Final Reward]={_reward:.3f}|RM_UNION={base_rewards[i]:.3f}|RM_SPLIT={base_rewards_split[i]}|{"|".join(penalty_log_str)}[{self.get_penalty_coef()}]\n')
                for i, note in enumerate(notes_summary, start=1):
                    print(f'\t【新增注释{i}】{repr(note)}')
            elif self.split == "train" and random.random() < 0.01:
                print(
                    f"--------------------------------[TRAIN]--------------------------------")
                print(
                    f"【Thought】({len(thought)})`{repr(self.clip_string(thought))}`")
                print(
                    f"【Refine】({self.get_document_len(batch_solution_str[i])})`{self.log_solution(batch_solution_str[i])}`")
                print(
                    f'【Raw】({len(batch_ground_truth[i]["ground_truth"])})`{self.log_ground_truth(batch_ground_truth[i])}`')
                print(
                    f'[Final Reward]={_reward:.3f}|RM_UNION={base_rewards[i]:.3f}|RM_SPLIT={base_rewards_split[i]}|{"|".join(penalty_log_str)}[{self.get_penalty_coef()}]\n')
                for i, note in enumerate(notes_summary, start=1):
                    print(f'\t【新增注释{i}】{repr(note)}')
        return final_results

    def get_notes_summary(self, solution):
        notes_and_conclusions = get_notes_and_conclusions(solution)
        notes = get_notes(solution)
        for _ in notes:
            if not any(_ in c for c in notes_and_conclusions):
                notes_and_conclusions.append(_)
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
