import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Union

import pandas as pd
from tqdm import tqdm

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.utils import build_judge
from vlmeval.dataset.utils.judge_cache import (dump_judge_cache, get_judge_cache_file,
                                               get_judge_detail_file, get_judge_score_file,
                                               load_judge_cache)
from vlmeval.dataset.utils.simplevqa import SimpleVQAEval, normalize_verdict
from vlmeval.smp import file, get_logger, misc

NUM_WORKERS = 16
CACHE_DUMP_INTERVAL = 20
logger = get_logger(__name__)

COMPARE_ANSWER_PROMPT = """
            请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。您的任务是将结果评定为：【正确】、【错误】或【未尝试】。
            首先，我们将列出每个评定类别的示例，然后请您对新问题的预测答案进行评定。
            ## 以下是【正确】的答复示例：
            ```
            问题：图中人物（贝拉克·奥巴马）的孩子叫什么名字？
            标准答案：玛丽亚·奥巴马和萨莎·奥巴马
            模型预测1：Malia Obama and Sasha Obama
            模型预测2：玛丽亚和萨沙
            模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认
            模型预测4：巴拉克·奥巴马有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。玛丽亚出生于1998年7月4日，萨莎出生于2001年6月10日。
            ```
            这些答复均为【正确】，因为：
            - 完整地包含了标准答案中的重要信息。
            - 不包含任何与标准答案矛盾的信息。
            - 只关注语义内容，中英文，大小写、标点、语法和顺序不重要。
            - 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。

            ## 以下是【错误】的答复示例：
            ```
            问题：图中人物（贝拉克·奥巴马）的孩子叫什么名字？
            标准答案：玛丽亚·奥巴马和萨莎·奥巴马
            模型预测1：玛丽亚
            模型预测2：玛丽亚、萨莎和苏珊
            模型预测3：巴拉克·奥巴马没有孩子
            模型预测4：我认为是玛丽亚和萨莎。或者是玛丽亚和杰基。或者是乔伊和玛丽亚。
            模型预测5：虽然我不知道他们的确切名字，但能说出巴拉克·奥巴马有三个孩子。
            模型预测6：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。那是正确的答案吗？
            ```
            这些答复均为【错误】，因为：
            - 答复中包含与标准答案矛盾的事实陈述。即使在陈述中略带保留（例如：“可能是”，“虽然我不确定，但我认为”），也视为错误。

            ## 以下是【未尝试】的答复示例：
            ```
            问题：图中人物（贝拉克·奥巴马）的孩子叫什么名字？
            标准答案：玛丽亚·奥巴马和萨莎·奥巴马
            模型预测1：我不知道。
            模型预测2：我需要更多关于您所指奥巴马的上下文。
            模型预测3：不查阅网络我无法回答这个问题，不过我知道巴拉克·奥巴马有两个孩子。
            模型预测4：巴拉克·奥巴马有两个孩子。我知道其中一个叫玛丽亚，但我不确定另一个的名字。
            模型预测5：我无法识别图中的人物。
            模型预测6：N/A。
            ```
            这些答复均为【未尝试】，因为：
            - 没有包含标准答案中的重要信息。
            - 回复中没有与标准答案矛盾的陈述。
            如果需要用字母作答，请使用 A/B/C 分别表示【正确】/【错误】/【未尝试】；最终 JSON 中的结论仍须使用中文。

            另外注意以下几点：
            - 对于标准答案为数字的问题，预测答案应和标准答案一致。例如，考虑问题“金山铁路黄浦江特大桥的全长是多少米？”，标准答案为“3518.17”：
            - 预测答案“3518”、“3518.1”、“3518.17”均为【正确】。
            - 预测答案“3520”和“3600”均为【错误】。
            - 预测答案“大约3500米”和“超过3000米”被视为【未尝试】，因为它们既不确认也不与标准答案矛盾。
            - 如果标准答案包含比问题更多的信息，预测答案只需包含问题中提到的信息。
            - 例如，考虑问题“菱镁矿的主要化学成分是什么？”,标准答案为“碳酸镁（MgCO3）”。“碳酸镁”或“MgCO3”均视为【正确】答案。
            - 如果从问题中明显可以推断出预测答案省略的信息，那么算作【正确】。
            - 例如，问题“巴鲁米尼的努拉吉遗迹在1997年被联合国教科文组织列为世界文化遗产，那么这遗址在哪个地区？”标准答案为“意大利撒丁岛”，预测答案“撒丁岛”被视为【正确】。
            - 如果能明显看出名字翻译版本不同但是是同一个人也认为【正确】。
            - 例如，如果标准答案是“Robinson”，那么回答鲁滨逊或者鲁滨孙均【正确】。
            - 预测答案和标准答案对应的是同一事物，但是称呼不同，如“天坛”和“祈年殿”，那么算作【正确】

            ## 下面是一个新的问题示例。对每一个预测答案，请只回复"正确"、"错误"、"未尝试"之一，不要道歉或纠正自己的错误，只需要评估该回答。
            ```
            问题: {question}
            正确答案: {answer}
            预测答案: {candidates}
            ```

            请严格按照以下格式回复，以JSON格式返回一个字典,而且字典第一层key不要替换为具体答案。不要返回其他任何内容。
            [回复格式]:
            ```json
            {{
                "预测答案0": "整体结论，只返回'正确','错误','未尝试'中的一个，无须添加其他文本",
                "预测答案1": "整体结论，只返回'正确','错误','未尝试'中的一个，无须添加其他文本",
                ...
            }}
            ```
            """.strip()


def simplevqa_judge_failed(result):
    """Return True when a cached judge result holds no usable verdict."""
    if not isinstance(result, dict):
        return True
    judge_results = result.get('judge_res')
    if not isinstance(judge_results, dict) or not judge_results:
        return True

    judge_res = judge_results.get('model_response')
    if isinstance(judge_res, dict):
        judge_res = judge_res.get('conclusion')
    return normalize_verdict(judge_res) is None


def _extract_json_object(text):
    """Best-effort extraction of a JSON object from a judge response."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    text = re.sub(r'^```(?:json|python)?', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'```$', '', text).strip()
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_verdict(text):
    """Parse a judge response and return its first normalized verdict."""
    parsed = _extract_json_object(text)
    if not parsed:
        return None
    for value in parsed.values():
        if isinstance(value, dict):
            value = value.get('conclusion')
        verdict = normalize_verdict(value)
        if verdict is not None:
            return verdict
    return None


def _failed_result(question, answer, model_response):
    return {
        'question': question,
        'answer': answer,
        'model_response': model_response,
        'judge_res': {'model_response': {'conclusion': '答案解析失败'}},
    }


class SimpleVQA(ImageBaseDataset):
    TYPE = "VQA"
    DEFAULT_JUDGE_MODEL = 'gpt-4o'
    DATASET_URL = {
        "SimpleVQA": "https://opencompass.openxlab.space/utils/VLMEval/SimpleVQA.tsv",
    }
    DATASET_MD5 = {
        "SimpleVQA": "e3ce3c11df59a2a15d37489a8b245a87",
    }

    def build_prompt(self, line: Union[int, pd.Series], qa_type: str = 'Direct') -> List[Dict[str, str]]:
        """
        Build a prompt for the model from a data line.

        Args:
            line: Either an index into the dataset or a pandas Series
            qa_type: Choose from ['Direct', 'CoT', 'PoT']

        Returns:
            List of message dictionaries containing the image and question
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = misc.toliststr(line["image"])
        else:
            tgt_path = self.dump_image(line)

        # load data line elements
        question = line['question']
        assert len(tgt_path) == 1

        # form messages
        msgs = []
        msgs = [dict(type='image', value=tgt_path[0])]
        msgs.append(dict(type='text', value=question))

        return msgs

    def get_scores(self, result_file: str, cache_file: str, **judge_kwargs: Any):
        data = file.load(result_file)
        judge_kwargs = judge_kwargs.copy()
        nproc = judge_kwargs.pop('nproc', NUM_WORKERS)

        indices = data['index'].tolist() if 'index' in data.columns else list(range(len(data)))
        row_by_key = {cache_key: row_idx for row_idx, cache_key in enumerate(indices)}
        cache = load_judge_cache(cache_file)
        judge_model = None

        def process_one(row_idx, cache_key):
            row = data.iloc[row_idx]
            question, answer, model_response = row['question'], row['answer'], row['prediction']
            candidates = "\n[预测答案0]：{}".format(model_response)
            prompt = COMPARE_ANSWER_PROMPT.format(
                question=question, answer=answer, candidates=candidates)
            verdict = None
            try:
                response = judge_model.generate(prompt, temperature=1)
                verdict = _extract_verdict(response)
            except Exception as err:
                logger.warning(
                    'SimpleVQA judge failed on index %s: %s: %s',
                    cache_key, type(err).__name__, err)

            if verdict is None:
                return cache_key, _failed_result(question, answer, model_response)
            return cache_key, {
                'question': question,
                'answer': answer,
                'model_response': model_response,
                'judge_res': {'model_response': {'conclusion': verdict}},
            }

        pending = [
            (row_idx, cache_key)
            for row_idx, cache_key in enumerate(indices)
            if cache_key not in cache or simplevqa_judge_failed(cache[cache_key])
        ]

        if pending:
            # Use the same judge construction path as other benchmarks so that
            # --judge / --judge-args / --judge-base-url / --judge-key are honoured.
            judge_model = build_judge(**judge_kwargs)
            with ThreadPoolExecutor(max_workers=max(nproc, 1)) as executor:
                futures = {
                    executor.submit(process_one, row_idx, cache_key): cache_key
                    for row_idx, cache_key in pending
                }

                for done, future in enumerate(tqdm(as_completed(futures), total=len(pending)), 1):
                    cache_key = futures[future]
                    try:
                        _, res_json = future.result()
                    except Exception as err:
                        logger.warning(
                            'SimpleVQA judge crashed on index %s: %s: %s',
                            cache_key, type(err).__name__, err)
                        row = data.iloc[row_by_key[cache_key]]
                        res_json = _failed_result(row['question'], row['answer'], row['prediction'])
                    cache[cache_key] = res_json
                    if done % CACHE_DUMP_INTERVAL == 0 or done == len(pending):
                        dump_judge_cache(cache, cache_file)

        # Ensure every requested index exists so scoring never raises KeyError.
        for row_idx, cache_key in enumerate(indices):
            if cache_key not in cache:
                row = data.iloc[row_idx]
                cache[cache_key] = _failed_result(row['question'], row['answer'], row['prediction'])

        num_failed = sum(simplevqa_judge_failed(cache[key]) for key in indices)
        if num_failed:
            logger.warning(
                'SimpleVQA: %d/%d samples could not be judged; they are counted as failed '
                'and will be retried on the next run.', num_failed, len(indices))

        results = [cache[key] for key in indices]
        scores = SimpleVQAEval(results)
        return results, pd.DataFrame([scores])

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> pd.DataFrame:
        """
        Evaluate SimpleVQA model predictions with an LLM judge.

        Args:
            eval_file: Path to the file containing model predictions
            **judge_kwargs: Judge configuration (model/key/api_base/retry/timeout/nproc),
                resolved by ``run.py`` from ``--judge``, ``--judge-args``,
                ``--judge-base-url`` and ``--judge-key``.

        Returns:
            DataFrame with aggregated evaluation scores
        """
        judge_name = judge_kwargs.setdefault('model', self.DEFAULT_JUDGE_MODEL)
        tmp_file = get_judge_cache_file(eval_file, 'eval', judge_name)
        detail_file = get_judge_detail_file(eval_file, 'eval', judge_name, 'json')
        score_file = get_judge_score_file(eval_file, judge_name, 'csv')

        results, score = self.get_scores(eval_file, cache_file=tmp_file, **judge_kwargs)
        file.dump(results, detail_file)

        file.dump(score, score_file)
        return score
