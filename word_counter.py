# Jonathan Shaki, Or Shachar 204920367, 209493709

from typing import Dict, List


def word_appearances(words: List[str]) -> Dict[str, int]:
    word_count_dict = {}
    for word in words:
        if word not in word_count_dict:
            word_count_dict[word] = 0
        word_count_dict[word] = word_count_dict[word] + 1
    return word_count_dict


def merge_dicts(x: Dict[str, int], y: Dict[str, int]) -> Dict[str, int]:
    return {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}


def word_appearances_in_cluster(cluster: List[List[str]]) -> Dict[str, int]:
    target = {}
    for document in cluster:
        word_count_document = word_appearances(document)
        target = merge_dicts(target, word_count_document)

    return target
