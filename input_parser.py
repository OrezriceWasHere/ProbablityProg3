from typing import List, Dict
from items import InputItem, SourceItem


def build_input_cache(filename: str) -> List[InputItem]:
    result = []

    with open(filename) as file:
        buffer = []

        for line in file:
            if line == "\n":
                continue
            if len(buffer) < 2:
                buffer.append(line)
            if len(buffer) == 2:
                item = parse_item(buffer)
                buffer.clear()
                result.append(item)

    return result


def parse_item(buffer):
    header = buffer[0].replace("<", "").replace(">", "").rstrip().split("\t")
    content = buffer[1].rstrip().split(" ")
    source, idd, topics = header[0], header[1], header[2:]

    return InputItem(
        SourceItem.TRAIN if source == SourceItem.TRAIN else SourceItem.TEST,
        int(idd),
        content,
        topics
    )


def count_word_appearances(words: List[str]) -> Dict[str, int]:
    word_count_dict = {}
    for word in words:
        if word not in word_count_dict:
            word_count_dict[word] = 0
        word_count_dict[word] = word_count_dict[word] + 1
    return word_count_dict
