# Jonathan Shaki, Or Shachar 204920367, 209493709

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
