import json
import random
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parent
ENCODING = tiktoken.get_encoding("o200k_base")
EDGE_CASES = [
    "",
    "Alice's email is alice@example.com.",
    "Bonjour, Привет, こんにちは, 你好.",
    "Emoji: 👩🏽‍💻 and 😀.",
    "Combining: cafe\u0301 and precomposed café.",
    "Line one\r\nLine two\twith spaces.",
    "<|endoftext|>",
]

RANDOM_PIECES = [
    "a",
    "Z",
    "0",
    " ",
    "\t",
    "\n",
    "\r\n",
    ".",
    ",",
    "'",
    "-",
    "_",
    "@",
    "é",
    "e\u0301",
    "ß",
    "Ж",
    "中",
    "日",
    "😀",
    "👩🏽‍💻",
]


def utf16_index(text: str, codepoint_index: int) -> int:
    return len(text[:codepoint_index].encode("utf-16-le")) // 2


def token_offsets(token_ids: list[int]) -> tuple[str, list[int], list[int]]:
    token_bytes = [ENCODING.decode_single_token_bytes(token_id) for token_id in token_ids]
    decoded = b"".join(token_bytes).decode("utf-8", errors="replace")

    char_byte_starts = []
    char_byte_ends = []
    byte_cursor = 0
    for char in decoded:
        char_byte_starts.append(byte_cursor)
        byte_cursor += len(char.encode("utf-8"))
        char_byte_ends.append(byte_cursor)

    starts = []
    ends = []
    token_byte_cursor = 0
    for raw_bytes in token_bytes:
        token_byte_start = token_byte_cursor
        token_byte_end = token_byte_start + len(raw_bytes)
        token_byte_cursor = token_byte_end

        start = 0
        while start < len(char_byte_ends) and char_byte_ends[start] <= token_byte_start:
            start += 1
        end = 0
        while end < len(char_byte_starts) and char_byte_starts[end] < token_byte_end:
            end += 1
        if end < start:
            end = start
        starts.append(utf16_index(decoded, start))
        ends.append(utf16_index(decoded, end))

    return decoded, starts, ends


def main() -> None:
    rng = random.Random(20260721)
    texts = list(EDGE_CASES)
    for _ in range(128):
        texts.append(
            "".join(rng.choice(RANDOM_PIECES) for _ in range(rng.randrange(0, 80)))
        )

    cases = []
    for text in texts:
        ids = ENCODING.encode(text, allowed_special="all")
        decoded, starts, ends = token_offsets(ids)
        cases.append(
            {
                "text": text,
                "ids": ids,
                "decoded": decoded,
                "starts": starts,
                "ends": ends,
            }
        )

    (ROOT / "tokenizer_oracle.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
