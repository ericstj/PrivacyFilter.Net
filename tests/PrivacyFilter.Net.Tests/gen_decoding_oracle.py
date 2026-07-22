import json
import random
from pathlib import Path

import numpy as np
import torch
import tiktoken

from opf._api import _redact_text
from opf._core.decoding import ViterbiCRFDecoder
from opf._core.runtime import (
    DetectedSpan,
    _apply_output_mode_to_detected_spans,
    _label_placeholder,
    _select_non_overlapping_spans,
)
from opf._core.sequence_labeling import build_label_info
from opf._core.spans import (
    decode_text_with_offsets,
    discard_overlapping_spans_by_label,
    labels_to_spans,
    token_spans_to_char_spans,
    trim_char_spans_whitespace,
)


ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260721)

CLASS_NAMES = ["O"]
for label in (
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
):
    CLASS_NAMES.extend(f"{boundary}-{label}" for boundary in ("B", "I", "E", "S"))

BIASES = {
    "transition_bias_background_stay": 0.125,
    "transition_bias_background_to_start": -0.25,
    "transition_bias_inside_to_continue": 0.375,
    "transition_bias_inside_to_end": -0.125,
    "transition_bias_end_to_background": 0.25,
    "transition_bias_end_to_start": -0.375,
}

RANDOM_PIECES = [
    "Alice",
    " ",
    "\t",
    "\n",
    ",",
    ".",
    "-",
    "é",
    "e\u0301",
    "中",
    "😀",
    "👩🏽‍💻",
    "alice@example.com",
    "+1 (425) 555-0100",
]


def utf16_index(text: str, codepoint_index: int) -> int:
    return len(text[:codepoint_index].encode("utf-16-le")) // 2


def main() -> None:
    label_info = build_label_info(CLASS_NAMES)
    decoder = ViterbiCRFDecoder(label_info, **BIASES)

    viterbi_cases = []
    lengths = [1, 2, 3, 4, 8, 16, 31]
    lengths.extend(int(value) for value in RNG.integers(1, 40, size=57))
    for length in lengths:
        scores = RNG.normal(0.0, 3.0, size=(length, len(CLASS_NAMES))).astype(
            np.float32
        )
        if length >= 2:
            scores[0, CLASS_NAMES.index("I-private_person")] = np.float32(12.0)
            scores[-1, CLASS_NAMES.index("B-private_email")] = np.float32(12.0)
        expected = decoder.decode(torch.from_numpy(scores))
        viterbi_cases.append(
            {
                "scores": scores.tolist(),
                "expected": expected,
            }
        )

    span_cases = []
    lengths = [0, 1, 2, 3, 4, 8, 16, 32]
    lengths.extend(int(value) for value in RNG.integers(0, 50, size=120))
    for length in lengths:
        labels = RNG.integers(0, len(CLASS_NAMES), size=length).astype(int).tolist()
        spans = labels_to_spans(
            {index: label for index, label in enumerate(labels)},
            label_info,
        )
        span_cases.append(
            {
                "labels": labels,
                "expected": [list(span) for span in spans],
            }
        )

    encoding = tiktoken.get_encoding("o200k_base")
    text_rng = random.Random(20260721)
    postprocess_cases = []
    for case_index in range(64):
        text = "".join(
            text_rng.choice(RANDOM_PIECES)
            for _ in range(text_rng.randrange(1, 30))
        )
        token_ids = encoding.encode(text, allowed_special="all")
        labels = RNG.integers(0, len(CLASS_NAMES), size=len(token_ids)).astype(int).tolist()
        trim_whitespace = case_index % 2 == 0
        discard_overlaps = case_index % 3 == 0
        output_mode = "redacted" if case_index % 4 == 0 else "typed"

        token_spans = labels_to_spans(
            {index: label for index, label in enumerate(labels)},
            label_info,
        )
        decoded_text, starts, ends = decode_text_with_offsets(token_ids, encoding)
        char_spans = token_spans_to_char_spans(token_spans, starts, ends)
        if trim_whitespace:
            char_spans = trim_char_spans_whitespace(char_spans, decoded_text)
        if discard_overlaps:
            char_spans = discard_overlapping_spans_by_label(char_spans)

        detected = []
        for label_index, start, end in char_spans:
            label = label_info.span_class_names[label_index]
            detected.append(
                DetectedSpan(
                    label=label,
                    start=start,
                    end=end,
                    text=decoded_text[start:end],
                    placeholder=_label_placeholder(label),
                )
            )
        display_spans = _apply_output_mode_to_detected_spans(
            _select_non_overlapping_spans(detected),
            output_mode=output_mode,
        )
        postprocess_cases.append(
            {
                "text": text,
                "labels": labels,
                "trimWhitespace": trim_whitespace,
                "discardOverlaps": discard_overlaps,
                "outputMode": output_mode,
                "expectedSpans": [
                    {
                        "label": span.label,
                        "start": utf16_index(decoded_text, span.start),
                        "end": utf16_index(decoded_text, span.end),
                        "text": span.text,
                        "placeholder": span.placeholder,
                    }
                    for span in display_spans
                ],
                "redactedText": _redact_text(decoded_text, tuple(display_spans)),
            }
        )

    payload = {
        "classNames": CLASS_NAMES,
        "biases": BIASES,
        "viterbiCases": viterbi_cases,
        "spanCases": span_cases,
        "postprocessCases": postprocess_cases,
    }
    (ROOT / "decoding_oracle.json").write_text(
        json.dumps(payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
