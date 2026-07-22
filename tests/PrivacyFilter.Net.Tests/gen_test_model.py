import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parent / "models" / "tiny"
ONNX_DIR = ROOT / "onnx"

LABELS = ["O"]
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
    LABELS.extend(f"{boundary}-{label}" for boundary in ("B", "I", "E", "S"))


def main() -> None:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
    )
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["batch_size", "sequence_length"]
    )
    logits = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, ["batch_size", "sequence_length", len(LABELS)]
    )

    bias = np.full((1, 1, len(LABELS)), -10.0, dtype=np.float32)
    bias[0, 0, LABELS.index("S-private_person")] = 10.0
    nodes = [
        helper.make_node("Shape", ["input_ids"], ["input_shape"]),
        helper.make_node(
            "Constant",
            [],
            ["class_count"],
            value=numpy_helper.from_array(np.array([len(LABELS)], dtype=np.int64)),
        ),
        helper.make_node("Concat", ["input_shape", "class_count"], ["output_shape"], axis=0),
        helper.make_node(
            "ConstantOfShape",
            ["output_shape"],
            ["zeros"],
            value=numpy_helper.from_array(np.array([0.0], dtype=np.float32)),
        ),
        helper.make_node(
            "Constant",
            [],
            ["bias"],
            value=numpy_helper.from_array(bias),
        ),
        helper.make_node("Add", ["zeros", "bias"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "privacy-filter-test-model",
        [input_ids, attention_mask],
        [logits],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    onnx.save(model, ONNX_DIR / "model.onnx")

    config = {
        "architectures": ["OpenAIPrivacyFilterForTokenClassification"],
        "encoding": "o200k_base",
        "id2label": {str(index): label for index, label in enumerate(LABELS)},
        "label2id": {label: index for index, label in enumerate(LABELS)},
        "num_labels": len(LABELS),
    }
    (ROOT / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    calibration = {
        "operating_points": {
            "default": {
                "biases": {
                    "transition_bias_background_stay": 0.0,
                    "transition_bias_background_to_start": 0.0,
                    "transition_bias_inside_to_continue": 0.0,
                    "transition_bias_inside_to_end": 0.0,
                    "transition_bias_end_to_background": 0.0,
                    "transition_bias_end_to_start": 0.0,
                }
            }
        }
    }
    (ROOT / "viterbi_calibration.json").write_text(
        json.dumps(calibration, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
