"""Run ordinary Laya prediction with an encoder pre-hook for 1 and 10 questions."""
import argparse
import json
import platform

import torch
import transformers

from laya import Agent


def questions(count):
    return {
        "q%d" % index: {
            "type": "noul",
            "instructions": "Does the state contain the marker word marker%d?" % index,
            "criteria": {"false": "the marker is absent", "true": "the marker is present"},
        }
        for index in range(count)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state", default="A short fixed state for inspecting encoder batch shapes.")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run this check on the target T4 runtime.")
    print(json.dumps({
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda": torch.version.cuda,
    }, sort_keys=True))
    agent = Agent(args.model, device=args.device)
    encoder = agent.model.encoder
    config = encoder.config
    print(json.dumps({
        "encoder_class": type(encoder).__name__, "attention": getattr(config, "_attn_implementation", None),
        "layer_types": getattr(config, "layer_types", None),
        "sliding_window": getattr(config, "sliding_window", None),
        "rope_parameters": getattr(config, "rope_parameters", None),
    }, default=str, sort_keys=True))

    def report_inputs(module, positional, keyword):
        del module
        input_ids = keyword.get("input_ids", positional[0] if positional else None)
        attention_mask = keyword.get("attention_mask")
        position_ids = keyword.get("position_ids")
        print("encoder_pre_hook", json.dumps({
            "input_ids": tuple(input_ids.shape) if input_ids is not None else None,
            "attention_mask": tuple(attention_mask.shape) if isinstance(attention_mask, torch.Tensor)
            else type(attention_mask).__name__,
            "position_ids": tuple(position_ids.shape) if position_ids is not None else None,
        }, sort_keys=True))

    handle = encoder.register_forward_pre_hook(report_inputs, with_kwargs=True)
    try:
        for count in (1, 10):
            print("question_count", count)
            agent.predict(args.state, questions(count))
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
