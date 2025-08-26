#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
from typing import Any, Dict, List

import fastchat.model
import torch
import transformers

# This special token is supposed to appear in the target of training instances. The loss function
# doesn't take it into consideration.
IGNORE_TOKEN_ID = transformers.trainer_pt_utils.LabelSmoother.ignore_index


def _sharegpt_conversation_to_qwen_prompt(conversation: List[Dict[str, Any]]) -> str:
    """
    Conversation example:
    [
        {"from": "human", "value": "How are you?"},
        {"from": "gpt", "value": "I'm good. How are you?"},
        {"from": "human", "value": "I am good too."}
    ]

    Converts to:
    "<|im_start|>user
    How are you?<|im_end|>
    <|im_start|>assistant
    I'm good. How are you?<|im_end|>
    <|im_start|>user
    I am good too.<|im_end|>"
    """
    prompt = ""
    for msg in conversation:
        if msg["from"] == "human":
            role = "user"
        elif msg["from"] == "gpt":
            role = "assistant"
        else:
            raise ValueError(f"Unknown role: {msg['from']}")

        prompt += f"<|im_start|>{role}\n{msg['value']}<|im_end|>\n"
    return prompt


def _create_labels(
    prompt: str,
    input_ids: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
) -> torch.Tensor:
    """Construct labels given the prompt string and related token IDs. For the following example
    prompt, it returns a labels tensor, where ignorable tokens are masked -100.

    "<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\nI'm good. How are you?<|im_end|>\n"
    [-100        ...                                            -100, 46, 47, 48, 49, 50, 200, 114 ...]

    """
    labels = input_ids.clone()
    labels[:] = IGNORE_TOKEN_ID  # Mask everything by default

    # Tokenize the prompt into conversation turns
    # Each turn is like: "<|im_start|>user\nHow are you?<|im_end|>\n"
    # or "<|im_start|>assistant\nI'm good.<|im_end|>\n"
    turns = prompt.split("<|im_end|>\n")
    cur_pos = 0

    for turn in turns:
        turn = turn.strip()
        if not turn:
            continue
        if "<|im_start|>assistant" in turn:
            # Find the start and end of the assistant's response in tokens
            # Tokenize the turn up to and including "assistant\n"
            prefix = "<|im_start|>assistant\n"
            prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
            turn_ids = tokenizer(turn, add_special_tokens=False).input_ids
            response_start = cur_pos + len(prefix_ids)
            response_end = cur_pos + len(turn_ids)
            # Unmask only the assistant's response tokens
            labels[response_start:response_end] = input_ids[response_start:response_end]
        # Move cur_pos forward by the number of tokens in this turn
        turn_ids = tokenizer(turn + "<|im_end|>\n", add_special_tokens=False).input_ids
        cur_pos += len(turn_ids)

    return labels


def sharegpt_record_to_qwen_training_instance(
    sharegpt_record: Dict[str, Any],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict[str, torch.Tensor]:
    """Given a conversation, outputs a training instance into which the conversation is
    converted.

    Returns:

      A dictionary of tokenized inputs, labels, and attention mask.

    """
    conversation = sharegpt_record["conversations"]
    prompt = _sharegpt_conversation_to_qwen_prompt(conversation)
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids[0]

    labels = _create_labels(prompt, input_ids, tokenizer)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": input_ids.ne(tokenizer.pad_token_id),
    }


def convert_alpaca_to_sharegpt(d: Dict[str, str]) -> Dict[str, Any]:
    return {
        "id": "0",
        "conversations": [
            {"from": "human", "value": d["instruction"]},
            {"from": "gpt", "value": d["output"]},
        ],
    }
