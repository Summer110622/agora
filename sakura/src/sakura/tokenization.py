"""An explicit fixture tokenizer; pretrained runs always retain their original tokenizer."""
from __future__ import annotations


class FixtureByteTokenizer:
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2

    def decode(self, token_ids, **kwargs):
        return bytes(int(token) - 3 for token in token_ids if 3 <= int(token) < 259).decode("utf-8", errors="replace")

    def encode(self, text, *, add_special_tokens=True):
        tokens = [byte + 3 for byte in text.encode("utf-8")]
        return [self.bos_token_id, *tokens, self.eos_token_id] if add_special_tokens else tokens

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=False,
                            return_dict=False, return_assistant_tokens_mask=False, **kwargs):
        if not tokenize:
            raise ValueError("The fixture tokenizer only supports tokenized output")
        ids, mask = [self.bos_token_id], [0]
        for message in messages:
            prefix = list(f"<{message['role']}>\n".encode("utf-8"))
            content = list(message["content"].encode("utf-8"))
            ids.extend(b + 3 for b in prefix + content)
            mask.extend([0] * len(prefix) + [int(message["role"] == "assistant")] * len(content))
            ids.append(self.eos_token_id)
            mask.append(int(message["role"] == "assistant"))
        if add_generation_prompt:
            prefix = list(b"<assistant>\n")
            ids.extend(b + 3 for b in prefix)
            mask.extend([0] * len(prefix))
        return {"input_ids": ids, "assistant_masks": mask} if return_dict else ids


def load_tokenizer(spec: dict, *, base_path: str | None = None):
    if spec["kind"] == "byte":
        return FixtureByteTokenizer()
    from .artifacts import verify_tokenizer
    verify_tokenizer(spec, base_path or spec["path"])
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(base_path or spec["path"], local_files_only=True,
                                         trust_remote_code=False)
