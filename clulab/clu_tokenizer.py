# https://github.com/clulab/scala-transformers/blob/acba0224358d9935174b607712a28fb5c85a7b74/encoder/src/main/python/clu_tokenizer.py#L4
from clulab.parameters import Parameters
from clulab.names import Names
from transformers import AutoTokenizer

class CluTokenizer:
    @classmethod
    def get_pretrained(cls, name: str = Parameters.transformer_name) -> AutoTokenizer:
        # add_prefix_space is needed only for the roberta tokenizer
        add_prefix_space = "roberta" in name.lower()
        # which transformer to use
        print(f"Loading tokenizer named \"{name}\" with add_prefix_space={add_prefix_space}...")
        tokenizer = AutoTokenizer.from_pretrained(name, model_input_names=[Names.INPUT_IDS, "token_type_ids", "attention_mask"], add_prefix_space=add_prefix_space)
        return tokenizer