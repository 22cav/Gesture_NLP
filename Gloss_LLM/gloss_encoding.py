from transformers import AutoTokenizer
from typing import List
from Gloss_LLM.constants import GLOSSES
from Gloss_LLM.constants import MODEL_ID

"""
['america', 'angry', 'sick', 'sign language', 'speak', 'teach', 'thank you', 'tired', 'together', 'understand', 
'worry', 'backpack', 'yesterday', 'baseball', 'breakfast', 'difficult', 'dinner', 'enough', 'airplane', 'goodbye']
Is the list of glosses which are encoded into multiple tokens.
20 tokens, so 1/10th of our vocabulary.
"""

def encode_gloss(gloss: str,tok: AutoTokenizer)->list[int]:
    """
    Encode a gloss into a list of IDs.
    """
    token_ids: List[int] = tok.encode(gloss, add_special_tokens=False)
    return token_ids


def encode_all_glosses(glosses: list[str],tok: AutoTokenizer)->list[list[int]]:
    """
    Encode all the glosses into a list of lists of IDs.
    We want to get the glosses which are encoded into multiple tokens.
    """
    encoded_glosses: list[list[int]] = [encode_gloss(gloss,tok) for gloss in glosses]
    encoded_glosses_multiple_tokens: list[str] = [gloss for gloss, ids in zip(glosses, encoded_glosses) if len(ids) > 1]
    return encoded_glosses_multiple_tokens


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    encoded_glosses_multiple_tokens = encode_all_glosses(GLOSSES,tok)
    print(encoded_glosses_multiple_tokens)