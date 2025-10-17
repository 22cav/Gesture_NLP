import torch
from torch import Tensor
from jaxtyping import Float
from transformers import PreTrainedTokenizerBase, AutoModelForCausalLM
from Gloss_LLM.constants import MODEL_ID, GLOSSES
from typing import Set
from Gloss_LLM.constants import DTYPE, DEVICE_MAP
from Gloss_LLM.gloss_encoding import encode_gloss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions as CausalLMOutput
from transformers import PreTrainedModel

def allowed_sequences(glosses: list[str],tok: PreTrainedTokenizerBase)->Set[tuple[int, ...]]:
    """
    Compute all the allowed sequences for the glosses.
    For a gloss we allow "gloss" and " gloss".
    """
    allowed_sequences: Set[tuple[int, ...]] = set()
    for w in glosses:
        ids_first = tuple(encode_gloss(w,tok))
        ids_after = tuple(encode_gloss(" " + w,tok))
        allowed_sequences.add(ids_first)
        allowed_sequences.add(ids_after)
    return allowed_sequences

def allowed_token_ids(glosses: list[str],tok: PreTrainedTokenizerBase,EOS: int)->tuple[list[int],Set[tuple[int, ...]]]:
    """
    Compute the allowed token IDs for the glosses.
    """
    sequences = allowed_sequences(glosses,tok)
    gloss_ids = sorted({tid for seq in sequences for tid in seq})
    return sorted(set(gloss_ids) | {EOS}), sequences


def token_str(tid: int,EOS: int,tok: PreTrainedTokenizerBase) -> str:
    """
    Convert a token ID to a string.
    """
    if tid == EOS:
        return "<eos>"
    # convert_ids_to_tokens montre le préfixe d'espace explicitement; decode([tid]) donne le rendu textuel
    s :str= tok.convert_ids_to_tokens(tid)
    # Affiche aussi la variante décodée (utile pour vérifier les espaces)
    try:
        d = tok.decode([tid])
        if d != s:
            s = f"{s} | {repr(d)}"
    except Exception:
        pass
    return s

def print_prob_table(ids:list[int], probs:Float[Tensor, "1 vocab"], EOS: int,tok: PreTrainedTokenizerBase, top_k:int|None=None)->list[tuple[int, float]]:
    """
    Print the probability table for the next gloss.
    """
    # ids: list of allowed IDs, probs: tensor [vocab] already softmaxed after masking
    rows :list[tuple[int, float]] = []
    for tid in ids:
        rows.append((tid, probs[tid].item()))
    rows.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        rows = rows[:top_k]
    width = max(len(token_str(t,EOS,tok)) for t, _ in rows) if rows else 10
    for tid, p in rows:
        print(f"{token_str(tid,EOS,tok):<{width}}  {p: .6f}")
    return rows


def compute_next_gloss_probability(gloss_sequence: list[str],glosses: list[str],tok: PreTrainedTokenizerBase,\
                                   model: PreTrainedModel,EOS: int,device: torch.device=torch.device("cpu"))->\
                            tuple[Float[Tensor, "1 vocab"],list[int],dict[str, Float[Tensor, "1 seq_len int"]]]:
    """
    Compute the probabilities for the next gloss given a gloss_sequence.
    """

    inputs :dict[str, Float[Tensor, "1 seq_len", torch.long]] = tok(gloss_sequence, return_tensors="pt").to(device)
    # computes the logits for the next token
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[:, -1, :]
        logits = logits.float()

    # masks the logits for the non-allowed tokens
    allow_token_ids, allowed_sequences = allowed_token_ids(glosses,tok,EOS)
    mask = torch.full_like(logits, float("-inf"))
    mask[0, allow_token_ids] = 0.0
    masked_logits = logits + mask

    # numerical stability
    masked_logits = torch.nan_to_num(masked_logits, nan=-1e9, posinf=1e9, neginf=-1e9)

    # stable softmax (subtract max)
    masked_logits = masked_logits - masked_logits.max(dim=-1, keepdim=True).values
    probs :Float[Tensor, "1 vocab"] = torch.softmax(masked_logits, dim=-1)
    return probs, allow_token_ids, inputs


   
def get_probability_of_a_gloss(gloss: str,tok: PreTrainedTokenizerBase,probs: Float[Tensor, "1 vocab"])->float:
    """
    Get the probability of a gloss.
    """
    gloss_ids = encode_gloss(gloss,tok)
    if len(gloss_ids) != 1:
        return 0.0
    else:
        return probs[0, gloss_ids].item()

def prompt_testing(prompt:list[str])-> None:
    tok = PreTrainedTokenizerBase.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE_MAP)
    model.eval()
    EOS = tok.eos_token_id

    device = model.device
    probs, allowed_token_ids, inputs = compute_next_gloss_probability(prompt,GLOSSES,tok,model,EOS,device)



    print("=== Next-token probabilities (over GLOSSES set) ===")
    rows = print_prob_table(allowed_token_ids, probs[0], EOS,tok, top_k=None)  # mets top_k=20 si tu veux tronquer

    # ------------ Échantillonnage contraint (optionnel) ------------
    # On ne prélève que sur les IDs autorisés (distribution déjà renormalisée)
    allowed_tensor = torch.tensor(allowed_token_ids, device=probs.device)
    allowed_probs = probs[0, allowed_tensor]
    next_index_in_allowed = torch.multinomial(allowed_probs, num_samples=1)
    next_tid = allowed_tensor[next_index_in_allowed].item()
    next_prob = allowed_probs[next_index_in_allowed].item()

    print("\nSampled next token:")
    print(f"  id={next_tid}, tok={token_str(int(next_tid),EOS,tok)}, p={next_prob:.6f}")

    # ------------ Concat et affichage de la séquence étendue ------------
    new_input_ids = torch.cat([inputs["input_ids"], torch.tensor([[next_tid]], device=inputs["input_ids"].device)], dim=-1)
    print("\nDecoded so far:")
    print(tok.decode(new_input_ids[0], skip_special_tokens=True))

if __name__ == "__main__":
    prompt_testing(prompt=["me", "want", "eat"])