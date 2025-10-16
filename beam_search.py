from typing import List, Dict, Tuple
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from Gloss_LLM.constants import GLOSSES
from Gloss_LLM.resized_vocabulary import get_probability_of_a_gloss
ProbabilityDistribution = Dict[str, float]
from Gloss_LLM.constants import MODEL_ID, DTYPE, DEVICE_MAP
from Gloss_LLM.resized_vocabulary import print_prob_table
from Gloss_LLM.resized_vocabulary import compute_next_gloss_probability
def from_dic_sequence_to_list_sequence(gloss_sequence: List[ProbabilityDistribution],log_proba: bool = False) -> List[List[Tuple[str, float]]]:
    """
    Convert a list of dictionary sequence to a list of list sequence.
    """
    gloss_sequence_log: List[List[Tuple[str, float]]] = []
    for slot in gloss_sequence:
        items = slot.items()
        step = []
        for tok, p in items:
            if log_proba:
                logp = -math.inf if p <= 0.0 else math.log(float(p))
            else:
                logp = -math.inf if p <= 0.0 else p
            step.append((tok, logp))
        gloss_sequence_log.append(step)
    return gloss_sequence_log

def beam_search_slots_given_probability_distribution(gloss_sequence: List[ProbabilityDistribution],beam_size: int = 4,n_best: int = 5) -> List[Tuple[List[str], float, float]]:
    """
    Beam search for len(gloss_sequence) slots.
    - beam_size : amount of sequences to keep
    - n_best : amount of sequences to return
    Return : list of (sequence[str], score_log, proba)
    """

    gloss_sequence_log = from_dic_sequence_to_list_sequence(gloss_sequence,log_proba=True)

    # the beam_size best sequences
    best_sequences: List[Tuple[List[str], float]] = [([], 0.0)]  # (sequence, sum_logp)

    # We look into each slot of the gloss sequence
    for slot in gloss_sequence_log:
        candidates: List[Tuple[List[str], float]] = []
        # We look into each best sequence
        for seq, score in best_sequences:
            # For each best sequence, we add the new token to the sequence and compute the new score
            for tok, logp in slot:
                # As we work with log, we add the log probability (same as multiplying the probabilities)
                candidates.append((seq + [tok], score + logp))
        # Computes the top-k new sequences
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_sequences = candidates[:beam_size]

    # Computes the top-n best sequences
    best_sequences.sort(key=lambda x: x[1], reverse=True)

    out = []
    for seq, score in best_sequences[:n_best]:
        # As we work with log, we need to convert the score to a probability
        out.append((seq, score, math.exp(score)))
    return out


def beam_search_slots_with_llm(gloss_sequence: List[ProbabilityDistribution],model: AutoModelForCausalLM,tokenizer: AutoTokenizer, beam_size: int = 4,n_best: int = 5,EOS: int = 1000,device: str = "cpu") -> List[Tuple[List[str], float, float]]:
    """
    Beam search for len(gloss_sequence) slots.
    - beam_size : amount of sequences to keep
    - n_best : amount of sequences to return
    Return : list of (sequence[str], score_log, proba)
    """
    gloss_sequence_list = from_dic_sequence_to_list_sequence(gloss_sequence,log_proba=False)

    best_sequences: List[Tuple[List[str], float]] = [([], 0.0)]  # (sequence, sum_logp)
    slot_index = 0
    for slot in gloss_sequence_list:
        candidates: List[Tuple[List[str], float]] = []
        # We look into each best sequence
        for seq, score in best_sequences:
            # if the index is 0 we use the probability distribution from the CV model
            if slot_index == 0:
                for token, proba in slot:
                    candidates.append((seq + [token], score + math.log(proba)))
            else:
                # if the index is not 0 we use the probability distribution from the LLM model
                llm_probabilities, allowed_token_ids, inputs = compute_next_gloss_probability(seq,GLOSSES,tokenizer,model,EOS,device)
                # we compute the probability of each token in the slot
                for token, proba in slot:
                    llm_proba = get_probability_of_a_gloss(token,tokenizer,llm_probabilities)
                    if llm_proba == 0.0:
                        candidates.append((seq + [token], -math.inf))
                    else:
                        candidates.append((seq + [token], score + math.log(llm_proba)))
            
        # Computes the top-k new sequences
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_sequences = candidates[:beam_size]
        if slot_index == 1:
            print(best_sequences)
        slot_index += 1
    
    # Computes the top-n best sequences
    best_sequences.sort(key=lambda x: x[1], reverse=True)

    out = []
    for seq, score in best_sequences[:n_best]:
        # As we work with log, we need to convert the score to a probability
        out.append((seq, score, math.exp(score)))
    return out




# Example
if __name__ == "__main__":
    slots = [
        {"i": 0.4, "you": 0.25, "we": 0.15, "they": 0.1, "he": 0.06, "she": 0.04},
        {"want": 0.5, "like": 0.2, "need": 0.15, "eat": 0.1, "go": 0.03, "watch": 0.02},
        {"a": 0.45, "the": 0.25, "some": 0.15, "to": 0.1, "of": 0.03, "my": 0.02},
        {"pizza": 0.35, "movie": 0.25, "restaurant": 0.2, "water": 0.1, "ring": 0.06, "airplane": 0.04},
        {"now": 0.3, "today": 0.25, "please": 0.2, "soon": 0.15, "outside": 0.06, "house": 0.04},
    ]

    tops = beam_search_slots_given_probability_distribution(slots, beam_size=4, n_best=3)
    for i, (seq, s_log, p) in enumerate(tops, 1):
        print(f"{i:>2}. {' '.join(seq):50s}  logP={s_log:.4f}  P={p:.6f}")
    
    
    
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE_MAP)
    model.eval()
    EOS = tok.eos_token_id

    device = model.device

    tops = beam_search_slots_with_llm(slots,model,tok,beam_size=4,n_best=3,EOS=EOS,device=device)
    for i, (seq, s_log, p) in enumerate(tops, 1):
        print(f"{i:>2}. {' '.join(seq):50s}  logP={s_log:.4f}  P={p:.6f}")