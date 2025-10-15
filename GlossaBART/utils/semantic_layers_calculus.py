import torch
import numpy as np
import math
from datas.data import load_data, DEFAULT_MODEL

def compute_attention_metrics(attn_layers, keep_idx, eps: float = 1e-12):
    """
    Compute off-diagonal ratio and entropy norm for each layer.
    They are the metrics used to know which layers are the most important for the semantic aspect.
    off-diagonal ratio : how much of the attention attends to other tokens than the current one.
    entropy norm : how much of the attention is spread across the tokens once we ignore self-attention.
    
    Best layers : 3,4,5
    """

    L = len(attn_layers)
    metrics = {"layer": [], "offdiag_ratio": [], "entropy_norm": []}
    Teff = len(keep_idx)

    for ell in range(L):
        A_l = attn_layers[ell][0]   # (H, T, T)
        H = A_l.shape[0]

        offdiag_vals = []
        entropy_vals = []

        for h in range(H):
            A = A_l[h]
            # restrict to kept tokens
            A = A.index_select(0, keep_idx).index_select(1, keep_idx)  # (Teff, Teff)
            if Teff < 2:
                continue

            # --- normalize row-wise ---
            row_sums = A.sum(dim=-1, keepdim=True).clamp_min(eps)
            A_norm = A / row_sums

            # --- off-diagonal mass ratio ---
            diag_mean = torch.diagonal(A_norm).mean().item()
            offdiag_vals.append(1.0 - diag_mean)

            # --- context-only entropy ---
            A_no_diag = A.clone()
            A_no_diag.fill_diagonal_(0.0)
            row_sums_ctx = A_no_diag.sum(dim=-1, keepdim=True)
            valid_rows = (row_sums_ctx.squeeze(-1) > eps)
            if valid_rows.any():
                P_ctx = torch.zeros_like(A_no_diag)
                P_ctx[valid_rows] = A_no_diag[valid_rows] / row_sums_ctx[valid_rows]
                P_safe = P_ctx.clamp_min(eps)
                H_rows = -(P_safe * torch.log(P_safe)).sum(dim=-1)
                entropy_vals.append(H_rows[valid_rows].mean().item())
            else:
                entropy_vals.append(0.0)

        if len(offdiag_vals) > 0:
            offdiag_mean = float(np.mean(offdiag_vals))
            entropy_mean = float(np.mean(entropy_vals))
        else:
            offdiag_mean, entropy_mean = float("nan"), float("nan")

        entropy_norm = entropy_mean / math.log(max(1, Teff - 1)) if Teff > 1 else float("nan")

        metrics["layer"].append(ell)
        metrics["offdiag_ratio"].append(offdiag_mean)
        metrics["entropy_norm"].append(entropy_norm)

    return metrics


def build_keep_indices(tokenizer, inputs, drop_first: bool = True) -> torch.LongTensor:
    """
    Build the indices to keep: non-padding, non-special,
    puis (optionnel) retire le premier token réel.
    """
    input_ids = inputs["input_ids"][0]
    attn_mask = inputs["attention_mask"][0].bool()
    special_ids = set(tokenizer.all_special_ids)
    non_special = torch.tensor([tid.item() not in special_ids for tid in input_ids], dtype=torch.bool)
    keep_mask = attn_mask & non_special
    keep_idx = torch.where(keep_mask)[0]
    if drop_first and keep_idx.numel() >= 2:
        keep_idx = keep_idx[1:]
    if keep_idx.numel() < 2:
        raise ValueError("Not enough useful tokens after filtering.")
    return keep_idx

def consensus_best_layers(gloss_texts: list[str],
                          tokenizer, model, device: str,
                          n_layers: int = 3,
                          drop_first: bool = True) -> list[int]:
    """
    Selection by vote: for each sentence, we compute the score (ratio+entropy_norm),
    we rank the layers; we cumulate the places (Borda-like) and we take the best ones.
    """ 
    ranks_sum = None
    L_ref = None

    for text in gloss_texts:
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, return_dict=True)
        attn_layers = outputs.encoder_attentions
        if attn_layers is None:
            raise RuntimeError("encoder_attentions=None. Utilise attn_implementation='eager'.")

        keep_idx = build_keep_indices(tokenizer, inputs, drop_first=drop_first)
        m = compute_attention_metrics(attn_layers, keep_idx)

        R = np.nan_to_num(np.array(m["offdiag_ratio"]), nan=0.0)
        E = np.nan_to_num(np.array(m["entropy_norm"]),  nan=0.0)
        score = R + E

        order = np.argsort(score)[::-1]  # better -> worse
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))  # rank 0 = better

        if L_ref is None:
            L_ref = len(order)
            ranks_sum = ranks.astype(float)
        else:
            if len(order) != L_ref:
                raise ValueError("Inconsistent number of layers between samples.")
            ranks_sum += ranks

    # smallest cumulative rank = best
    best = np.argsort(ranks_sum)[:n_layers]
    return best.tolist()


if __name__ == "__main__":
    tokenizer, model, device = load_data()
    gloss_texts = ["PERSON WANT BUY", "PERSON WANT BUY CAR", "PERSON WANT BUY DIE", "PERSON WANT BUY DIE CAR"]
    best_layers = consensus_best_layers(gloss_texts, tokenizer, model, device, n_layers=3, drop_first=True)
    print("Best layers (consensus vote):", best_layers)
