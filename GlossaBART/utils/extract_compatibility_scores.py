import torch
import matplotlib.pyplot as plt
from transformers import BartTokenizer, BartModel
from datas.data import DEFAULT_MODEL

def compatibility_score(gloss_sequence: str, new_gloss: str, model: BartModel, tokenizer: BartTokenizer, device: str = "cpu",
                        selected_layers: list[int] = [3, 4, 5]):
    """
    Compute the compatibility score for a new gloss.
    """
    # Ne pas redéfinir device ici, utiliser celui passé en paramètre
    model.to(device)
    model.eval()
    full_sequence = gloss_sequence + " " + new_gloss
    inputs = tokenizer(full_sequence, return_tensors="pt", add_special_tokens=True)
    
    # Déplacer les tenseurs d'entrée vers le même périphérique que le modèle
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        attn_layers = outputs.encoder_attentions  # [batch, heads, T, T]
    valid_layers = [layer[0] for i, layer in enumerate(attn_layers) if i in selected_layers]
    if not valid_layers:
        raise ValueError("No valid layers selected. Check layer indices or model depth.")

    attn_stack = torch.stack(valid_layers, dim=0)  # (len(selected_layers), H, T, T)
    attn_mean = attn_stack.mean(dim=(0, 1))  # average over selected layers + all heads → (T, T)

    # keep tokens
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    T = attn_mean.shape[0]

    # mask padding
    attn_mask = inputs["attention_mask"][0].bool()
    # mask special tokens
    special_ids = set(tokenizer.all_special_ids)
    special_mask = torch.tensor([tid not in special_ids for tid in input_ids], dtype=torch.bool, device=device)

    # combine (non pad + non special)
    keep_mask = attn_mask & special_mask
    keep_idx = torch.where(keep_mask)[0]

    # remove first useful token
    if len(keep_idx) <= 2:
        raise ValueError("Pas assez de tokens non spéciaux pour analyse.")
    keep_idx = keep_idx[1:]  # remove first real token

    # corresponding attention matrix
    A = attn_mean.index_select(0, keep_idx).index_select(1, keep_idx)
    tokens_kept = [tokens[i] for i in keep_idx.tolist()]
    T_eff = A.shape[0]

    # clean up
    A = A.clone()
    A.fill_diagonal_(0.0)  # remove self-attention

    # index of new gloss
    idx_new = T_eff - 1  # last non-special, non-initial token

    # directional compatibility
    forward = A[idx_new, :idx_new]
    backward = A[:idx_new, idx_new]

    compat_forward = forward.mean().item() if forward.numel() > 0 else float("nan")
    compat_backward = backward.mean().item() if backward.numel() > 0 else float("nan")
    return compat_forward, compat_backward







# visualizations
# a) Attention from new gloss to previous
"""
# centrality score
centrality = A.sum(dim=0)  # column sum
centrality_norm = (centrality / (centrality.sum() + 1e-12)).cpu().numpy()

plt.figure(figsize=(7, 3.2))
plt.bar(tokens_kept[:idx_new], forward.cpu().numpy() if forward.numel() > 0 else [])
plt.title(f"Attention (new gloss → previous): {tokens_kept[idx_new]}")
plt.ylabel("Attention weight")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# b) Centrality (global importance received)
plt.figure(figsize=(8.5, 3.6))
plt.bar(tokens_kept, centrality_norm)
plt.title("Centrality (col-sum, diag=0, avg. layers+heads, without first token)")
plt.ylabel("Relative weight")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
"""