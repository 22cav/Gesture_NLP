"""
Translate a gloss sequence into a spoken language sentence.
"""
import os
import sys
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Glossa-BART model
DEFAULT_MODEL = "rrrr66254/Glossa-BART"
# Gloss sentence
DEFAULT_GLOSS_SENTENCE = "ME WANT BUY CAR"

def load_model(model_name: str = DEFAULT_MODEL):
    """
    Loads the model and tokenizer. 
    Uses GPU if available.
    """
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

@torch.no_grad()
def translate_gloss(gloss_sentence: str,tokenizer: BartTokenizer,model: BartForConditionalGeneration,device: str = "cpu",
                    max_new_tokens: int = 40, num_beams: int = 5, length_penalty: float = 1.0, repetition_penalty: float = 1.1):
    """
    Translates a gloss sentence into a spoken language sentence using Bart model or one of it's fine tuned versions.
    Uses beam search to generate the best sentence.
    """
    # Prepare the tensors
    enc = tokenizer(gloss_sentence, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    # Generation (autoregressive decoding)
    # Uses beam search to generate the best sentence.
    outputs = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        early_stopping=True,
        no_repeat_ngram_size=3
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def main():
    # 1) Get the gloss sentence (default value)
    gloss_sentence = DEFAULT_GLOSS_SENTENCE
    if not gloss_sentence:
        gloss_sentence = DEFAULT_GLOSS_SENTENCE

    # 2) Load model and tokenizer
    tokenizer, model, device = load_model(DEFAULT_MODEL)

    # 3) Translation
    translation = translate_gloss(gloss_sentence, tokenizer, model, device)

    # 4) Output
    print("Input (GLOSS):", gloss_sentence)
    print("Output (spoken language):", translation)

if __name__ == "__main__":
    main()
