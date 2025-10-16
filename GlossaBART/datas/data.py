import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Glossa-BART model
DEFAULT_MODEL = "rrrr66254/Glossa-BART"

def load_data(model_name: str = DEFAULT_MODEL)->tuple[BartTokenizer, BartForConditionalGeneration, str]:
    """
    Loads the model and tokenizer. 
    Uses GPU if available.
    """
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name, attn_implementation="eager",)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device
