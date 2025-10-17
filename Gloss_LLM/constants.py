# Constants
import torch
MODEL_ID = "Qwen/Qwen3-0.6B"
GLOSSES=["a","act","also","learn","leave","like","live","love","make","man","many","me","meet","america","mom","money","moon",
        "more","move","movie","must","my","name","and","need","never","new","night","no","not","now","on","one","open","angry",
        "our","out","pay","people","place","play","please","read","ready","right","animal","run","sad","say","school","see","she","sick","sign",
        "sign language","sit","answer","sleep","small","some","something","sometimes","speak","start","stay","stop","area","study","support","sure",
        "take","talk","teach","tell","thank you","that","their","art","them","then","there","they","think","this","time","tired","to","today","ask",
        "together","true","try","under","understand","up","very","wait","walk","want","asl","watch","water","we","wear","what","when","where","which",
        "white","who","action","baby","why","will","win","with","without","woman","word","work","worry","write","backpack","wrong","yes","yesterday",
        "you","your","bad","baseball","beautiful","because","bed","before","big","birthday","activity","book","boy","break","breakfast","business","but",
        "buy","bye","call","camera","adult","can","car","cat","child","city","clean","clock","college","come","company","after","computer","cook","dad",
        "day","different","difficult","dinner","doctor","again","dog","done","door","drink","drive","eat","education","end","enough","family","age","fast",
        "father","feel","find","finish","food","for","friend","from","game","airplane","get","girl","give","go","good","goodbye","happy","have","hello","help",
        "all","here","home","hot","house","how","i","idea","if","in","know","."]
DTYPE = torch.bfloat16  # ou torch.float16 / torch.float32 selon ton HW
DEVICE_MAP = "auto"