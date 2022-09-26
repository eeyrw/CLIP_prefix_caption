from transformers import GPT2TokenizerFast,GPT2LMHeadModel
import torch

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
a = torch.tensor(tokenizer.encode("OSError: Windows requires Developer."))
gpt = GPT2LMHeadModel.from_pretrained('gpt2')
embedding_text = gpt.transformer.wte(a)
print(embedding_text.shape)