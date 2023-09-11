from transformers import AutoTokenizer
import pandas as pd

# load tokenizer
transformer_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(transformer_name)

# tokenize text
text = 'I am the walrus.'
output = tokenizer(text)

# display results
print(pd.DataFrame(
[output.tokens(), output.word_ids(), output.input_ids],
index=['tokens', 'word_ids', 'input_ids'],
))