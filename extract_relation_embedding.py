import numpy as np
from transformers import AutoTokenizer, AutoModel
import spacy


def get_relation_vector(tokenizer, model, sentence, items):
    item1_vector = get_averaged_vector(tokenizer, model, sentence, items[0])
    item2_vector = get_averaged_vector(tokenizer, model, sentence, items[1])
    if item1_vector is not None and item2_vector is not None:
        return np.concatenate((item1_vector, item2_vector), axis=1)


def get_averaged_vector(tokenizer, model, sentence, word):
    inputs = tokenizer(sentence, truncation=True, return_tensors="pt")
    word_char = sentence.index(word)
    token_id = inputs.char_to_token(word_char)
    output = model(**inputs, output_hidden_states=True).hidden_states
    vectors = [
        output[layer_no][:, token_id, :].detach().numpy() for layer_no in range(8, 12)
    ]
    return np.average(vectors, axis=0)


def get_verbobj(sentence):
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ == "dobj":
                    return (token.text, child.text)


example_sentences = [
    "I ate the soup.",
    "I cooked pasta for dinner.",
    "I organized the dinner.",
    "I attended the conference.",
]

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModel.from_pretrained("xlm-roberta-base")
nlp = spacy.load("en_core_web_trf")

for sentence in example_sentences:
    items = get_verbobj(sentence)
    relation_vector = get_relation_vector(tokenizer, model, sentence, items)
