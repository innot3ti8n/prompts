import spacy
from spacy.language import Language
from spacy.tokens import Doc

# Load the language model
nlp = spacy.load("en_core_web_sm")

# Custom component for detecting quotes for dialogue
@Language.component("detect_quotes_for_dialogue")
def detect_quotes_for_dialogue(doc):
    result = doc._.result
    quote_open = False
    for token in doc:
        if token.text in ['"', "'"]:
            quote_open = not quote_open
        if quote_open:
            # Inside dialogue, checking for quotes
            result.append({
                "comp_index": 1,  # Index for quotes for dialogue
                "start": token.idx,
                "end": token.idx + len(token.text),
                "flag": 10 if quote_open else 11
            })
    return doc

# Custom component for detecting commas separating clauses
@Language.component("detect_commas_separating_clauses")
def detect_commas_separating_clauses(doc):
    result = doc._.result
    for token in doc:
        if token.dep_ in ['advcl', 'relcl']:
            comma_found = False
            for child in token.children:
                if child.text == ',':
                    comma_found = True
                    result.append({
                        "comp_index": 2,  # Index for commas separating clauses
                        "start": child.idx,
                        "end": child.idx + len(child.text),
                        "flag": 10
                    })
            if not comma_found:
                result.append({
                    "comp_index": 2,
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "flag": 11  # Flag 11 if comma is missing
                })
    return doc

# Custom component for detecting subordinating clauses
@Language.component("detect_subordinating_clauses")
def detect_subordinating_clauses(doc):
    result = doc._.result
    for token in doc:
        if token.dep_ == 'mark' and token.head.dep_ == 'advcl':
            result.append({
                "comp_index": 3,  # Index for subordinating clauses
                "start": token.idx,
                "end": token.idx + len(token.text),
                "flag": 10
            })
    return doc

# Custom component for detecting complex dialogue
@Language.component("detect_complex_dialogue")
def detect_complex_dialogue(doc):
    result = doc._.result
    dialogue = False
    for token in doc:
        if token.text in ['"', "'"]:
            dialogue = not dialogue
        if dialogue and token.pos_ in ['VERB', 'PRON']:
            result.append({
                "comp_index": 4,  # Index for complex dialogue
                "start": token.idx,
                "end": token.idx + len(token.text),
                "flag": 10
            })
    return doc

# Custom component for detecting simple punctuation (basic sentence end)
@Language.component("detect_simple_punctuation")
def detect_simple_punctuation(doc):
    result = doc._.result
    for sent in doc.sents:
        if sent[-1].text not in '.!?':
            result.append({
                "comp_index": 5,  # Index for simple punctuation
                "start": sent.end_char - 1,
                "end": sent.end_char,
                "flag": 11  # Flag 11 if simple punctuation is missing
            })
        else:
            result.append({
                "comp_index": 5,
                "start": sent.end_char - 1,
                "end": sent.end_char,
                "flag": 10  # Flag 10 if simple punctuation is correct
            })
    return doc

# Custom component for detecting complex punctuation (handling colons, semicolons, etc.)
@Language.component("detect_complex_punctuation")
def detect_complex_punctuation(doc):
    result = doc._.result
    for token in doc:
        if token.text in [':', ';', '--']:
            result.append({
                "comp_index": 6,  # Index for complex punctuation
                "start": token.idx,
                "end": token.idx + len(token.text),
                "flag": 10  # Flag 10 for correct complex punctuation usage
            })
    return doc

# Register custom attributes
Doc.set_extension("result", default=[])

# Add all components to the pipeline
nlp.add_pipe("detect_quotes_for_dialogue", last=True)
nlp.add_pipe("detect_commas_separating_clauses", last=True)
nlp.add_pipe("detect_subordinating_clauses", last=True)
nlp.add_pipe("detect_complex_dialogue", last=True)
nlp.add_pipe("detect_simple_punctuation", last=True)
nlp.add_pipe("detect_complex_punctuation", last=True)

# Function to process text and return the results
def process_text(text):
    # Process text
    doc = nlp(text)

    # Access the result
    result = doc._.result

    return result

# Example text to analyze
text = '''"I can't believe it's already autumn," said Emma. "It feels like we just celebrated summer!" She paused, reflecting on the seasons. After a brief moment, she added, "Remember last year when we visited the park? That was incredible."
As they walked, John replied, "Yes, I do—especially the picnic; we had sandwiches, chips, and fruit." He smiled, continuing, "Autumn has always been my favorite season—cool air, colorful leaves, and crisp mornings."
Emma nodded in agreement, "It’s hard to beat the beauty of nature during this time of year."
They both cherished these moments.'''

# Process the text
result = process_text(text)

# Print the result
for r in result:
    print(r)
