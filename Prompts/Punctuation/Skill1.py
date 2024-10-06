import spacy
from spacy.language import Language
from spacy.tokens import Doc

# Load the language model
nlp = spacy.load("en_core_web_sm")

# Custom component for detecting proper nouns and ensuring they are capitalized
@Language.component("detect_proper_nouns")
def detect_proper_nouns(doc):
    result = doc._.result
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE']:
            # Check if the proper noun is capitalized
            is_capitalized = ent.text.istitle()
            result.append({
                "comp_index": 1,  # Index for proper nouns
                "start": ent.start_char,
                "end": ent.end_char,
                "flag": 10 if is_capitalized else 11  # Flag 10 if capitalized, 11 if not
            })
    return doc

# Custom component for detecting key events and ensuring they are capitalized
@Language.component("detect_key_events")
def detect_key_events(doc):
    result = doc._.result
    for ent in doc.ents:
        if ent.label_ in ['EVENT']:
            # Check if the key event is capitalized
            is_capitalized = ent.text.istitle()
            result.append({
                "comp_index": 2,  # Index for key events
                "start": ent.start_char,
                "end": ent.end_char,
                "flag": 10 if is_capitalized else 11  # Flag 10 if capitalized, 11 if not
            })
    return doc

# Custom component for detecting possessive apostrophes
@Language.component("detect_possessive_apostrophes")
def detect_possessive_apostrophes(doc):
    result = doc._.result
    for token in doc:
        if token.dep_ == 'poss' and "’s" in token.head.text:
            result.append({
                "comp_index": 3,
                "start": token.head.idx,
                "end": token.head.idx + len(token.head.text),
                "flag": 10 if token.head.text.endswith("’s") or token.head.text.endswith("s’") else 11
            })

    return doc

# Custom component for detecting sentence boundary punctuation
@Language.component("detect_sentence_boundary_punctuation")
def detect_sentence_boundary_punctuation(doc):
    result = doc._.result
    for sent in doc.sents:
        # Check if the last character is a quotation mark
        if sent.text[-1] == '”':
            # Check the character before the quotation mark
            if sent.text[-2] in '.!?':
                flag = 10  # Correct usage
                punctuation_start = sent.end_char - 2
                punctuation_end = sent.end_char - 1
            else:
                flag = 11  # Incorrect usage
                punctuation_start = sent.end_char - 1
                punctuation_end = sent.end_char
        else:
            # Check the last character directly
            if sent.text[-1] in '.!?':
                flag = 10  # Correct usage
                punctuation_start = sent.end_char - 1
                punctuation_end = sent.end_char
            else:
                flag = 11  # Incorrect usage
                punctuation_start = sent.end_char - 1
                punctuation_end = sent.end_char

        result.append({
            "comp_index": 4,
            "start": punctuation_start,
            "end": punctuation_end,
            "flag": flag
        })
    
    return doc

# Custom component for detecting commas in lists (improved)
@Language.component("detect_commas_in_lists")
def detect_commas_in_lists(doc):
    result = doc._.result
    for token in doc:
        # Detect items in a list using conjunctions (e.g., apples, oranges, and bananas)
        if token.dep_ == 'cc' and token.text.lower() in ['and', 'or']:
            # Check if the previous item in the list has a comma
            if token.i > 1 and token.nbor(-2).text == ',':
                flag = 10  # Correct usage
            else:
                flag = 11  # Incorrect usage
            
            result.append({
                "comp_index": 5,
                "start": token.nbor(-2).idx if token.i > 1 else token.idx,  # Highlight the previous item and its comma
                "end": token.nbor(-2).idx + len(token.nbor(-2).text) if token.i > 1 else token.idx + len(token.text),
                "flag": flag
            })
    return doc

# Custom component for detecting commas in dates
@Language.component("detect_commas_in_dates")
def detect_commas_in_dates(doc):
    result = doc._.result
    for ent in doc.ents:
        if ent.label_ == 'DATE':
        # Check for commas in the date entity
            if ',' in ent.text:
                flag = 10  # Correct usage
                # Find the position of the comma
                comma_index = ent.text.index(',')
                start = ent.start_char + comma_index
                end = start + 1
            else:
                flag = 11  # Incorrect usage
                # If no comma, highlight the end of the entity
                start = ent.end_char - 1
                end = ent.end_char

            result.append({
                "comp_index": 6,
                "start": start,
                "end": end,
                "flag": flag
            })
    return doc

# Custom component for detecting commas for pauses (improved)
@Language.component("detect_commas_for_pauses")
def detect_commas_for_pauses(doc):
    result = doc._.result
    for token in doc:
        # Check for introductory clauses or adverbial clauses (advcl) that require a comma
        if token.dep_ == 'advcl' and token.head.pos_ == 'VERB':
            # Ensure a comma is present after the clause
            if token.i < len(doc) - 1 and token.nbor(1).text == ',':
                flag = 10  # Correct usage
                start = token.nbor(1).idx
                end = start + 1
            else:
                flag = 11  # Incorrect usage
                start = token.idx
                end = token.idx + len(token.text)

            result.append({
                "comp_index": 7,
                "start": start,
                "end": end,
                "flag": flag
            })
    return doc

# Custom component for detecting commas in quotes (improved with boundary checks)
@Language.component("detect_commas_in_quotes")
def detect_commas_in_quotes(doc):
    result = doc._.result
    quote_open = False
    for token in doc:
        if token.text in ['"', "'"]:
            quote_open = not quote_open
        if not quote_open:
            # Ensure token index is greater than 0 before accessing previous token
            if token.i > 0 and token.nbor(-1).text == ',':
                flag = 10  # Correct usage
            else:
                flag = 11  # Incorrect usage
            
            result.append({
                "comp_index": 8,
                "start": token.nbor(-1).idx if token.i > 0 else token.idx,
                "end": token.nbor(-1).idx + len(token.nbor(-1).text) if token.i > 0 else token.idx + len(token.text),
                "flag": flag
            })
    return doc

# Register custom attributes
Doc.set_extension("result", default=[])

# Add all components to the pipeline for detection only
nlp.add_pipe("detect_proper_nouns", last=True)
nlp.add_pipe("detect_key_events", last=True)
nlp.add_pipe("detect_possessive_apostrophes", last=True)
nlp.add_pipe("detect_sentence_boundary_punctuation", last=True)
nlp.add_pipe("detect_commas_in_lists", last=True)
nlp.add_pipe("detect_commas_in_dates", last=True)
nlp.add_pipe("detect_commas_for_pauses", last=True)
nlp.add_pipe("detect_commas_in_quotes", last=True)

# Function to process text and return the results
def process_text(text):
    # Process text
    doc = nlp(text)

    # Access the result
    result = doc._.result

    return result    

# Example text to analyze
text = '''on july 4 2021 sarahs friends gathered to celebrate independence day at central park in new york. sarah along with her best friends emily jake and lisa organized the event "this is going to be amazing" said sarah excitement in her voice
as the group set up their picnic they shared stories about past independence days sarah recounted "last year we visited washington dc and saw the fireworks at the lincoln memorial it was unforgettable" emily added remember how jakes car broke down on the way?" everyone laughed
while enjoying apples oranges and sandwiches they watched the sky turn pink as the sun began to set sarah looked at the horizon and whispered "this moment this day is perfect
afterward as the fireworks lit up the sky jake remarked "i cant believe its already been a year since our last celebration lisa nodded agreeing "time flies but these memories last forever
the group packed up as the night grew darker same time next year?" asked emily everyone agreed promising to make this gathering an annual tradition
'''

# Process the text
result = process_text(text)

# Print the result
for r in result:
    print(r)
