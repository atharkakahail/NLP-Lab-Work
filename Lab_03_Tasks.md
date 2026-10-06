# Lab 03 — Regex, Tokenization & Sentence Segmentation

**Course:** AIC365 – Natural Language Processing Fundamentals  
**Objective:** Pull structured information out of raw, unstructured text using regular expressions, and correctly break text into words and sentences using NLTK and spaCy.

---

## Stage v (verify)

### Sample Text for Activities 1 & 2
First, let's define a sample text block that contains emails, order numbers, phone numbers, and currency/numeric quantities.

```python
sample_text = """
Hello Support,

I am writing to inquire about my recent order. My order number is ORD-59302-XY, placed on 12/05/2023.
The total amount was $145.50 for the 3 items I purchased. 
I also have another pending order (ORD-11294-AB) that costs €89.99.

Please contact me at john.doe123@email.co.uk or via my phone number: (555) 123-4567.
Alternatively, you can reach my wife at jane_doe+shopping@gmail.com or 555-987-6543.

Thank you, 
John Doe
"""
print("Sample text loaded successfully.")
```

**Output:**
```text
Sample text loaded successfully.
```

### Activity 1
**Task:** Extract all emails, order numbers, and phone numbers from the given raw text block using regex.

```python
import re

# Regex patterns
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
order_pattern = r'ORD-\d{5}-[A-Z]{2}'
# Matches formats like (555) 123-4567 or 555-987-6543
phone_pattern = r'\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}'

# Extraction
emails = re.findall(email_pattern, sample_text)
order_numbers = re.findall(order_pattern, sample_text)
phone_numbers = re.findall(phone_pattern, sample_text)

print("--- Activity 1: Regex Extraction ---")
print(f"Emails found: {emails}")
print(f"Order numbers found: {order_numbers}")
print(f"Phone numbers found: {phone_numbers}")
```

**Output:**
```text
--- Activity 1: Regex Extraction ---
Emails found: ['john.doe123@email.co.uk', 'jane_doe+shopping@gmail.com']
Order numbers found: ['ORD-59302-XY', 'ORD-11294-AB']
Phone numbers found: ['(555) 123-4567', '555-987-6543']
```

### Activity 2
**Task:** Extract all currency amounts and numeric quantities from the same text block using token attributes (not regex), and compare the results with the regex-based approach.

```python
import spacy

# Load the small English spaCy model
nlp = spacy.load("en_core_web_sm")
doc = nlp(sample_text)

currency_amounts = []
numeric_quantities = []

# Iterate through tokens to check attributes
for i, token in enumerate(doc):
    # Check for currency symbol followed by a number (e.g., $145.50)
    if token.is_currency and i + 1 < len(doc) and doc[i+1].like_num:
        currency_amounts.append(f"{token.text}{doc[i+1].text}")
    
    # Extract purely numeric quantities (ignoring parts of order numbers or phone numbers when possible)
    if token.like_num:
        numeric_quantities.append(token.text)

print("--- Activity 2: spaCy Token Attributes ---")
print(f"Currency amounts found: {currency_amounts}")
print(f"All numeric quantities found: {numeric_quantities}")

print("\n--- Comparison (Regex vs Token Attributes) ---")
print("Regex allows for precise, pattern-based extraction (like specific order number formats or complex email structures).")
print("Token attributes (spaCy) provide a more semantic understanding (like knowing '$' is a currency symbol and '3' is a number), but they can be noisy if the tokenization splits things unexpectedly (e.g., phone numbers split into multiple numeric tokens).")
```

**Output:**
```text
--- Activity 2: spaCy Token Attributes ---
Currency amounts found: ['$145.50', '€89.99']
All numeric quantities found: ['59302', '12/05/2023', '145.50', '3', '11294', '89.99', '123', '555', '123', '4567', '555', '987', '6543']

--- Comparison (Regex vs Token Attributes) ---
Regex allows for precise, pattern-based extraction (like specific order number formats or complex email structures).
Token attributes (spaCy) provide a more semantic understanding (like knowing '$' is a currency symbol and '3' is a number), but they can be noisy if the tokenization splits things unexpectedly (e.g., phone numbers split into multiple numeric tokens).
```

### Sample Document for Activities 3 & 4

```python
sample_doc = """
Dr. Smith isn't entirely sure about the diagnosis. He said, "It's a complicated case, isn't it?"
The patient arrived at 5:30 p.m. on Thursday. Let's hope they'll recover quickly! 
I've contacted the specialist in Washington D.C. for a consultation.
"""
print("Sample document loaded successfully.")
```

**Output:**
```text
Sample document loaded successfully.
```

### Activity 3
**Task:** Tokenize the given document using both NLTK and spaCy, and note where the outputs differ (punctuation, contractions).

```python
import nltk
from nltk.tokenize import word_tokenize

# NLTK Tokenization
nltk_tokens = word_tokenize(sample_doc)

# spaCy Tokenization
doc_spacy = nlp(sample_doc)
spacy_tokens = [token.text for token in doc_spacy if not token.is_space] # Exclude pure whitespace for easier comparison

print("--- NLTK Tokens (first 20) ---")
print(nltk_tokens[:20])

print("\n--- spaCy Tokens (first 20) ---")
print(spacy_tokens[:20])

print("\n--- Differences Noted ---")
print("1. Contractions: NLTK splits \"isn't\" into ['is', \"n't\"]. spaCy also splits it into ['is', \"n't\"].")
print("2. Quotes: NLTK treats quotes like `` and '' sometimes, depending on the exact tokenization, whereas spaCy keeps them as standard characters (\").")
print("3. Abbreviations: NLTK might split \"Dr.\" into ['Dr', '.'], while spaCy often keeps \"Dr.\" as a single token because it recognizes it as a title/abbreviation.")
print("4. NLTK word_tokenize strips out spaces completely, while spaCy preserves them as tokens (which we manually filtered out above for comparison).")
```

**Output:**
```text
--- NLTK Tokens (first 20) ---
['Dr.', 'Smith', 'is', "n't", 'entirely', 'sure', 'about', 'the', 'diagnosis', '.', 'He', 'said', ',', '``', 'It', "'s", 'a', 'complicated', 'case', ',']

--- spaCy Tokens (first 20) ---
['Dr.', 'Smith', 'is', "n't", 'entirely', 'sure', 'about', 'the', 'diagnosis', '.', 'He', 'said', ',', '"', 'It', "'s", 'a', 'complicated', 'case', ',']

--- Differences Noted ---
1. Contractions: NLTK splits "isn't" into ['is', "n't"]. spaCy also splits it into ['is', "n't"].
2. Quotes: NLTK treats quotes like `` and '' sometimes, depending on the exact tokenization, whereas spaCy keeps them as standard characters (").
3. Abbreviations: NLTK might split "Dr." into ['Dr', '.'], while spaCy often keeps "Dr." as a single token because it recognizes it as a title/abbreviation.
4. NLTK word_tokenize strips out spaces completely, while spaCy preserves them as tokens (which we manually filtered out above for comparison).
```

### Activity 4
**Task:** Write a program to segment a 5–6 sentence paragraph using both the NLTK and spaCy sentencizers, and flag any mismatch between them.

```python
from nltk.tokenize import sent_tokenize

# NLTK Sentence Segmentation
nltk_sentences = sent_tokenize(sample_doc.strip())

# spaCy Sentence Segmentation
spacy_sentences = [sent.text.strip() for sent in doc_spacy.sents if sent.text.strip()]

print("--- NLTK Sentences ---")
for i, s in enumerate(nltk_sentences, 1):
    print(f"{i}. {s}")

print("\n--- spaCy Sentences ---")
for i, s in enumerate(spacy_sentences, 1):
    print(f"{i}. {s}")

print("\n--- Mismatch Flagging ---")
if len(nltk_sentences) != len(spacy_sentences):
    print(f"MISMATCH: NLTK found {len(nltk_sentences)} sentences, while spaCy found {len(spacy_sentences)} sentences.")
else:
    mismatches = 0
    for i, (n_sent, s_sent) in enumerate(zip(nltk_sentences, spacy_sentences)):
        # Simple cleaning to compare content rather than exact whitespace matching
        n_clean = " ".join(n_sent.split())
        s_clean = " ".join(s_sent.split())
        if n_clean != s_clean:
            print(f"Mismatch found at sentence {i+1}:")
            print(f"  NLTK : {n_clean}")
            print(f"  spaCy: {s_clean}")
            mismatches += 1
    
    if mismatches == 0:
        print("No mismatches found! Both libraries segmented the sentences identically.")
```

**Output:**
```text
--- NLTK Sentences ---
1. Dr. Smith isn't entirely sure about the diagnosis.
2. He said, "It's a complicated case, isn't it?"
3. The patient arrived at 5:30 p.m. on Thursday.
4. Let's hope they'll recover quickly!
5. I've contacted the specialist in Washington D.C. for a consultation.

--- spaCy Sentences ---
1. Dr. Smith isn't entirely sure about the diagnosis.
2. He said, "It's a complicated case, isn't it?"
3. The patient arrived at 5:30 p.m. on Thursday.
4. Let's hope they'll recover quickly!
5. I've contacted the specialist in Washington D.C. for a consultation.

--- Mismatch Flagging ---
No mismatches found! Both libraries segmented the sentences identically.
```

---
*End of Lab 03*

