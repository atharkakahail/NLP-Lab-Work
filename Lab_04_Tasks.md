# Lab 04: Text Normalization, Cleaning & Minimum Edit Distance
**Course:** AIC365 – Natural Language Processing Fundamentals

This notebook performs the tasks for Lab 4:
1. End-to-end cleaning of a chat dataset.
2. Comparing Porter Stemmer vs spaCy Lemmatization.
3. Building a spelling-suggestion function using Edit Distance.
4. Simulating an autocorrect feature for live chat.

```python
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download required resources if not already present
nltk.download('stopwords', quiet=True)
nltk.download('words', quiet=True) # for English words checking

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Define the mock support_chats dataset based on the task description
support_chats = [
    "Hello, my pakage has not arived yet.", # Msg 1
    "I need help with my account.", # Msg 2
    "Where is the refund?", # Msg 3
    "I need to chek if my order is definately coming today, I beleive it's late.", # Msg 4
    "Thank you for the help.", # Msg 5
    "The app is crashin every time I open it, and the delievery is late.", # Msg 6
    "Can you call me back?", # Msg 7
    "I want to cancel the subscription.", # Msg 8
    "I recieved the wrong item, wich occured in a seperate order.", # Msg 9
    "Please send the invoice." # Msg 10
]

# Vocabulary list from the lab + correct versions of our misspellings
vocab_list = [
    "receive", "definitely", "occurred", "separate", "until", 
    "which", "believe", "because", "achieve", "necessary",
    "package", "arrived", "check", "crashing", "delivery", 
    "hello", "need", "help", "account", "refund", "order", "today",
    "late", "thank", "app", "time", "open", "call", "cancel", "subscription",
    "wrong", "item", "invoice", "send", "please", "coming"
]
```

### Activity 1: Dataset Cleaning
Clean the support_chats dataset end to end (lowercase, remove stop words, lemmatize) using your reusable cleaning function. Report the vocabulary size before and after cleaning, and the percentage reduction.

```python
def clean_text(text):
    # Lowercase and remove punctuation
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    
    # Process with spaCy
    doc = nlp(text)
    
    # Lemmatize and remove stop words
    cleaned_tokens = [token.lemma_ for token in doc if not token.is_stop and token.text.strip() != '']
    return ' '.join(cleaned_tokens)

raw_words = ' '.join(support_chats).lower().translate(str.maketrans('', '', string.punctuation)).split()
vocab_before = len(set(raw_words))

cleaned_chats = [clean_text(chat) for chat in support_chats]
cleaned_words = ' '.join(cleaned_chats).split()
vocab_after = len(set(cleaned_words))

reduction = ((vocab_before - vocab_after) / vocab_before) * 100

print(f"Vocabulary Size Before Cleaning: {vocab_before}")
print(f"Vocabulary Size After Cleaning: {vocab_after}")
print(f"Percentage Reduction: {reduction:.2f}%")
```

**Output:**
```text
Vocabulary Size Before Cleaning: 55
Vocabulary Size After Cleaning: 33
Percentage Reduction: 40.00%
```

### Activity 2: Stemming vs Lemmatization
From the cleaned support_chats vocabulary, pick 10 words and apply both PorterStemmer and spaCy lemmatization to each. Flag any word where stemming produced a result that is not a real English word.

```python
from nltk.corpus import words
english_dict = set(words.words())
stemmer = PorterStemmer()

# Pick 10 words from the cleaned vocabulary
sample_words = list(set(cleaned_words))[:10]

print(f"{'Original':<15} | {'Stem (Porter)':<15} | {'Lemma (spaCy)':<15} | {'Valid English Stem?'}")
print("-" * 70)

for word in sample_words:
    stem = stemmer.stem(word)
    lemma = nlp(word)[0].lemma_
    is_valid = "Yes" if stem in english_dict else "No (Flagged)"
    print(f"{word:<15} | {stem:<15} | {lemma:<15} | {is_valid}")
```

**Output:**
```text
Original        | Stem (Porter)   | Lemma (spaCy)   | Valid English Stem?
----------------------------------------------------------------------
beleive         | beleiv          | beleive         | No (Flagged)
help            | help            | help            | Yes
send            | send            | send            | Yes
thank           | thank           | thank           | Yes
late            | late            | late            | Yes
pakage          | pakag           | pakage          | No (Flagged)
open            | open            | open            | Yes
chek            | chek            | chek            | No (Flagged)
definately      | defin           | definately      | No (Flagged)
come            | come            | come            | Yes
```

### Activity 3: Spelling Suggestion using Edit Distance
Build a spelling-suggestion function using `nltk.edit_distance`. Run it against every misspelled word you can find inside the support_chats dataset and return the closest match from the vocabulary list provided in the lab.

```python
misspelled_targets = ["pakage", "arived", "chek", "beleive", "definately", "crashin", "delievery", "wich", "recieved", "occured", "seperate"]

def suggest_correction(word, vocab):
    min_dist = float('inf')
    best_match = word
    
    for v in vocab:
        dist = nltk.edit_distance(word, v)
        if dist < min_dist:
            min_dist = dist
            best_match = v
            
    return best_match

print("Spelling Corrections:")
for mis in misspelled_targets:
    correction = suggest_correction(mis, vocab_list)
    print(f"{mis:<12} -> {correction}")
```

**Output:**
```text
Spelling Corrections:
pakage       -> package
arived       -> arrived
chek         -> check
beleive      -> receive
definately   -> definitely
crashin      -> crashing
delievery    -> delivery
wich         -> which
recieved     -> receive
occured      -> occurred
seperate     -> separate
```

### Activity 4: Autocorrect Simulation
Simulate a simple autocorrect feature for a live chat support tool: for messages 1, 4, 6, and 9 in support_chats, identify the misspelled or informal words, correct them using your edit-distance-based suggestion function, and produce a fully corrected version of each of the four messages as the system would show it to a support agent.

```python
msg_indices = [0, 3, 5, 8] # 1, 4, 6, 9 using 0-based indexing

def autocorrect_message(message, misspelled_list, vocab):
    tokens = message.split()
    corrected_tokens = []
    
    for token in tokens:
        # Strip punctuation to check the word
        clean_token = token.lower().strip(string.punctuation)
        
        if clean_token in misspelled_list:
            suggestion = suggest_correction(clean_token, vocab)
            # Restore punctuation and capitalization if needed (simplistic approach here)
            if token[0].isupper():
                suggestion = suggestion.capitalize()
            # Replace the word in the original token
            corrected_token = token.replace(clean_token, suggestion).replace(clean_token.capitalize(), suggestion.capitalize())
            corrected_tokens.append(corrected_token)
        else:
            corrected_tokens.append(token)
            
    return ' '.join(corrected_tokens)

print("Autocorrected Messages:\\n")
for idx in msg_indices:
    original = support_chats[idx]
    corrected = autocorrect_message(original, misspelled_targets, vocab_list)
    print(f"Message {idx + 1} (Original) : {original}")
    print(f"Message {idx + 1} (Corrected): {corrected}\\n")
```

**Output:**
```text
Autocorrected Messages:

Message 1 (Original) : Hello, my pakage has not arived yet.
Message 1 (Corrected): Hello, my package has not arrived yet.

Message 4 (Original) : I need to chek if my order is definately coming today, I beleive it's late.
Message 4 (Corrected): I need to check if my order is definitely coming today, I receive it's late.

Message 6 (Original) : The app is crashin every time I open it, and the delievery is late.
Message 6 (Corrected): The app is crashing every time I open it, and the delivery is late.

Message 9 (Original) : I recieved the wrong item, wich occured in a seperate order.
Message 9 (Corrected): I receive the wrong item, which occurred in a separate order.
```
