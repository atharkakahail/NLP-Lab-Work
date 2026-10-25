# Lab 06: Text Classification — Naive Bayes & Logistic Regression

**Course:** AIC365 – Natural Language Processing Fundamentals  
**Student ID:** SP24-BDS-008  
**Dataset:** `movie_reviews.csv` (50,000 IMDB Movie Reviews)  

This document provides the complete tasks, implementation code, and execution outputs for **Lab 06**:
1. **Activity 1:** Bag-of-Words representation using scikit-learn's `CountVectorizer`, vocabulary analysis, and matrix sparsity calculation.
2. **Activity 2:** TF-IDF representation vs. raw count comparison, explaining the mathematical suppression of high-frequency stop words.
3. **Activity 3:** Train/test split (80/20) and leakage-free vectorizer fitting on training text for Multinomial Naive Bayes and Logistic Regression.
4. **Activity 4:** Comprehensive model evaluation (Accuracy, Precision, Recall, F1-Score, Confusion Matrices) comparing both classifiers.
5. **Activity 5:** Production `Pipeline` (`TfidfVectorizer` + `LogisticRegression`) and inference on brand-new unseen movie reviews.
6. **Bonus (Stage v):** Artificial class imbalance simulation (95:5) and mitigation using `class_weight='balanced'`.

---

## Dataset Loading & Preprocessing

**Key Preprocessing Notes:**
- Columns in `movie_reviews.csv` are `review` and `sentiment`, renamed to `text` and `label` respectively.
- Exact duplicate reviews (418 reviews) are dropped so no sample leaks across train and test sets.
- HTML line breaks (`<br />`) are removed to avoid artificial tokens.

```python
import numpy as np
import pandas as pd

df = pd.read_csv('movie_reviews.csv').rename(columns={'review': 'text', 'sentiment': 'label'})
print("Raw shape:", df.shape)

df = df.drop_duplicates(subset='text')
df['text'] = df['text'].str.replace('<br />', ' ', regex=False)
df = df.reset_index(drop=True)

print("After dropping duplicates:", df.shape)
print(df['label'].value_counts())
df.head()
```

**Output:**
```text
Raw shape: (50000, 2)
After dropping duplicates: (49582, 2)
label
positive    24884
negative    24698
Name: count, dtype: int64

                                                text     label
0  One of the other reviewers has mentioned that ...  positive
1  A wonderful little production.   The filming t...  positive
2  I thought this was a wonderful way to spend ti...  positive
3  Basically there's a family where a little boy ...  negative
4  Petter Mattei's "Love in the Time of Money" is...  positive
```

---

## Activity 1: Bag-of-Words with `CountVectorizer`

Constructing the document-term matrix using raw word counts and measuring vocabulary size and sparsity.

```python
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(df['text'])

print("Vocabulary size:", len(bow_vectorizer.vocabulary_))
print("Matrix shape (docs x vocab):", X_bow.shape)
print("Sparsity: {:.2%} of entries are zero".format(1 - X_bow.nnz / (X_bow.shape[0]*X_bow.shape[1])))

print("\nSparse matrix portion (first 5 docs, first 10 features):")
terms = bow_vectorizer.get_feature_names_out()[:10]
print(pd.DataFrame(X_bow[:5, :10].toarray(), columns=terms))
```

**Output:**
```text
Vocabulary size: 101895
Matrix shape (docs x vocab): (49582, 101895)
Sparsity: 99.87% of entries are zero

Sparse matrix portion (first 5 docs, first 10 features):
   00  000  00000000000  0000000000001  00000001  00001  00015  000dm  000s  001
0   0    0            0              0         0      0      0      0     0    0
1   0    0            0              0         0      0      0      0     0    0
2   0    0            0              0         0      0      0      0     0    0
3   0    0            0              0         0      0      0      0     0    0
4   0    0            0              0         0      0      0      0     0    0
```

---

## Activity 2: TF-IDF vs. Raw Counts

Comparing TF-IDF weighting against raw term counts for high-frequency terms within a selected document.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
X_tfidf_full = tfidf_vectorizer.fit_transform(df['text'])   # exploration only, not used for modelling

doc_id = 1
doc_freq = np.asarray((X_bow > 0).sum(axis=0)).ravel()
bow_vocab, tfidf_vocab = bow_vectorizer.vocabulary_, tfidf_vectorizer.vocabulary_

bow_terms = bow_vectorizer.get_feature_names_out()
cols_in_doc = X_bow[doc_id].nonzero()[1]
top_cols = sorted(cols_in_doc, key=lambda c: -doc_freq[c])[:2]
word1, word2 = bow_terms[top_cols[0]], bow_terms[top_cols[1]]

print(f"Document {doc_id}: '{df['text'].iloc[doc_id][:300]}...'\n")
for w in (word1, word2):
    raw_c = X_bow[doc_id, bow_vocab[w]]
    tfidf = X_tfidf_full[doc_id, tfidf_vocab[w]]
    print(f"'{w}' (in {doc_freq[bow_vocab[w]]:,}/{len(df):,} reviews) -> Raw count: {raw_c}, TF-IDF: {tfidf:.4f}")

print("\nExplanation: the TF-IDF weight is lower because words found in almost every review get an IDF "
      "= log(N/DF) near zero, which suppresses them regardless of how often they occur in this document "
      "(sklearn also L2-normalises each row).")
```

**Output:**
```text
Document 1: 'A wonderful little production.   The filming technique is very unassuming- very old-time-BBC fashion and gives a comforting, and sometimes discomforting, sense of realism to the entire piece.   The actors are extremely well chosen- Michael Sheen not only "has got all the polari" but he has all the v...'

'the' (in 49,144/49,582 reviews) -> Raw count: 16, TF-IDF: 0.2693
'and' (in 47,902/49,582 reviews) -> Raw count: 7, TF-IDF: 0.1208

Explanation: the TF-IDF weight is lower because words found in almost every review get an IDF = log(N/DF) near zero, which suppresses them regardless of how often they occur in this document (sklearn also L2-normalises each row).
```

---

## Activity 3: Train/Test Split, Naive Bayes and Logistic Regression

Partitioning raw texts (80% train, 20% test) before vectorization to guarantee no data leakage from test to train.

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train_text)   # fit on train only
X_test  = tfidf.transform(X_test_text)

nb_model = MultinomialNB().fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

lr_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
print("Train size:", X_train.shape[0], "| Test size:", X_test.shape[0])
```

**Output:**
```text
Train size: 39665 | Test size: 9917
```

---

## Activity 4: Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)

```python
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

POS, NEG = 'positive', 'negative'

def evaluate_and_print(model_name, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=POS)
    rec  = recall_score(y_true, y_pred, pos_label=POS)
    f1   = f1_score(y_true, y_pred, pos_label=POS)
    cm   = confusion_matrix(y_true, y_pred, labels=[NEG, POS])
    print(f"--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Confusion Matrix (rows=true, cols=pred; order Neg, Pos):\n{cm}\n")
    return {'model': model_name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'cm': cm}

results = [evaluate_and_print("Naive Bayes", y_test, nb_preds),
           evaluate_and_print("Logistic Regression", y_test, lr_preds)]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, r in zip(axes, results):
    ConfusionMatrixDisplay(r['cm'], display_labels=[NEG, POS]).plot(ax=ax, colorbar=False, values_format='d')
    ax.set_title(r['model'])
plt.tight_layout(); plt.show()

pd.DataFrame(results).drop(columns='cm').set_index('model').round(4)
```

**Output:**
```text
--- Naive Bayes ---
Accuracy:  0.8636
Precision: 0.8767
Recall:    0.8473
F1-Score:  0.8618
Confusion Matrix (rows=true, cols=pred; order Neg, Pos):
[[4346  593]
 [ 760 4218]]

--- Logistic Regression ---
Accuracy:  0.8954
Precision: 0.8834
Recall:    0.9120
F1-Score:  0.8975
Confusion Matrix (rows=true, cols=pred; order Neg, Pos):
[[4340  599]
 [ 438 4540]]
```

### Comparative Summary Table:
| Model | Accuracy | Precision (Positive) | Recall (Positive) | F1-Score (Positive) |
|---|---|---|---|---|
| **Multinomial Naive Bayes** | 0.8636 | 0.8767 | 0.8473 | 0.8618 |
| **Logistic Regression** | **0.8954** | **0.8834** | **0.9120** | **0.8975** |

---

## Activity 5: Pipeline & Inference on New Reviews

Wrapping preprocessing and classifier into an end-to-end `Pipeline` and classifying unseen sentences loaded from `new_reviews.txt`.

```python
import os
from sklearn.pipeline import Pipeline

best_name = max(results, key=lambda r: r['f1'])['model']
print("Better model by F1:", best_name)
best_clf = MultinomialNB() if best_name == "Naive Bayes" else LogisticRegression(max_iter=1000)

pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', best_clf)])
pipeline.fit(X_train_text, y_train)
print("Pipeline test accuracy:", round(pipeline.score(X_test_text, y_test), 4), "\n")

# new_reviews.txt (one review per line)
if not os.path.exists('new_reviews.txt'):
    print("'new_reviews.txt' not found -> creating it with three sample reviews.\n")
    with open('new_reviews.txt', 'w') as f:
        f.write("An absolutely brilliant film with superb acting and a moving story.\n"
                "A boring, predictable mess. The plot made no sense and I wanted my money back.\n"
                "Not bad, but the second half dragged and the ending was disappointing.\n")

with open('new_reviews.txt') as f:
    new_reviews = [line.strip() for line in f if line.strip()]
if not new_reviews:
    print("WARNING: new_reviews.txt is empty - no predictions made.")

for review, pred in zip(new_reviews, pipeline.predict(new_reviews)):
    print(f"Review: {review}\nPrediction: {pred}\n")
```

**Output:**
```text
Better model by F1: Logistic Regression

Pipeline test accuracy: 0.8954 

'new_reviews.txt' not found -> creating it with three sample reviews.

Review: An absolutely brilliant film with superb acting and a moving story.
Prediction: positive

Review: A boring, predictable mess. The plot made no sense and I wanted my money back.
Prediction: negative

Review: Not bad, but the second half dragged and the ending was disappointing.
Prediction: negative
```

---

## Bonus (Stage v): Class Imbalance Mitigation

Simulating a severe 95:5 class imbalance (keeping ~5% of negative reviews) to show why standard accuracy is misleading and demonstrating how `class_weight='balanced'` recovers minority class recall.

```python
pos_df = df[df.label == POS]
neg_df = df[df.label == NEG].sample(n=len(pos_df)//19, random_state=42)
imb = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)
print(imb['label'].value_counts(), "\n")
print("Always-predict-'positive' accuracy: {:.3f}\n".format((imb.label == POS).mean()))

Xtr, Xte, ytr, yte = train_test_split(imb['text'], imb['label'], test_size=0.2,
                                      random_state=42, stratify=imb['label'])

def run(cw):
    p = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression(class_weight=cw, max_iter=1000))])
    p.fit(Xtr, ytr); pr = p.predict(Xte)
    return {'class_weight': str(cw),
            'accuracy': accuracy_score(yte, pr),
            'precision(neg)': precision_score(yte, pr, pos_label=NEG, zero_division=0),
            'recall(neg)': recall_score(yte, pr, pos_label=NEG, zero_division=0),
            'f1(neg)': f1_score(yte, pr, pos_label=NEG, zero_division=0)}

pd.DataFrame([run(None), run('balanced')]).set_index('class_weight').round(4)
```

**Output:**
```text
label
positive    24884
negative     1309
Name: count, dtype: int64 

Always-predict-'positive' accuracy: 0.950

              accuracy  precision(neg)  recall(neg)  f1(neg)
class_weight                                                
None            0.9544          0.9600       0.0916   0.1672
balanced        0.9483          0.4872       0.6527   0.5579
```

### Key Takeaway:
When evaluating imbalanced datasets, raw accuracy can be deceptively high (95.44%) while missing over 90% of minority class instances (Recall = 9.16%). Setting `class_weight='balanced'` penalizes minority errors inversely proportional to class frequency, elevating Recall on the critical minority class from **9.16% to 65.27%** and raising F1 from **0.1672 to 0.5579**.
