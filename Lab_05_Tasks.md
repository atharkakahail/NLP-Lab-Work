# Lab 05: N-gram Language Models & Smoothing

**Course:** AIC365 – Natural Language Processing Fundamentals  
**Stage:** Stage v (verify)  
**Corpus:** `product_reviews_corpus`  

This document provides the complete tasks, implementation code, and execution outputs for **Lab 05 (Stage v)**:
1. **Activity 1:** Bigram Smoothing Comparison (Laplace vs Add-$k$ / Lidstone) and numerical justification on held-out text.
2. **Activity 2:** Next-word prediction for a review-writing assistant given `"the camera quality exceeded my"`, ranking `"expectations"`, `"budget"`, and `"screen"`.
3. **Activity 3:** Model order comparison (Bigram vs Trigram) with generalization analysis on unseen text.
4. **Activity 4:** Out-of-vocabulary (OOV) word identification, comparing unsmoothed MLE against smoothed model probabilities and perplexities.

---

## Setup & Corpus Loading

```python
import os
import math
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.util import ngrams
from nltk.lm import MLE, Laplace, Lidstone, KneserNeyInterpolated, WittenBellInterpolated

# Ensure tokenizer resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

print("Libraries imported and NLTK resources ready.")
```

**Output:**
```text
Libraries imported and NLTK resources ready.
```

### Dataset Preparation (`product_reviews_corpus`)

```python
# Load product_reviews_corpus from disk
corpus_file = "product_reviews_corpus.txt"

with open(corpus_file, "r", encoding="utf-8") as f:
    product_reviews_corpus = [line.strip().lower() for line in f if line.strip()]

# Tokenize training corpus
tokenized_corpus = [word_tokenize(sentence) for sentence in product_reviews_corpus]

# Vocabulary calculation
flat_tokens = [w for s in tokenized_corpus for w in s]
unique_vocab = sorted(list(set(flat_tokens)))

print(f"Total training review sentences: {len(product_reviews_corpus)}")
print(f"Total token count in corpus:     {len(flat_tokens)}")
print(f"Unique vocabulary size (V):      {len(unique_vocab)}")

# Define the held-out sentence from Stage a1 above
held_out = "the movie was fantastic and the acting was convincing"
tokenized_held_out = word_tokenize(held_out.lower())

print(f"\nHeld-out sentence: \"{held_out}\"")
print(f"Tokenized held-out: {tokenized_held_out}")
```

**Output:**
```text
Total training review sentences: 25
Total token count in corpus:     210
Unique vocabulary size (V):      107

Held-out sentence: "the movie was fantastic and the acting was convincing"
Tokenized held-out: ['the', 'movie', 'was', 'fantastic', 'and', 'the', 'acting', 'was', 'convincing']
```

---

## Activity 1: Bigram Smoothing Comparison

**Task:** On the `product_reviews_corpus` dataset, fit two smoothing methods of your choice on a bigram model and report the perplexity of each on the held-out sentence above. State which smoothing method performed better and justify your choice using the numbers.

```python
# Prepare held-out bigrams
held_out_padded = list(pad_both_ends(tokenized_held_out, n=2))
held_out_bigrams = list(ngrams(held_out_padded, 2))

# 1. Fit Laplace (Add-1) Bigram Model
train_data_laplace, vocab_laplace = padded_everygram_pipeline(2, tokenized_corpus)
laplace_bigram = Laplace(2)
laplace_bigram.fit(train_data_laplace, vocab_laplace)
laplace_perplexity = laplace_bigram.perplexity(held_out_bigrams)

# 2. Fit Add-k (Lidstone with k=0.5) Bigram Model
train_data_addk, vocab_addk = padded_everygram_pipeline(2, tokenized_corpus)
addk_bigram = Lidstone(gamma=0.5, order=2)
addk_bigram.fit(train_data_addk, vocab_addk)
addk_perplexity = addk_bigram.perplexity(held_out_bigrams)

# Tabulate Comparison
activity_1_df = pd.DataFrame({
    "Smoothing Method": ["Laplace Smoothing (Add-1)", "Add-k Smoothing (k=0.5)"],
    "Hyperparameter (k)": [1.0, 0.5],
    "Held-Out Perplexity": [laplace_perplexity, addk_perplexity]
})

print("=== Activity 1: Bigram Smoothing Comparison ===")
print(activity_1_df.to_string(index=False))

# Evaluation and Justification
best_method = "Add-k Smoothing (k=0.5)" if addk_perplexity < laplace_perplexity else "Laplace Smoothing"
diff = abs(laplace_perplexity - addk_perplexity)
percent_improvement = (diff / max(laplace_perplexity, addk_perplexity)) * 100

print(f"\nBest Performing Smoothing Method: {best_method}")
print(f"Numerical Justification:")
print(f"- Laplace (k=1.0) Perplexity: {laplace_perplexity:.4f}")
print(f"- Add-k   (k=0.5) Perplexity: {addk_perplexity:.4f}")
print(f"- Add-0.5 achieved a lower (superior) perplexity by {diff:.4f} ({percent_improvement:.2f}% improvement).")
print(f"Reason: Laplace adds a full count of 1.0 to every bigram, redistributing an excessive V * 1.0 = {len(laplace_bigram.vocab)} probability units to unseen combinations.")
print(f"Add-0.5 adds only 0.5 (reducing the denominator inflation to V * 0.5 = {len(laplace_bigram.vocab) * 0.5}), thereby preserving more probability mass for observed linguistic structures.")
```

**Output:**
```text
=== Activity 1: Bigram Smoothing Comparison ===
         Smoothing Method  Hyperparameter (k)  Held-Out Perplexity
Laplace Smoothing (Add-1)                 1.0            81.155476
  Add-k Smoothing (k=0.5)                 0.5            77.677234

Best Performing Smoothing Method: Add-k Smoothing (k=0.5)
Numerical Justification:
- Laplace (k=1.0) Perplexity: 81.1555
- Add-k   (k=0.5) Perplexity: 77.6772
- Add-0.5 achieved a lower (superior) perplexity by 3.4782 (4.29% improvement).
Reason: Laplace adds a full count of 1.0 to every bigram, redistributing an excessive V * 1.0 = 110 probability units to unseen combinations.
Add-0.5 adds only 0.5 (reducing the denominator inflation to V * 0.5 = 55.0), thereby preserving more probability mass for observed linguistic structures.
```

---

## Activity 2: Next-Word Predictor for a Review-Writing Assistant

**Task:** Build a simple next-word predictor for a review-writing assistant: given the partial sentence `"the camera quality exceeded my"`, use your best-performing smoothed model to rank three candidate next words: `"expectations"`, `"budget"`, and `"screen"`. Report the ranked probabilities and which word the assistant would suggest first.

```python
# Using best-performing smoothed model from Activity 1 (Add-k with k=0.5)
best_model = addk_bigram

context_words = ["my"]
candidate_words = ["expectations", "budget", "screen"]

# 1. Compute Conditional Probabilities: P(candidate | "my")
cond_probs = {}
for word in candidate_words:
    p = best_model.score(word, context_words)
    cond_probs[word] = p

# 2. Compute Full Completed Sentence Probabilities
def compute_sentence_probability(sentence, model):
    tokens = word_tokenize(sentence.lower())
    padded = list(pad_both_ends(tokens, n=2))
    bgs = list(ngrams(padded, 2))
    prob = 1.0
    for bg in bgs:
        prob *= model.score(bg[1], [bg[0]])
    return prob

base_partial = "the camera quality exceeded my"
sentence_probs = {}
for word in candidate_words:
    full_sentence = f"{base_partial} {word}"
    sentence_probs[word] = compute_sentence_probability(full_sentence, best_model)

# Create ranked dataframe
results_df = pd.DataFrame({
    "Candidate Word": candidate_words,
    "P(word | 'my')": [cond_probs[w] for w in candidate_words],
    "Full Sentence Probability": [sentence_probs[w] for w in candidate_words]
})

# Rank candidates by probability descending
results_df["Rank"] = results_df["P(word | 'my')"].rank(ascending=False, method="dense").astype(int)
results_df = results_df.sort_values(by="Rank").reset_index(drop=True)

print("=== Activity 2: Next-Word Prediction Ranking ===")
print(results_df.to_string(index=False))

suggested_first = results_df.iloc[0]["Candidate Word"]
highest_prob = results_df.iloc[0]["P(word | 'my')"]

print(f"\n=== Review-Writing Assistant Recommendation ===")
print(f"The assistant would suggest: \"{suggested_first}\" first (Probability: {highest_prob:.6f}).")
print(f"\nLinguistic Interpretation:")
print(f"- 'expectations' forms the collocated phrase 'exceeded my expectations', which is heavily attested in positive reviews.")
print(f"- 'budget' is observed in phrases like 'within my budget', yielding a moderate probability when smoothed.")
print(f"- 'screen' rarely follows 'my' in standard syntax, receiving the lowest probability among the candidates.")
```

**Output:**
```text
=== Activity 2: Next-Word Prediction Ranking ===
Candidate Word  P(word | 'my')  Full Sentence Probability  Rank
  expectations        0.101562               1.503226e-08     1
        budget        0.054688               8.512964e-09     2
        screen        0.007812               2.391050e-10     3

=== Review-Writing Assistant Recommendation ===
The assistant would suggest: "expectations" first (Probability: 0.101562).

Linguistic Interpretation:
- 'expectations' forms the collocated phrase 'exceeded my expectations', which is heavily attested in positive reviews.
- 'budget' is observed in phrases like 'within my budget', yielding a moderate probability when smoothed.
- 'screen' rarely follows 'my' in standard syntax, receiving the lowest probability among the candidates.
```

---

## Activity 3: Bigram vs Trigram Model Generalization

**Task:** Train both a bigram and a trigram model (using your chosen smoothing method) on the `product_reviews_corpus` dataset. Compute the perplexity of each on the held-out sentence and explain, based on the actual perplexity values, which model generalizes better to this unseen sentence.

```python
# 1. Fit Bigram Model (N=2) with Add-k (k=0.5)
train_data_2, vocab_2 = padded_everygram_pipeline(2, tokenized_corpus)
bigram_model = Lidstone(gamma=0.5, order=2)
bigram_model.fit(train_data_2, vocab_2)

held_out_padded_2 = list(pad_both_ends(tokenized_held_out, n=2))
held_out_bigrams = list(ngrams(held_out_padded_2, 2))
perplexity_bigram = bigram_model.perplexity(held_out_bigrams)

# 2. Fit Trigram Model (N=3) with Add-k (k=0.5)
train_data_3, vocab_3 = padded_everygram_pipeline(3, tokenized_corpus)
trigram_model = Lidstone(gamma=0.5, order=3)
trigram_model.fit(train_data_3, vocab_3)

held_out_padded_3 = list(pad_both_ends(tokenized_held_out, n=3))
held_out_trigrams = list(ngrams(held_out_padded_3, 3))
perplexity_trigram = trigram_model.perplexity(held_out_trigrams)

# Tabulate Comparison
model_comparison_df = pd.DataFrame({
    "Model Order": ["Bigram (N=2)", "Trigram (N=3)"],
    "Smoothing Method": ["Add-k (k=0.5)", "Add-k (k=0.5)"],
    "N-gram Tuples Evaluated": [len(held_out_bigrams), len(held_out_trigrams)],
    "Held-Out Perplexity": [perplexity_bigram, perplexity_trigram]
})

print("=== Activity 3: Model Order Generalization Comparison ===")
print(model_comparison_df.to_string(index=False))

better_model = "Bigram Model (N=2)" if perplexity_bigram < perplexity_trigram else "Trigram Model (N=3)"
print(f"\nConclusion: The {better_model} generalizes better to this unseen sentence.")
print(f"Explanation:")
print(f"- Bigram Perplexity:  {perplexity_bigram:.4f}")
print(f"- Trigram Perplexity: {perplexity_trigram:.4f}")
print(f"Perplexity measures the model's average branching factor (uncertainty). A lower perplexity indicates superior generalization.")
print(f"The trigram model suffers more acutely from data sparsity because 3-word histories like ('the', 'movie') or ('movie', 'was') never appeared in the training corpus.")
print(f"Consequently, the trigram model must fall back onto smoothed fractions for nearly all test trigrams, accumulating greater uncertainty (higher perplexity) than the more flexible bigram model.")
```

**Output:**
```text
=== Activity 3: Model Order Generalization Comparison ===
  Model Order Smoothing Method  N-gram Tuples Evaluated  Held-Out Perplexity
 Bigram (N=2)    Add-k (k=0.5)                       10            77.677234
Trigram (N=3)    Add-k (k=0.5)                       11            83.400490

Conclusion: The Bigram Model (N=2) generalizes better to this unseen sentence.
Explanation:
- Bigram Perplexity:  77.6772
- Trigram Perplexity: 83.4005
Perplexity measures the model's average branching factor (uncertainty). A lower perplexity indicates superior generalization.
The trigram model suffers more acutely from data sparsity because 3-word histories like ('the', 'movie') or ('movie', 'was') never appeared in the training corpus.
Consequently, the trigram model must fall back onto smoothed fractions for nearly all test trigrams, accumulating greater uncertainty (higher perplexity) than the more flexible bigram model.
```

---

## Activity 4: Out-of-Vocabulary (OOV) Word Impact

**Task:** Identify a word in the held-out sentence that does not appear anywhere in the `product_reviews_corpus` training data. Compute the probability of the sentence containing this out-of-vocabulary word under an unsmoothed model versus your best smoothed model, and report both perplexity values.

```python
# 1. Identify Out-of-Vocabulary (OOV) words in held_out
training_vocab_set = set(unique_vocab)
oov_candidates = [word for word in tokenized_held_out if word not in training_vocab_set]

print("=== Activity 4: Vocabulary Analysis ===")
print(f"Held-out sentence tokens: {tokenized_held_out}")
print(f"Words found not in product_reviews_corpus: {oov_candidates}")

chosen_oov_word = oov_candidates[0]
print(f"\nSelected Out-Of-Vocabulary (OOV) Word: \"{chosen_oov_word}\"")

# 2. Build Unsmoothed Bigram Model (MLE)
train_data_mle, vocab_mle = padded_everygram_pipeline(2, tokenized_corpus)
mle_bigram = MLE(2)
mle_bigram.fit(train_data_mle, vocab_mle)

# Compute probability under MLE
mle_prob = 1.0
zero_prob_bigrams = []
for bg in held_out_bigrams:
    score = mle_bigram.score(bg[1], [bg[0]])
    mle_prob *= score
    if score == 0.0:
        zero_prob_bigrams.append(bg)

mle_perplexity = mle_bigram.perplexity(held_out_bigrams)

# 3. Compute probability under Best Smoothed Model (Add-k, k=0.5)
smoothed_prob = 1.0
for bg in held_out_bigrams:
    score = addk_bigram.score(bg[1], [bg[0]])
    smoothed_prob *= score

smoothed_perplexity = addk_bigram.perplexity(held_out_bigrams)

# Tabulate results
oov_comparison_df = pd.DataFrame({
    "Model Type": ["Unsmoothed Model (MLE)", "Best Smoothed Model (Add-k, k=0.5)"],
    "Contains OOV Word?": [f"Yes ('{chosen_oov_word}')", f"Yes ('{chosen_oov_word}')"],
    "Sentence Probability": [mle_prob, smoothed_prob],
    "Sentence Perplexity": [mle_perplexity, smoothed_perplexity]
})

print("\n=== OOV Evaluation: Unsmoothed vs Smoothed Model ===")
print(oov_comparison_df.to_string(index=False))

print(f"\nZero-probability bigrams under MLE:")
for bg in zero_prob_bigrams:
    print(f"  - {bg} -> P({bg[1]} | {bg[0]}) = 0.0")

print(f"\nDetailed Findings:")
print(f"1. Under the Unsmoothed MLE Model:")
print(f"   - Sentence Probability = {mle_prob}")
print(f"   - Sentence Perplexity  = {mle_perplexity} (infinite)")
print(f"   - Explanation: Because '{chosen_oov_word}' was unseen in training data, bigrams containing it receive a count of 0.")
print(f"     A single zero probability nullifies the entire multiplicative chain, collapsing sentence probability to 0.0 and driving perplexity to infinity.")
print(f"2. Under the Best Smoothed Model (Add-k, k=0.5):")
print(f"   - Sentence Probability = {smoothed_prob:.10e}")
print(f"   - Sentence Perplexity  = {smoothed_perplexity:.4f}")
print(f"   - Explanation: The smoothing function reserves fractional probability mass for unseen vocabulary and unobserved bigrams.")
print(f"     This converts the zero-probability failure into a valid, finite score, allowing robust evaluation of arbitrary unseen text.")
```

**Output:**
```text
=== Activity 4: Vocabulary Analysis ===
Held-out sentence tokens: ['the', 'movie', 'was', 'fantastic', 'and', 'the', 'acting', 'was', 'convincing']
Words found not in product_reviews_corpus: ['movie', 'fantastic', 'acting', 'convincing']

Selected Out-Of-Vocabulary (OOV) Word: "movie"

=== OOV Evaluation: Unsmoothed vs Smoothed Model ===
                        Model Type Contains OOV Word?  Sentence Probability  Sentence Perplexity
            Unsmoothed Model (MLE)      Yes ('movie')          0.000000e+00                  inf
Best Smoothed Model (Add-k, k=0.5)      Yes ('movie')          1.250439e-19            77.677234

Zero-probability bigrams under MLE:
  - ('the', 'movie') -> P(movie | the) = 0.0
  - ('movie', 'was') -> P(was | movie) = 0.0
  - ('was', 'fantastic') -> P(fantastic | was) = 0.0
  - ('fantastic', 'and') -> P(and | fantastic) = 0.0
  - ('the', 'acting') -> P(acting | the) = 0.0
  - ('acting', 'was') -> P(was | acting) = 0.0
  - ('was', 'convincing') -> P(convincing | was) = 0.0
  - ('convincing', '</s>') -> P(</s> | convincing) = 0.0

Detailed Findings:
1. Under the Unsmoothed MLE Model:
   - Sentence Probability = 0.0
   - Sentence Perplexity  = inf (infinite)
   - Explanation: Because 'movie' was unseen in training data, bigrams containing it receive a count of 0.
     A single zero probability nullifies the entire multiplicative chain, collapsing sentence probability to 0.0 and driving perplexity to infinity.
2. Under the Best Smoothed Model (Add-k, k=0.5):
   - Sentence Probability = 1.2504389137e-19
   - Sentence Perplexity  = 77.6772
   - Explanation: The smoothing function reserves fractional probability mass for unseen vocabulary and unobserved bigrams.
     This converts the zero-probability failure into a valid, finite score, allowing robust evaluation of arbitrary unseen text.
```

---

## Summary of Results

| Activity | Task | Key Outcome / Result | Linguistic / Engineering Takeaway |
|:---:|:---|:---|:---|
| **Activity 1** | Bigram Smoothing Comparison | Laplace Perplexity: **81.16**<br>Add-0.5 Perplexity: **77.68** | Add-0.5 outperforms Laplace by avoiding excessive probability mass drainage from observed n-grams. |
| **Activity 2** | Next-Word Prediction | 1. `expectations` (0.1016)<br>2. `budget` (0.0547)<br>3. `screen` (0.0078) | The assistant suggests **"expectations"** first, leveraging high collocation strength in review contexts. |
| **Activity 3** | Bigram vs Trigram Generalization | Bigram Perplexity: **77.68**<br>Trigram Perplexity: **83.40** | Bigram generalizes better to unseen text due to significantly lower context sparsity than trigrams ($V^2 \ll V^3$). |
| **Activity 4** | OOV Word Handling | Unsmoothed Perplexity: **inf** (Prob: **0.0**)<br>Smoothed Perplexity: **77.68** (Prob: **1.25e-19**) | Smoothing eliminates the zero-probability catastrophe, converting catastrophic MLE failure into calibrated evaluation. |
