import nltk
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import keybert
from rouge import Rouge

# Load pre-trained models
nltk.download('punkt')
rouge = Rouge()
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Define functions for similarity metrics


def rouge_score(file1, file2):
    ref = open(file1).read()
    hyp = open(file2).read()
    scores = rouge.get_scores(hyp, ref)
    return scores[0]['rouge-l']['f'] * 100

def embedded_cosine_score(file1, file2):
    ref = open(file1).read()
    hyp = open(file2).read()
    ref_embedding = model.encode([ref])
    hyp_embedding = model.encode([hyp])
    score = cosine_similarity(ref_embedding, hyp_embedding)
    return score[0][0]* 100

def frequency_cosine_score(file1, file2):
    ref = open(file1).read()
    hyp = open(file2).read()
    ref_freq = nltk.FreqDist(nltk.word_tokenize(ref))
    hyp_freq = nltk.FreqDist(nltk.word_tokenize(hyp))
    ref_vec = np.array([ref_freq[word] for word in ref_freq])
    hyp_vec = np.array([hyp_freq[word] for word in ref_freq])
    score = cosine_similarity([ref_vec], [hyp_vec])
    return score[0][0]* 100



# Example usage
file1 = 'assets\\human.txt'
file2 = 'assets\\application.txt'


rouge = rouge_score(file1, file2)
embedded_cosine = embedded_cosine_score(file1, file2)
frequency_cosine = frequency_cosine_score(file1, file2)


print(f"ROUGE score: {rouge:.2f}%")
print(f"Embedded cosine score: {embedded_cosine:.2f}%")
print(f"Frequency cosine score: {frequency_cosine:.2f}%")

import nltk
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import keybert
from rouge import Rouge
from sklearn.metrics import precision_recall_fscore_support

# Load pre-trained models
nltk.download('punkt')
rouge = Rouge()
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Define functions for similarity metrics
def bleu_score(file1, file2):
    ref = open(file1).read()
    hyp = open(file2).read()
    ref = nltk.sent_tokenize(ref)
    hyp = nltk.sent_tokenize(hyp)
    score = sentence_bleu(ref, hyp)
    return score

def rouge_score(file1, file2):
    ref = open(file1).read()
    hyp = open(file2).read()
    scores = rouge.get_scores(hyp, ref)
    return scores[0]['rouge-l']['f']

def embedded_cosine_score(file1, file2):
    ref = open(file1).read()
    hyp = open(file2).read()
    ref_embedding = model.encode([ref])
    hyp_embedding = model.encode([hyp])
    score = cosine_similarity(ref_embedding, hyp_embedding)
    return score[0][0]

def frequency_cosine_score(file1, file2):
    ref = open(file1).read()
    hyp = open(file2).read()
    ref_freq = nltk.FreqDist(nltk.word_tokenize(ref))
    hyp_freq = nltk.FreqDist(nltk.word_tokenize(hyp))
    ref_vec = np.array([ref_freq[word] for word in ref_freq])
    hyp_vec = np.array([hyp_freq[word] for word in ref_freq])
    score = cosine_similarity([ref_vec], [hyp_vec])
    return score[0][0]

def keybert_score(file1, file2):
    ref = open(file1).read()
    hyp = open(file2).read()
    extractor = keybert.KeyBERT('distilbert-base-nli-mean-tokens')
    ref_keywords = extractor.extract_keywords(ref, keyphrase_ngram_range=(1,2), stop_words='english')
    hyp_keywords = extractor.extract_keywords(hyp, keyphrase_ngram_range=(1,2), stop_words='english')
    score = len(set(ref_keywords).intersection(hyp_keywords)) / len(set(ref_keywords))
    return score

def precision_recall_f1(file1, file2):
    ref = open(file1).read()
    hyp = open(file2).read()
    ref_sents = nltk.sent_tokenize(ref)
    hyp_sents = nltk.sent_tokenize(hyp)
    ref_summary = ' '.join(ref_sents)
    hyp_summary = ' '.join(hyp_sents)
    precision, recall, f1, _ = precision_recall_fscore_support([ref_summary], [hyp_summary], average='binary')
    return precision, recall, f1

def precision_score(file1, file2):
    ref = open(file1).read()
    hyp = open(file2).read()
    ref_tokens = set(nltk.word_tokenize(ref))
    hyp_tokens = set(nltk.word_tokenize(hyp))
    true_positives = len(ref_tokens.intersection(hyp_tokens))
    false_positives = len(hyp_tokens - ref_tokens)
    precision = true_positives / (true_positives + false_positives)
    return precision

def recall_score(file1, file2):
    ref = open(file1).read()
    hyp = open(file2).read()
    ref_tokens = set(nltk.word_tokenize(ref))
    hyp_tokens = set(nltk.word_tokenize(hyp))
    true_positives = len(ref_tokens.intersection(hyp_tokens))
    false_negatives = len(ref_tokens - hyp_tokens)
    recall = true_positives / (true_positives + false_negatives)
    return recall

def f1_score(file1, file2):
    precision = precision_score(file1, file2)
    recall = recall_score(file1, file2)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Example usage
file1 = 'assets\\human.txt'
file2 = 'assets\\application.txt'

bleu = bleu_score(file1, file2)
rouge = rouge_score(file1, file2)
embedded_cosine = embedded_cosine_score(file1, file2)
frequency_cosine = frequency_cosine_score(file1, file2)
# keybert = keybert_score(file1, file2)
# precision, recall, f1 = precision_recall_f1(file1, file2)
precision_Score = precision_score(file1, file2)
recall_Score = recall_score(file1, file2)
f1_Score = f1_score(file1, file2)


print(f"ROUGE score: {rouge*100:.2f}%")
print(f"Embedded cosine score: {embedded_cosine*100:.2f}%")
print(f"Frequency cosine score: {frequency_cosine*100:.2f}%")
print(f"Precision_Score: {precision_Score*100:.2f}%")
print(f"Recall_Score: {recall_Score*100:.2f}%")
print(f"F1 Score: {f1_Score*100:.2f}%")

