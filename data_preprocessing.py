from collections import Counter,OrderedDict
import numpy as np
import nltk
import re
import spacy
import scipy
import os

# define hyperparameters
n_authors = 10
min_docs = 50
max_docs = 100

if not os.isdir("data/ML_Reddit-{}-{}-{}".format(n_authors,min_docs,max_docs)):
    os.mkdirs("data/ML_Reddit-{}-{}-{}".format(n_authors,min_docs,max_docs))

keepers = ["how","should","should've","could","can","need","needn","why","few",
"more","most","all","any","against","because","ought","must","mustn","mustn't",
"shouldn","shouldn't","couldn't","couldn","shan't", "needn't"]
stop = []
for word in set(nltk.corpus.stopwords.words('english')):
    if word not in keepers:
        stop.append(str(word))

all_authors = []
with open("data/ML_Reddit/prolific_authors","r") as authors:
    for line in authors:
        all_authors.append(line.strip())

author_counts = Counter(all_authors)
sampled_authors = []
counter = 0
for key in author_counts.keys():
    if author_counts[key] >= min_docs and author_counts[key] <= max_docs:
        sampled_authors.append(key)
    if len(sampled_authors) == n_authors:
        break
if len(sampled_authors) < n_authors:
    raise Exception("only {} authors were found in the specified number of documents range. Consider adjusting the extrema and trying again.".format(len(sampled_authors)))

author_indices = {}
for idx,author in enumerate(sampled_authors):
    author_indices[author] = idx


sampled_texts = []
comment_author = []
with open("data/ML_Reddit-{}-{}-{}/prolific_texts".format(n_authors,min_docs,max_docs),"r") as texts:
    for idx,line in enumerate(texts):
        if all_authors[idx] in sampled_authors:
            sampled_texts.append(line)
            comment_author.append(all_authors[idx])

with open("data/ML_Reddit-{}-{}-{}/author_map".format(n_authors,min_docs,max_docs),"w") as author_file:
    for author in sampled_authors:
        print(author.strip(),end="\n",file=author_file)

auth_ind_array = []
for author in comment_author:
    auth_ind_array.append(author_indices[author])
np.array(auth_ind_array,dtype=np.float32)
np.save("data/ML_Reddit-{}-{}-{}/author_indices.npy".format(n_authors,min_docs,max_docs),auth_ind_array)

def _clean(text):

    # check input arguments for valid type
    assert type(text) is str

    replace = {"should've": "shouldve", "mustn't": "mustnt",
               "shouldn't": "shouldnt", "couldn't": "couldnt", "shan't": "shant",
               "needn't": "neednt", "-": ""}
    substrs = sorted(replace, key=len, reverse=True)
    regexp = re.compile('|'.join(map(re.escape, substrs)))
    stop_free = regexp.sub(
        lambda match: replace[match.group(0)], text)

    # remove special characters
    special_free = ""
    for word in stop_free.split():
        if "http" not in word and "www" not in word:  # remove links
            word = re.sub('[^A-Za-z0-9]+', ' ', word)
            if word.strip() != "":
                special_free = special_free + " " + word.strip()

    # check for stopwords again
    special_free = " ".join([i for i in special_free.split() if i not in
                             stop])

    return special_free

def LDA_clean(text):

    special_free = _clean(text)
    # remove stopwords --> check to see if apostrophes are properly encoded
    stop_free = " ".join([i for i in special_free.lower().split() if i.lower() not
                          in stop])
    # load lemmatizer with automatic POS tagging
    lemmatizer = spacy.load('en', disable=['parser', 'ner'])
    # Extract the lemma for each token and join
    lemmatized = lemmatizer(stop_free)
    normalized = " ".join([token.lemma_ for token in lemmatized])
    return normalized

vocabulary = {}
sampled_texts = [LDA_clean(text) for text in sampled_texts]

for text in sampled_texts:
    for word in text.strip().split():
        if word in vocabulary.keys():
            vocabulary[word] += 1
        else:
            vocabulary[word] = 1

cleaned_vocab = vocabulary.copy()
for word in vocabulary.keys():
    if vocabulary[word] > 5:
        del cleaned_vocab[word]

vocab_idx2word = {}
vocab_word2idx = {}
counter = 0
for word in cleaned_vocab.keys():
    vocab_idx2word[counter] = word
    vocab_word2idx[word] = counter
    counter+=1

with open("data/ML_Reddit/vocabulary.txt","w") as vocab_file:
    for i in range(len(vocab_idx2word)):
        print(vocab_idx2word[i],end="\n",file=vocab_file)

counts = scipy.sparse.csr_matrix((len(sampled_texts),len(cleaned_vocab)),dtype=np.float32)

for idx,text in enumerate(sampled_texts):
    for word in text.strip().split():
        if word in vocab_word2idx.keys():
            counts[idx,vocab_word2idx[word]] += 1

scipy.sparse.save_npz("data/ML_Reddit-{}-{}-{}/counts.npz".format(n_authors,min_docs,max_docs), counts)
