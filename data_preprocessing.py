from collections import Counter,OrderedDict
import numpy as np
import nltk
import re
import spacy
import scipy
import os
import sys

# define hyperparameters
n_authors = "unlimited"
min_docs = 50
min_length = 200
# get 3 topics instead

if not os.path.isdir("data/ML_Reddit-{}-{}-{}/clean".format(n_authors,min_docs,min_length)):
    os.makedirs("data/ML_Reddit-{}-{}-{}/clean".format(n_authors,min_docs,min_length))

keepers = ["how","should","should've","could","can","need","needn","why","few",
"more","most","all","any","against","because","ought","must","mustn","mustn't",
"shouldn","shouldn't","couldn't","couldn","shan't", "needn't"]
stop = []
for word in set(nltk.corpus.stopwords.words('english')):
    if word not in keepers:
        stop.append(str(word))

all_authors = []
with open("data/ML_Reddit/prolific_authors","r",encoding='utf-8-sig') as authors:
    for line in authors:
        all_authors.append(line.strip())

author_counts = {i:0 for i in np.unique(all_authors)}

# sampled_authors = ["MrFlesh","oddmanout","Phrag","NoMoreNicksLeft","permaculture",
# "aletoledo","thetimeisnow","MyaloMark","mexicodoug","rainman_104","mutatron",
# "otakucode","cuteman","donh","garyp714","Stormflux","seeker135","dirtymoney","folderol"]



lengths = []
with open("data/ML_Reddit/prolific_texts","r",encoding='utf-8-sig') as texts:
    for idx,line in enumerate(texts):
        if len(line.strip().split()) >= min_length:
            author_counts[all_authors[idx]] += 1


sampled_authors = []
for author in author_counts:
    if author_counts[author] >= min_docs:
        sampled_authors.append(author)

sampled_texts = []
comment_author = []
with open("data/ML_Reddit/prolific_texts","r",encoding='utf-8-sig') as texts, open("data/ML_Reddit-{}-{}-{}/sampled_texts".format(n_authors,min_docs,min_length),"w") as f:
    for id_,line in enumerate(texts):
        if all_authors[id_] in sampled_authors:
            sampled_texts.append(line)
            comment_author.append(all_authors[id_])
            print(line.strip(),end="\n",file=f)

print("Number of sampled texts: {}".format(len(sampled_texts)))

author_indices = {}
for idx,author in enumerate(sampled_authors):
    author_indices[author] = idx

with open("data/ML_Reddit-{}-{}-{}/clean/author_map.txt".format(n_authors,min_docs,min_length),"w") as author_file:
    for author in sampled_authors:
        print(author.strip(),end="\n",file=author_file)

auth_ind_array = []
for author in comment_author:
    auth_ind_array.append(author_indices[author])
np.array(auth_ind_array,dtype=np.float32)
np.save("data/ML_Reddit-{}-{}-{}/clean/author_indices.npy".format(n_authors,min_docs,min_length),auth_ind_array)

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

# load lemmatizer with automatic POS tagging
lemmatizer = spacy.load('en_core_web_sm', disable=['tagger','parser', 'ner'])

def LDA_clean(text):

    special_free = _clean(text)
    # remove stopwords --> check to see if apostrophes are properly encoded
    stop_free = " ".join([i for i in special_free.lower().split() if i.lower() not
                          in stop])
    # Extract the lemma for each token and join
    lemmatized = lemmatizer(stop_free)
    normalized = " ".join([token.lemma_ for token in lemmatized])
    return normalized

vocabulary = {}
for id_,text in enumerate(sampled_texts):
    if ((id_+1) % 1000) == 0:
        print(id_+1)
    sampled_texts[id_] = LDA_clean(text)

print("Text preprocessing finished.")

for text in sampled_texts:
    for word in text.strip().split():
        if word in vocabulary.keys():
            vocabulary[word] += 1
        else:
            vocabulary[word] = 1

cleaned_vocab = vocabulary.copy()
for word in vocabulary.keys():
    if vocabulary[word] == 1:
        del cleaned_vocab[word]

print("Vocabulary size: {}".format(len(cleaned_vocab)))

vocab_idx2word = {}
vocab_word2idx = {}
counter = 0
for word in cleaned_vocab.keys():
    vocab_idx2word[counter] = word
    vocab_word2idx[word] = counter
    counter+=1

with open("data/ML_Reddit-{}-{}-{}/clean/vocabulary.txt".format(n_authors,min_docs,min_length),"w") as vocab_file:
    for i in range(len(vocab_idx2word)):
        print(vocab_idx2word[i],end="\n",file=vocab_file)

counts = np.zeros((len(sampled_texts),len(cleaned_vocab)),dtype=np.float32)

for idx,text in enumerate(sampled_texts):
    if ((idx+1) % 1000) == 0:
        print(idx+1)
    for word in text.strip().split():
        if word in vocab_word2idx.keys():
            counts[idx,vocab_word2idx[word]] += 1

counts = scipy.sparse.csr_matrix(counts,dtype=np.float32)

print("Creating the frequency matrix finished.")

scipy.sparse.save_npz("data/ML_Reddit-{}-{}-{}/clean/counts.npz".format(n_authors,min_docs,min_length), counts)
