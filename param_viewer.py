import numpy as np
import sys
import matplotlib.pyplot as plt

# params = np.load("data/ML_Reddit/tbip-fits/params/document_loc.npy")
mu = np.load("data/ML_Reddit-20-100-1000-200/tbip-fits/params/document_loc.npy")
sigma = np.load("data/ML_Reddit-20-100-1000-200/tbip-fits/params/document_scale.npy")
result = np.exp((mu + sigma) / 2)

ideal_mu = np.load("data/ML_Reddit-20-100-1000-200/tbip-fits/params/ideal_point_loc.npy")
print(ideal_mu.shape)
ideal_sigma = np.load("data/ML_Reddit-20-100-1000-200/tbip-fits/params/ideal_point_scale.npy")

ideotopic_mu = np.load("data/ML_Reddit-20-100-1000-200/tbip-fits/params/ideological_topic_loc.npy")
ideotopic_sigma = np.load("data/ML_Reddit-20-100-1000-200/tbip-fits/params/ideological_topic_scale.npy")


arr = np.sum(result,axis=0) / len(mu)
top_topics = arr.argsort()[-5:][::-1]

mu2 = np.load("data/ML_Reddit-20-100-1000-200/tbip-fits/params/objective_topic_loc.npy")
sigma2 = np.load("data/ML_Reddit-20-100-1000-200/tbip-fits/params/objective_topic_scale.npy")
result2 = np.exp((mu + sigma) / 2)

id2word = {}
with open("data/ML_Reddit-20-100-1000-200/clean/vocabulary.txt","r") as f:
    for idx,line in enumerate(f):
        if line.strip() != "":
            id2word[idx] = line.strip()

topic_extreme_words = {i:[] for i in range(50)}
for idx,row in enumerate(ideotopic_mu):
    sorted_rows = row.argsort()
    rowids_high = sorted_rows[-5:][::-1]
    for id_ in rowids_high:
        topic_extreme_words[idx].append(id2word[id_])
    rowids_low = sorted_rows[:5]
    for id_ in rowids_low:
        topic_extreme_words[idx].append(id2word[id_])

for topic in topic_extreme_words:
    print(topic_extreme_words[topic])

top_ids = {}
top_words = {i:[] for i in top_topics}
for idx,line in enumerate(result2):
    if idx in top_topics:
        top_ids[idx] = line.argsort()[-20:][::-1]
        for idx2 in top_ids[idx]:
            top_words[idx].append(id2word[idx2])

## Identify unique and almost unique top words of top topics
uniques = {key:[] for key in top_words.keys()}
twos = {key:[] for key in top_words.keys()}
other_top_words = {key:[] for key in top_words.keys()}
for topic in top_words.keys():
    if topic in top_topics:
        for other_topic in top_words.keys():
            if other_topic in top_topics:
                if other_topic != topic:
                    for word in top_words[other_topic]:
                        other_top_words[topic].append(word)

for word in top_words[topic]:
    if word not in other_top_words[topic]:
        uniques[topic].append(word)

counts = {}

for topic in top_words.keys():
    for word in top_words[topic]:
        if word in counts.keys():
            counts[word] += 1
        else:
            counts[word] = 1

for topic in top_words.keys():
    for word in top_words[topic]:# top_word[idx / 2] = top_word[idx / 2][:5] # top 6
        if counts[word] == 1:
            uniques[topic].append(word)
        elif counts[word] == 2:
            twos[topic].append(word)

print(uniques)
print(twos)

counter = 0
with open("data/ML_Reddit/prolific_texts","r") as f:
    for line in f:
        if "mandate" in line:
            counter += 1

author_names = []
with open("data/ML_Reddit-20-100-1000-200/clean/author_map.txt","r") as f:
    for line in f:
        if line.strip() != "":
            author_names.append(line.strip())


print(len(author_names))

plt.bar(height=ideal_mu,x=author_names)
plt.xticks(rotation=90)
plt.show()
