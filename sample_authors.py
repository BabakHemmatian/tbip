# we identified: Phrag, MrFlesh, thetimeisnow, MyaloMark, mexicodoug
author_list = ["MrFlesh","oddmanout","Phrag","NoMoreNicksLeft","permaculture",
"aletoledo","thetimeisnow","MyaloMark","mexicodoug","rainman_104","mutatron",
"otakucode","cuteman","donh","nixonrichard","garyp714","Stormflux","seeker135",
"dirtymoney","folderol"]

sample_num = 5

author_counts = {i:[] for i in author_list}
with open("prolific_authors","r") as authors:
    for id_,author in enumerate(authors):
        if author.strip() in author_list:
            if len(author_counts[author.strip()]) >= sample_num:
                pass
            else:
                author_counts[author.strip()].append(id_)

        finished = 0
        for authorship in author_counts:
            if author_counts[authorship] == sample_num:
                finished += 1

        if finished == len(author_list):
            break

author_texts = {i:[] for i in author_list}
with open("prolific_texts","r") as texts:
    reader = texts.readlines()
with open("prolific_texts","w") as texts:
    for line in reader:
        if line.strip() != "":
            print(line.strip(),end="\n",file=texts)

with open("prolific_texts","r") as texts:
    for id_,text in enumerate(texts):
        for author in author_list:
            if id_ in author_counts[author]:
                author_texts[author].append(text.strip())

with open("prolific_sampled_comments", "w") as sample:
    for author in author_texts:
        print("\n"+author+":\n",file=sample)
        for text in author_texts[author]:
            print(text,end="\n",file=sample)
