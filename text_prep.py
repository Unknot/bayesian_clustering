from numpy import cumsum
from numpy import unique
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import string
import matplotlib.pyplot as plt


STOP = stopwords.words("english") + list(string.punctuation)
S_HOLMES_LEN = 6000 #220 samples
HUCK_FINN_LEN = 6000


def get_human_names(text):
    person_list = []
    person = []
    name = ""
    
    for sent in sent_tokenize(text):
        tokens = nltk.tokenize.word_tokenize(sent)
        pos = nltk.pos_tag(tokens)
        sentt = nltk.ne_chunk(pos, binary = False)
        
        for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
            for leaf in subtree.leaves():
                person.append(leaf[0])
    #        if len(person) > 1: #avoid grabbing lone surnames
            for part in person:
                name += part + ' '
            if name[:-1] not in person_list:
                person_list.append(name[:-1])
            name = ''
            person = []

    return (person_list)


def extract_names_raw(text, file):
    names = get_human_names(text)
    with open(file, 'w') as f:
        for item in names:
            f.write("%s\n" % item)
    

def main():
    with open("./huck_finn.txt", mode="r") as file:
        huck_finn = file.read()
        
    with open("./s_holmes.txt", mode="r") as file:
        s_holmes = file.read()
        
    print(len(s_holmes))
    print(len(huck_finn))
    
#    extract_names_raw(s_holmes, "./s_holmes_names.txt")
#    extract_names_raw(huck_finn, "./huck_finn_names.txt")
    
#    # After the names have been checked, load them and order by count of words
#    cleaned_s_holmes_names = []
#    with open("s_holmes_names.txt", "r") as file:
#        for line in file:
#            cleaned_s_holmes_names.append(line[:-1])
#    cleaned_s_holmes_names.sort(key = lambda x: -len(x.split(" ")))
#    for name in cleaned_s_holmes_names:
#        s_holmes = s_holmes.replace(name, "xxnamexx")
#        
#    cleaned_huck_finn_names = []
#    with open("huck_finn_names.txt", "r") as file:
#        for line in file:
#            cleaned_huck_finn_names.append(line[:-1])
#    for name in cleaned_huck_finn_names:
#        huck_finn = huck_finn.replace(name, "xxnamexx")
#    
#    print("_______")
#    print(len(s_holmes))
#    print(len(huck_finn))
    
            
    
#    s_holmes_sent = sent_tokenize(s_holmes)
#    print(s_holmes_sent[0])
#    for i in range(10):
#        print(nltk.pos_tag(word_tokenize(s_holmes_sent[i].lower())))
#        print("____________________")
    
#    s_holmes_words = [i for i in word_tokenize(s_holmes.lower()) if i not in STOP]
#    print(nltk.pos_tag(s_holmes_words[:50]))
#    s_holmes_sizes = [len(i) for i in s_holmes_words]
#    print(s_holmes_sizes[:50])
    
    s_holmes_tokens = np.array(word_tokenize(s_holmes.lower()))
    s_holmes_lens = [len(w) for w in s_holmes_tokens]
    s_holmes_indexes = cumsum(s_holmes_lens) // S_HOLMES_LEN
#    print(s_holmes_tokens[:50])
    print(s_holmes_indexes)
#    print(sum(s_holmes_indexes == 0))
#    print(sum(s_holmes_indexes == 1))
#    print(sum(s_holmes_indexes == 100))
#    print(sum(s_holmes_indexes == 220))
#    print(sum(s_holmes_indexes == 221))
    
    corpus = []
    for ind in unique(s_holmes_indexes):
        corpus.append(" ".join(s_holmes_tokens[s_holmes_indexes == ind]))
#    print(corpus)
#    print(len(corpus))
    
    huck_finn_tokens = np.array(word_tokenize(huck_finn.lower()))
    huck_finn_lens = [len(w) for w in huck_finn_tokens]
    huck_finn_indexes = cumsum(huck_finn_lens) // HUCK_FINN_LEN
    print(huck_finn_indexes)
#    print(sum(huck_finn_indexes == 0))
#    print(sum(huck_finn_indexes == 1))
#    print(sum(huck_finn_indexes == 100))
#    print(sum(huck_finn_indexes == 227))
#    print(sum(huck_finn_indexes == 228))
    
    for ind in unique(huck_finn_indexes):
        corpus.append(" ".join(huck_finn_tokens[huck_finn_indexes == ind]))
        
    print(len(corpus))
    
    
#    # tf-idf
#    vectorizer = TfidfVectorizer(min_df=2, stop_words=stopwords.words("english"))
#    X = vectorizer.fit_transform(corpus)
#    print(X.shape)
##    print(X[1,:])
    
    # ltf-real_entropy
    vectorizer = CountVectorizer(min_df=2, stop_words=stopwords.words("english"))
    X = vectorizer.fit_transform(corpus)
    print("X.shape: ", X.shape)
    X = X.A
    global_frequencies = X.sum(axis = 1)
    print(global_frequencies)
    print(X[0, :] / global_frequencies[0])

    for i in range(X.shape[0]):
        p = X[i, :] / global_frequencies[i]
        real_entropy = - np.sum(p[p != 0] * np.log2(p[p != 0]))
        X[i, :] = np.log2(X[i, :] + 1) * real_entropy
        
    
    svd = TruncatedSVD(n_components=15, n_iter=7)
    svd.fit(X)
    print(svd.singular_values_)
    print(svd.transform(X).shape)
    
    # Xk = normalize(svd.transform(X), axis=1, norm="l2")
    Xk = svd.transform(X)
#    print("*______*")
#    print(X.A)
#    print(Xk[0, :])
#    print("*______*")
    
    
    S = Xk @ Xk.T
    print(S.shape)
    
#    print(type(S))
    np.savetxt("S_text.csv", S, delimiter=",")
    
    plt.imshow(S, cmap='hot', interpolation='nearest')
    plt.show()
    
#    print(len(vectorizer.get_feature_names()))
#    print(vectorizer.get_feature_names())
    
    
    


if __name__ == "__main__":
    main()