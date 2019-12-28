import nltk
from nltk import FreqDist
nltk.download('stopwords') # run this one time
import pandas as pd
pd.set_option("display.max_colwidth", 200)
import numpy as np
import re
import spacy
import gensim
from gensim import corpora
# libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from nltk.corpus import stopwords

df = pd.read_excel('F:\gn-python\excel2.xlsx',lines=True)
# df = pd.read_json('Automotive_5.json', lines=True)
df.head()

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
# function to plot most frequent terms
def freq_words(xx, terms = 30):
    # all_words = ''.join(str([text for text in (x for x in xx)]))
    all_words = ''.join(str([text for text in xx]))
    # print(all_words)
    all_words = all_words.split(' ')
    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = terms)
    plt.figure(figsize=(15,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()

# function to remove stopwords
def remove_stopwords(rev):
    all_words = ''.join(str([text for text in rev]))
    all_words = all_words.split(' ')
    rev_new = " ".join([i for i in all_words if i not in stop_words])
    return rev_new
# remove unwanted characters, numbers and symbols

def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
       output = []
       for sent in texts:
             doc = nlp(" ".join(sent))
             output.append([token.lemma_ for token in doc if token.pos_ in tags])
       return output



for i in range(3,77,1):
    df['字段'+str(i)] = df['字段'+str(i)].str.replace("[^a-zA-Z#]"," ")
stop_words = stopwords.words('english')
# remove short words (length < 3)
for i in range(3,77,1):
    df['字段'+str(i)] = df['字段'+str(i)].apply(lambda x: ' '.join([w for w in str(x).split(' ') if len(w)>2]))
# remove stopwords from the text
reviewsl = []
for i in range(3,77,1):
    reviewsl.append(remove_stopwords(df['字段'+str(i)]))
# freq_words(reviews)
reviews_res = []
reviews_dic = []
for reviews in reviewsl:
    # make entire text lowercase
    reviews = [r.lower() for r in [reviews]]
    tokenized_reviews = pd.Series(reviews).apply(lambda x: str(x).split(' '))
    reviews_2 = lemmatization(tokenized_reviews)
    reviews_3 = []
    for i in range(len(reviews_2)):
        while 'nan' in reviews_2[i]:
            reviews_2[i].remove('nan')
        while 'kickstarter' in reviews_2[i]:
            reviews_2[i].remove('kickstarter')
        while 'com' in reviews_2[i]:
            reviews_2[i].remove('com')
        while 'www' in reviews_2[i]:
            reviews_2[i].remove('www')
        while 'https' in reviews_2[i]:
            reviews_2[i].remove('https')
        while 'backer' in reviews_2[i]:
            reviews_2[i].remove('backer')
        reviews_res.append(' '.join(reviews_2[i]))
        print(reviews_2[i])
        reviews_dic.append(reviews_2[i])
freq_words(reviews_res, 15)


dictionary = corpora.Dictionary(reviews_dic)
doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_dic]

# Creating the object for LDA model using gensim library

LDA = gensim.models.ldamodel.LdaModel


# Build LDA model

lda_model = LDA(corpus=doc_term_matrix,

                                   id2word=dictionary,

                                   num_topics=3,

                                   random_state=100,

                                   chunksize=1000,

                                   passes=50)

print(lda_model.print_topics(num_topics=3, num_words=5))

#
# # Visualize the topics
#
# pyLDAvis.enable_notebook()
#
# vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
#
# pyLDAvis.show(vis)