from Data.Load_data import importIV, importB, importDM, importMI, add_review, clean_reviews
import pickle
import random
from features.process_text.tokenize import word_tokenize_scikit, ngrams_tokenize, add_category_B, add_category_DM, add_category_IV, add_category_MI
from Data.Load_data import class1, class2, class3, class4, class5
from Data.Load_data import  remove_nouns
from sklearn.decomposition import TruncatedSVD
from features.process_reviews.feature_weighting import tfidf_bow, ngrams_tfidf
from models.supervised_classification import Classification, prediction
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from evaluation.metrics import f1_score, accuracy, precision, recall
import pandas as pd
import matplotlib.pyplot as plt


#Importing each categories dataset

Dataset_instant_video = importIV('reviews_Amazon_Instant_Video.json.gz')

Dataset_baby = importB('reviews_Baby.json.gz')

Dataset_music_instruments = importMI('reviews_Music_Instruments.json.gz')

Dataset_digital_music = importDM('reviews_Digital_Music.json.gz')

# Building necessary sample from each category


#Instant Video

data_instant_video_1 = random.choices(class1(Dataset_instant_video), k = 20000)
data_instant_video_2 = random.choices(class2(Dataset_instant_video), k = 20000)
data_instant_video_3 = random.choices(class3(Dataset_instant_video), k = 20000)
data_instant_video_4 = random.choices(class4(Dataset_instant_video), k = 20000)
data_instant_video_5 = random.choices(class5(Dataset_instant_video), k = 20000)

data_instant_video = []
data_instant_video.extend(data_instant_video_1)
data_instant_video.extend(data_instant_video_2)
data_instant_video.extend(data_instant_video_3)
data_instant_video.extend(data_instant_video_4)
data_instant_video.extend(data_instant_video_5)


#Baby

data_baby_1 = random.choices(class1(Dataset_baby), k = 20000)
data_baby_2 = random.choices(class2(Dataset_baby), k = 20000)
data_baby_3 = random.choices(class3(Dataset_baby), k = 20000)
data_baby_4 = random.choices(class4(Dataset_baby), k = 20000)
data_baby_5 = random.choices(class5(Dataset_baby), k = 20000)

data_baby = []
data_baby.extend(data_baby_1)
data_baby.extend(data_baby_2)
data_baby.extend(data_baby_3)
data_baby.extend(data_baby_4)
data_baby.extend(data_baby_5)


#Music Instruments


data_music_instruments_1 = random.choices(class1(Dataset_music_instruments), k = 20000)
data_music_instruments_2 = random.choices(class2(Dataset_music_instruments), k = 20000)
data_music_instruments_3 = random.choices(class3(Dataset_music_instruments), k = 20000)
data_music_instruments_4 = random.choices(class4(Dataset_music_instruments), k = 20000)
data_music_instruments_5 = random.choices(class5(Dataset_music_instruments), k = 20000)

data_music_instruments = []
data_music_instruments.extend(data_music_instruments_1)
data_music_instruments.extend(data_music_instruments_2)
data_music_instruments.extend(data_music_instruments_3)
data_music_instruments.extend(data_music_instruments_4)
data_music_instruments.extend(data_music_instruments_5)

#Digital Music

data_digital_music_1 = random.choices(class1(Dataset_digital_music), k = 20000)
data_digital_music_2 = random.choices(class2(Dataset_digital_music), k = 20000)
data_digital_music_3 = random.choices(class3(Dataset_digital_music), k = 20000)
data_digital_music_4 = random.choices(class4(Dataset_digital_music), k = 20000)
data_digital_music_5 = random.choices(class5(Dataset_digital_music), k = 20000)

data_digital_music = []
data_digital_music.extend(data_digital_music_1)
data_digital_music.extend(data_digital_music_2)
data_digital_music.extend(data_digital_music_3)
data_digital_music.extend(data_digital_music_4)
data_digital_music.extend(data_digital_music_5)

#Adding the category column
data_instant_video = add_category_IV(data_instant_video)
data_baby = add_category_B(data_baby)
data_digital_music = add_category_DM(data_digital_music)
data_music_instruments = add_category_MI(data_music_instruments)


# Building Complete dataset

Data = []

Data.extend(data_instant_video)

Data.extend(data_baby)

Data.extend(data_digital_music)

Data.extend(data_music_instruments)

#Creating and cleaning test, training and sample data

test_data = random.choices(Data, k = 120000)
train_data = random.choices(Data, k = 280000)


train_data = add_review(train_data)
train_data_cleaned = clean_reviews(train_data)


test_data = add_review(test_data)
test_data_cleaned = clean_reviews(test_data)


train_data_cleaned_sample = random.choices(train_data_cleaned, k = 1000)

"""Export Train Data with Pickle"""
with open('Data\\Pickle\\train_data_cleaned.pickle', 'wb') as handle:
   pickle.dump(train_data_cleaned, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Sample Data with Pickle"""
with open('Data\\Pickle\\train_data_cleaned_sample.pickle', 'wb') as handle:
   pickle.dump(train_data_cleaned_sample, handle, protocol=pickle.HIGHEST_PROTOCOL)   
   
"""Export Test Data with Pickle"""
with open('Data\\Pickle\\test_data_cleaned_.pickle', 'wb') as handle:
   pickle.dump(test_data_cleaned, handle, protocol=pickle.HIGHEST_PROTOCOL)   



"""Import Train Data with Pickle"""
with open('Data\\Pickle\\train_data_cleaned.pickle', 'rb') as handle:
    train_data_cleaned = pickle.load(handle)

"""Import Test Data with Pickle"""
with open('Data\\Pickle\\test_data_cleaned_.pickle', 'rb') as handle:
    test_data_cleaned = pickle.load(handle)

"""Import Sample Data with Pickle"""
with open('Data\\Pickle\\train_data_cleaned_sample.pickle', 'rb') as handle:
    train_data_cleaned_sample = pickle.load(handle)


#Creating review and category lists, and removing empty reviews

review_text_list_sample = [review.reviewtext_cleaned for review in train_data_cleaned_sample]
review_text_list_sample = [' ' if v is None else v for v in review_text_list_sample]

review_text_list_test = [review.reviewtext_cleaned for review in test_data_cleaned]
review_text_list_test = [' ' if v is None else v for v in review_text_list_test]

review_text_list = [review.reviewtext_cleaned for review in train_data_cleaned]
review_text_list = [' ' if v is None else v for v in review_text_list]

review_category_list = [review.category for review in train_data_cleaned]
review_category_list_test = [review.category for review in test_data_cleaned]

review_text_list_nouns = remove_nouns(review_text_list)
review_text_list_nouns_sample = remove_nouns(review_text_list_sample)
review_text_list_nouns_test = remove_nouns(review_text_list_test)


#Seeing cleaned text

for i in review_text_list:
       print(i)





# # Tokenization (BOW)

bow_index, bow_vectorizer = word_tokenize_scikit(review_text_list)
bow_index_sample, bow_vectorizer_sample = word_tokenize_scikit(review_text_list_sample)

bigram_index_sample, bigram_vectorizer_sample = ngrams_tokenize(review_text_list_sample, 1, 2)
bigram_index, bigram_vectorizer = ngrams_tokenize(review_text_list, 1, 2)

bow_index_nouns_sample, bow_vectorizer_nouns_sample = word_tokenize_scikit(review_text_list_nouns_sample)
bow_index_nouns, bow_vectorizer_nouns = word_tokenize_scikit(review_text_list_nouns)

bigramonly_index_sample, bigramonly_vectorizer_sample = ngrams_tokenize(review_text_list_sample, 2, 2)
bigramonly_index, bigramonly_vectorizer = ngrams_tokenize(review_text_list, 2, 2)

#Top ten words graphs

array = bow_index_sample.toarray()
dfbow = pd.DataFrame(array)
words = bow_vectorizer_sample.get_feature_names()
dfbow.columns = words
sum = dfbow.sum()
topsum = sum.nlargest(n=10)
topsum.plot()
plt.show()



array2 = bow_index_nouns_sample.toarray()
dfbow2 = pd.DataFrame(array2)
words2 = bow_vectorizer_nouns_sample.get_feature_names()
dfbow2.columns = words2
sum2 = dfbow2.sum()
topsum2 = sum2.nlargest(n=10)
topsum2
plt.show()


array3 = bigram_index_sample.toarray()
dfbow3 = pd.DataFrame(array3)
words3 = bigram_vectorizer_sample.get_feature_names()
dfbow3.columns = words3
sum3 = dfbow3.sum()
topsum3 = sum3.nlargest(n=10)
topsum3.plot()
plt.show()


array0 = bigramonly_index_sample.toarray()
dfbow0 = pd.DataFrame(array3)
words0 = bigramonly_vectorizer_sample.get_feature_names()
dfbow0.columns = words0
sum0 = dfbow0.sum()
topsum0 = sum0.nlargest(n=10)
topsum0.plot()
plt.show()

# # TFIDF
tfidf_index, tfidf_vectorizer = tfidf_bow(review_text_list)
tfidf_index_sample, tfidf_vectorizer_sample = tfidf_bow(review_text_list_sample)

bigramtfidf_index, bigramtfidf_vectorizer = ngrams_tfidf(review_text_list, 1, 2)
bigramtfidf_index_sample, bigramtfidf_vectorizer_sample = ngrams_tfidf(review_text_list_sample, 1, 2)

tfidf_index_nouns, tfidf_vectorizer_nouns = tfidf_bow(review_text_list_nouns)
tfidf_index_nouns_sample, tfidf_vectorizer_nouns_sample = tfidf_bow(review_text_list_nouns_sample)

bigramonlytfidf_index, bigramonlytfidf_vectorizer = ngrams_tfidf(review_text_list, 2, 2)
bigramonlytfidf_index_sample, bigramonlytfidf_vectorizer_sample = ngrams_tfidf(review_text_list_sample, 2, 2)


tfidf_index_test = tfidf_vectorizer.transform(review_text_list_test)

bigramtfidf_index_test = bigramtfidf_vectorizer.transform(review_text_list_test)

tfidf_index_nouns_test = tfidf_vectorizer_nouns.transform(review_text_list_nouns_test)

bigramonlytfidf_index_test = bigramonlytfidf_vectorizer.transform(review_text_list_test)


#tfidf plots

array4 = tfidf_index_sample.toarray()
dfbow4 = pd.DataFrame(array4)
words4 = tfidf_vectorizer_sample.get_feature_names()
dfbow4.shape
dfbow4.columns = words4
sum4 = dfbow4.sum()
topsum4= sum4.nlargest(n=10)
topsum4.plot()
plt.show()


array5 = tfidf_index_nouns_sample.toarray()
dfbow5 = pd.DataFrame(array5)
words5 = tfidf_vectorizer_nouns_sample.get_feature_names()
dfbow5.shape
dfbow5.columns = words5
sum5 = dfbow4.sum()
topsum5= sum5.nlargest(n=10)
topsum5.plot()
plt.show()



array6 = bigramtfidf_index_sample.toarray()
dfbow6 = pd.DataFrame(array6)
words6 = bigramtfidf_vectorizer_sample.get_feature_names()
dfbow6.shape
dfbow6.columns = words6
sum6 = dfbow6.sum()
topsum6 = sum6.nlargest(n=10)
topsum6.plot()
plt.show()


array7 = bigramonlytfidf_index_sample.toarray()
dfbow7 = pd.DataFrame(array7)
words7 = bigramonlytfidf_vectorizer_sample.get_feature_names()
dfbow7.shape
dfbow7.columns = words7
sum7 = dfbow7.sum()
topsum7 = sum7.nlargest(n=10)
topsum7.plot()
plt.show()


# LSA

#Regular BOW

lsa = TruncatedSVD(n_components=100)
lsa_matrix = lsa.fit_transform(tfidf_index)

lsa2 = lsa.fit(tfidf_index)
lsa_matrix = lsa_matrix[:,[0,1,2,4,5]]
lsa_matrix_test = lsa_matrix_test[:,[0,1,2,4,5]]
lsa_matrix_test = lsa2.transform(tfidf_index_test)

names=tfidf_vectorizer.get_feature_names()

#Top ten words in each concept
for i, comp in enumerate(lsa.components_):
    termsInComp = zip (names,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print (" ")


for i, comp in enumerate(lsa.components_):
    termsInComp = zip (names,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=False) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print (" ")



#Singular values Representaion of categories per concept

lsa2.singular_values_
df = lsa_matrix
df = pd.DataFrame(df)
df['category'] = review_category_list
df1=df[[0, 1, 2, 3, 4, 5, 'category']]
df1.groupby(['category',]).mean()


#BOW with only nouns

lsa_nouns = TruncatedSVD(n_components=100)
lsa_nouns_matrix = lsa_nouns.fit_transform(tfidf_index_nouns)
lsa_nouns2 = lsa_nouns.fit(tfidf_index_nouns)
lsa_nouns_matrix= lsa_nouns_matrix[:,[0,1,2,3,4]]
lsa_nouns_matrix_test = lsa_nouns2.transform(tfidf_index_nouns_test)
lsa_nouns_matrix_test= lsa_nouns_matrix_test[:,[0,1,2,3,4]]
names2=tfidf_vectorizer_nouns.get_feature_names()


#Top ten words for each concept

for i, comp in enumerate(lsa_nouns.components_):
    termsInComp = zip (names2,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print (" ")

for i, comp in enumerate(lsa_nouns.components_):
    termsInComp = zip (names2,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=False) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print (" ")


#Representation of each category in each concept
lsa_nouns2.singular_values_
df2 = lsa_nouns_matrix
df2 = pd.DataFrame(df2)
df2['category'] = review_category_list
df3=df2[[0, 1, 2, 3, 4, 5, 'category']]
df3.groupby(['category',]).mean()



#Words with Bigrams

lsa_bigram = TruncatedSVD(n_components=50)
lsa_bigram2 = lsa_bigram.fit(bigramtfidf_index)
lsa_bigram_matrix = lsa_bigram.fit_transform(bigramtfidf_index)
lsa_bigram_matrix = lsa_bigram_matrix[:,[0,1,2,3,4]]
lsa_bigram_matrix_test = lsa_bigram_matrix_test[:,[0,1,2,3,4]]
lsa_bigram_matrix_test = lsa_bigram2.transform(bigramtfidf_index_test)
names3=bigramtfidf_vectorizer.get_feature_names()


#Top ten words for each concept
for i, comp in enumerate(lsa_bigram2.components_):
    termsInComp = zip (names3,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print (" ")

    print (" ")

for i, comp in enumerate(lsa_bigram.components_):
    termsInComp = zip (names3,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=False) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print(" ")

#Singular values and representation of each category i each concept

lsa_bigram2.singular_values_
df3 = lsa_bigram_matrix
df3 = pd.DataFrame(df3)
df3['category'] = review_category_list
df4=df3[[0, 1, 2, 3, 4, 'category']]
df4.groupby(['category',]).mean()


# Bigrams only


lsa_bigramonly = TruncatedSVD(n_components=50)
lsa_bigramonly2 = lsa_bigramonly.fit(bigramonlytfidf_index)
lsa_bigramonly_matrix = lsa_bigramonly.fit_transform(bigramonlytfidf_index)
lsa_bigramonly_matrix = lsa_bigramonly_matrix[:,[0,1,2,3,4]]
lsa_bigramonly_matrix_test = lsa_bigramonly_matrix_test[:,[0,1,2,3,4]]
lsa_bigramonly_matrix_test = lsa_bigramonly2.transform(bigramtfidf_index_test)
names4=bigramonlytfidf_vectorizer.get_feature_names()


#Top ten words for each concept

for i, comp in enumerate(lsa_bigramonly2.components_):
    termsInComp = zip (names3,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print (" ")

    print (" ")

for i, comp in enumerate(lsa_bigramonly.components_):
    termsInComp = zip (names3,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=False) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print(" ")

#Singular values and representation of each category i each concept

lsa_bigramonly2.singular_values_
df3 = lsa_bigramonly_matrix
df3 = pd.DataFrame(df3)
df3['category'] = review_category_list
df4=df3[[0, 1, 2, 3, 4, 'category']]
df4.groupby(['category',]).mean()




"""
Classification algorithms
"""

# Regular BOW

mnb_matrix = minmax_scale(lsa_matrix)

mnb_matrix_test = minmax_scale(lsa_matrix_test)


mnb_model = Classification(MultinomialNB(),mnb_matrix, review_category_list)

svc_model =  Classification(SVC(),lsa_matrix, review_category_list)

svc_model_class =  Classification(SVC(C=1.0, kernel='linear'),lsa_matrix, review_category_list)


predicted_mnb = prediction(mnb_model, mnb_matrix_test)

predicted_svc = prediction(svc_model, lsa_matrix_test)

predicted_svc_class = prediction(svc_model_class, lsa_matrix_test)


#bow with nouns

mnb_matrix_nouns = minmax_scale(lsa_nouns_matrix)

mnb_matrix_nouns_test = minmax_scale(lsa_nouns_matrix_test)

mnb_model_nouns = Classification(MultinomialNB(),mnb_matrix_nouns, review_category_list)

svc_model_nouns =  Classification(SVC(),lsa_nouns_matrix, review_category_list)

svc_model_nouns_class =  Classification(SVC(C=1.0, kernel='linear'),lsa_nouns_matrix, review_category_list)



predicted_mnb_nouns = prediction(mnb_model_nouns, mnb_matrix_nouns_test)

predicted_svc_nouns = prediction(svc_model_nouns, lsa_nouns_matrix_test)

predicted_svc_nouns_class = prediction(svc_model_nouns_class, lsa_nouns_matrix_test)




#Words with Bigrams

mnb_matrix_bigrams = minmax_scale(lsa_bigram_matrix)

mnb_matrix_bigrams_test = minmax_scale(lsa_bigram_matrix_test)

mnb_model_bigrams = Classification(MultinomialNB(),mnb_matrix_bigrams, review_category_list)

svc_model_bigrams_class =  Classification(SVC(C=1.0, kernel='linear'),lsa_bigram_matrix, review_category_list)

svc_model_bigrams =  Classification(SVC(),lsa_bigram_matrix, review_category_list)



predicted_mnb_bigrams = prediction(mnb_model_bigrams, mnb_matrix_bigrams_test)

predicted_svc_bigrams_class = prediction(svc_model_bigrams_class, lsa_bigram_matrix_test)

predicted_svc_bigrams = prediction(svc_model_bigrams, lsa_bigram_matrix_test)



#Bigrams only

mnb_matrix_bigramsonly = minmax_scale(lsa_bigramonly_matrix)

mnb_matrix_bigramsonly_test = minmax_scale(lsa_bigramonly_matrix_test)

mnb_model_bigramsonly = Classification(MultinomialNB(),mnb_matrix_bigramsonly, review_category_list)

svc_model_bigramsonly_class =  Classification(SVC(C=1.0, kernel='linear'),lsa_bigramonly_matrix, review_category_list)

svc_model_bigramsonly =  Classification(SVC(),lsa_bigramonly_matrix, review_category_list)



predicted_mnb_bigramsonly = prediction(mnb_model_bigramsonly, mnb_matrix_bigramsonly_test)

predicted_svc_bigramsonly_class = prediction(svc_model_bigramsonly_class, lsa_bigramonly_matrix_test)

predicted_svc_bigramsonly = prediction(svc_model_bigramsonly, lsa_bigramonly_matrix_test)



"""
    Evaluation metrics
"""

#BOW

f1_score_mnb = f1_score(review_category_list_test, predicted_mnb, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_mnb))


accuracy_mnb = accuracy(review_category_list_test, predicted_mnb)
#
print('ACCURACY RESULT: '+str(accuracy_mnb))


precision_mnb = precision(review_category_list_test, predicted_mnb, average='macro')
#
print('PRECISION RESULT: '+str(precision_mnb))


recall_mnb = recall(review_category_list_test, predicted_mnb, average='macro')
#
print('RECALL RESULT: '+str(recall_mnb))






f1_score_svc = f1_score(review_category_list_test, predicted_svc, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_svc))


accuracy_svc = accuracy(review_category_list_test, predicted_svc)
#
print('ACCURACY RESULT: '+str(accuracy_svc))


precision_svc = precision(review_category_list_test, predicted_svc, average='macro')
#
print('PRECISION RESULT: '+str(precision_svc))


recall_svc = recall(review_category_list_test, predicted_svc, average='macro')
#
print('RECALL RESULT: '+str(recall_svc))




f1_score_svc_class = f1_score(review_category_list_test, predicted_svc_class, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_svc_class))


accuracy_svc_class = accuracy(review_category_list_test, predicted_svc_class)
#
print('ACCURACY RESULT: '+str(accuracy_svc_class))


precision_svc_class = precision(review_category_list_test, predicted_svc_class, average='macro')
#
print('PRECISION RESULT: '+str(precision_svc_class))


recall_svc_class = recall(review_category_list_test, predicted_svc_class, average='macro')
#
print('RECALL RESULT: '+str(recall_svc_class))


#BOW (Nouns)



f1_score_mnb_nouns = f1_score(review_category_list_test, predicted_mnb_nouns, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_mnb_nouns))


accuracy_mnb_nouns = accuracy(review_category_list_test, predicted_mnb_nouns)
#
print('ACCURACY RESULT: '+str(accuracy_mnb_nouns))


precision_mnb_nouns = precision(review_category_list_test, predicted_mnb_nouns, average='macro')
#
print('PRECISION RESULT: '+str(precision_mnb_nouns))


recall_mnb_nouns = recall(review_category_list_test, predicted_mnb_nouns, average='macro')
#
print('RECALL RESULT: '+str(recall_mnb_nouns))





f1_score_svc_nouns = f1_score(review_category_list_test, predicted_svc_nouns, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_svc_nouns))


accuracy_svc_nouns = accuracy(review_category_list_test, predicted_svc_nouns)
#
print('ACCURACY RESULT: '+str(accuracy_svc_nouns))


precision_svc_nouns = precision(review_category_list_test, predicted_svc_nouns, average='macro')
#
print('PRECISION RESULT: '+str(precision_svc_nouns))


recall_svc_nouns = recall(review_category_list_test, predicted_svc_nouns, average='macro')
#
print('RECALL RESULT: '+str(recall_svc_nouns))




f1_score_svc_nouns_class = f1_score(review_category_list_test, predicted_svc_nouns_class, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_svc_nouns_class))


accuracy_svc_nouns_class = accuracy(review_category_list_test, predicted_svc_nouns_class)
#
print('ACCURACY RESULT: '+str(accuracy_svc_nouns_class))


precision_svc_nouns_class = precision(review_category_list_test, predicted_svc_nouns_class, average='macro')
#
print('PRECISION RESULT: '+str(precision_svc_nouns_class))


recall_svc_nouns_class = recall(review_category_list_test, predicted_svc_nouns_class, average='macro')
#
print('RECALL RESULT: '+str(recall_svc_nouns_class))



#Words with Bigrams



f1_score_mnb_bigrams = f1_score(review_category_list_test, predicted_mnb_bigrams, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_mnb_bigrams))


accuracy_mnb_bigrams = accuracy(review_category_list_test, predicted_mnb_bigrams)
#
print('ACCURACY RESULT: '+str(accuracy_mnb_bigrams))


precision_mnb_bigrams = precision(review_category_list_test, predicted_mnb_bigrams, average='macro')
#
print('PRECISION RESULT: '+str(precision_mnb_bigrams))


recall_mnb_bigrams = recall(review_category_list_test, predicted_mnb_bigrams, average='macro')
#
print('RECALL RESULT: '+str(recall_mnb_bigrams))





f1_score_svc_bigrams = f1_score(review_category_list_test, predicted_svc_bigrams, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_svc_bigrams))


accuracy_svc_bigrams = accuracy(review_category_list_test, predicted_svc_bigrams)
#
print('ACCURACY RESULT: '+str(accuracy_svc_bigrams))


precision_svc_bigrams = precision(review_category_list_test, predicted_svc_bigrams, average='macro')
#
print('PRECISION RESULT: '+str(precision_svc_bigrams))


recall_svc_bigrams = recall(review_category_list_test, predicted_svc_bigrams, average='macro')
#
print('RECALL RESULT: '+str(recall_svc_bigrams))




f1_score_svc_bigrams_class = f1_score(review_category_list_test, predicted_svc_bigrams_class, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_svc_bigrams_class))


accuracy_svc_bigrams_class = accuracy(review_category_list_test, predicted_svc_bigrams_class)
#
print('ACCURACY RESULT: '+str(accuracy_svc_bigrams_class))


precision_svc_bigrams_class = precision(review_category_list_test, predicted_svc_bigrams_class, average='macro')
#
print('PRECISION RESULT: '+str(precision_svc_bigrams_class))


recall_svc_bigrams_class = recall(review_category_list_test, predicted_svc_bigrams_class, average='macro')
#
print('RECALL RESULT: '+str(recall_svc_bigrams_class))




#Bigrams Only



f1_score_mnb_bigramsonly = f1_score(review_category_list_test, predicted_mnb_bigramsonly, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_mnb_bigrams))


accuracy_mnb_bigramsonly = accuracy(review_category_list_test, predicted_mnb_bigramsonly)
#
print('ACCURACY RESULT: '+str(accuracy_mnb_bigrams))


precision_mnb_bigramsonly = precision(review_category_list_test, predicted_mnb_bigramsonly, average='macro')
#
print('PRECISION RESULT: '+str(precision_mnb_bigrams))


recall_mnb_bigramsonly = recall(review_category_list_test, predicted_mnb_bigramsonly, average='macro')
#
print('RECALL RESULT: '+str(recall_mnb_bigrams))





f1_score_svc_bigramsonly = f1_score(review_category_list_test, predicted_svc_bigramsonly, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_svc_bigrams))


accuracy_svc_bigramsonly = accuracy(review_category_list_test, predicted_svc_bigramsonly)
#
print('ACCURACY RESULT: '+str(accuracy_svc_bigrams))


precision_svc_bigramsonly = precision(review_category_list_test, predicted_svc_bigramsonly, average='macro')
#
print('PRECISION RESULT: '+str(precision_svc_bigrams))


recall_svc_bigramsonly = recall(review_category_list_test, predicted_svc_bigramsonly, average='macro')
#
print('RECALL RESULT: '+str(recall_svc_bigrams))




f1_score_svc_bigramsonly_class = f1_score(review_category_list_test, predicted_svc_bigramsonly_class, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_svc_bigrams_class))


accuracy_svc_bigramsonly_class = accuracy(review_category_list_test, predicted_svc_bigramsonly_class)
#
print('ACCURACY RESULT: '+str(accuracy_svc_bigrams_class))


precision_svc_bigramsonly_class = precision(review_category_list_test, predicted_svc_bigramsonly_class, average='macro')
#
print('PRECISION RESULT: '+str(precision_svc_bigrams_class))


recall_svc_bigramsonly_class = recall(review_category_list_test, predicted_svc_bigramsonly_class, average='macro')
#
print('RECALL RESULT: '+str(recall_svc_bigrams_class))


