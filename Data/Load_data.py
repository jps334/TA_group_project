import gzip
from review.review import create_review_from_dict
from Data.Get_file_path import get_data_path
from features.process_reviews.clean_reviews import clean_review
import nltk
from nltk.tokenize import word_tokenize as wt



""" - Functions toimport the full data set without restrictions"""
def load(path):
    g = gzip.open(get_data_path("01_raw\\reviews_Amazon_Instant_Video.json.gz"), 'rb')
    for l in g:
        yield eval(l)

def importIV(path):
    list = []
    for d in load(path):
        list.append(d)
    return list



def load2(path):
    g = gzip.open(get_data_path("01_raw\\reviews_Baby.json.gz"), 'rb')
    for l in g:
        yield eval(l)

def importB(path):
    list = []
    for d in load2(path):
        list.append(d)
    return list


def load3(path):
    g = gzip.open(get_data_path("01_raw\\reviews_Digital_Music.json.gz"), 'rb')
    for l in g:
        yield eval(l)

def importDM(path):
    list = []
    for d in load3(path):
        list.append(d)
    return list

def load4(path):
    g = gzip.open(get_data_path("01_raw\\reviews_Musical_Instruments.json.gz"), 'rb')
    for l in g:
        yield eval(l)

def importMI(path):
    list = []
    for d in load4(path):
        list.append(d)
    return list


#filtering by overall functions

def class1(Dataset):
    list = []
    for d in Dataset:
        if d.get('overall') == 1.0:
            list.append(d)
    return list

def class2(Dataset):
    list = []
    for d in Dataset:
        if d.get('overall') == 2.0:
            list.append(d)
    return list

def class3(Dataset):
    list = []
    for d in Dataset:
        if d.get('overall') == 3.0:
            list.append(d)
    return list

def class4(Dataset):
    list = []
    for d in Dataset:
        if d.get('overall') == 4.0:
            list.append(d)
    return list

def class5(Dataset):
    list = []
    for d in Dataset:
        if d.get('overall') == 5.0:
            list.append(d)
    return list

#Making reviews function

def add_review(review_dict_list):
    review_set = []
    for review_dict in review_dict_list:
        review = create_review_from_dict(review_dict)
        review_set.append(review)
    return review_set

def clean_reviews(reviews):
    cleaned_reviews = list()
    i = 0
    for review in reviews:
        if i % 100 == 0:
            print(i / len(reviews))
        review_cleaned = clean_review(review)
        cleaned_reviews.append(review_cleaned)
        i += 1
    return cleaned_reviews

# Checking and removing nouna


def check_nouns(sentence):
    tokenized = wt(sentence)
    tags = nltk.pos_tag(tokenized)
    return " ".join([i[0] for i in tags if i[1][0] == 'N' ])

def remove_nouns(list):
    list2 = []
    for string in list:
        list2.append(check_nouns(string))
    return list2

