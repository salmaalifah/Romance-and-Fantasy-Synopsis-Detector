from nltk.corpus import  wordnet, stopwords
import random, nltk, pickle, string
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier, accuracy
from nltk.tokenize import wordpunct_tokenize as wt
from nltk.probability import FreqDist
from nltk.tag import pos_tag as pt
from nltk.chunk import ne_chunk
import bs4 as BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
from heapq import nlargest

from nltk.util import pr



def get_model():
    classifier = None
    try:
        file_model = open('model.pikel', 'rb')
        classifier = pickle.load(file_model)
    except:
        classifier = train_data()
        file_model = open('model.pikel', 'wb')
        pickle.dump(classifier, file_model)

    file_model.close()
    return classifier


def preprocessing_dataset(sentences):
    word_list = []
    porter_stemmer = PorterStemmer()
    lema = WordNetLemmatizer()
    eng_stopwords = stopwords.words('english')

    for word in wt(sentences):
        preprocessing_word = lema.lemmatize(word)
        preprocessing_word = porter_stemmer.stem(preprocessing_word)

        if preprocessing_word not in eng_stopwords and preprocessing_word not in string.punctuation:
            word_list.append(word)

    return word_list

def train_data():
    file_dataset_fantasy = open('fantasy_dataset.txt', encoding="utf8")
    fantasy = [line.rstrip('\n') for line in file_dataset_fantasy]
    
    file_dataset_romance = open('romance_dataset.txt', encoding="utf8")
    romance = [line.rstrip('\n') for line in file_dataset_romance]
    
    random.shuffle(fantasy)
    random.shuffle(romance)
    fantasy = fantasy[:190]
    romance = romance[:190]

    lema_fantasy = [y for x in fantasy for y in preprocessing_dataset(x)]
    lema_romance= [y for x in romance for y in preprocessing_dataset(x)]

    dataset = [({i:True},"fantasy") for i in lema_fantasy] + [({i:True},"romance") for i in lema_romance]

    train_count = int(len(dataset) * 80/100)
    i = 0
    final_acc = 0
    classifier = None
    max_accuracy = 90
    epoch = 50
    
    while final_acc < max_accuracy and i < epoch +1:
        random.shuffle(dataset)
        train_data = dataset[:train_count]
        test_data = dataset[train_count:]

        temp_classifier = NaiveBayesClassifier.train(train_data)
        temp_accuracy = accuracy(temp_classifier, test_data) * 100

        if temp_accuracy > final_acc:
            classifier = temp_classifier
            final_acc = temp_accuracy

        print('Epoch: {}, Accuracy: {}'.format(i, final_acc))
        i += 1

    classifier.show_most_informative_features()
    return classifier

sentence_synopsis = []
def input_new_synopsis():
    classifier = get_model()
    sentence= ""
    while(len(sentence)< 15):
        sentence = input("Input sysnopsis[Must be more than 15 characters]: ")

    sentence_synopsis.append(sentence)
    words = preprocessing_dataset(sentence)

    fan = 0
    rom = 0
    for word in words:
        result = classifier.classify(FreqDist([word]))
        if result == "fantasy":
            fan += 1
        if result == "romance":
            rom += 1

    if(fan > rom):
        print("It's a Fantasy book\n")
    elif(fan < rom):
        print("It's a Romance book\n")
    elif(fan == rom):
        print("It's a Fantasy and Romance book\n")

def synopsis_analysis():
   
    if(len(sentence_synopsis) != 0):
        print("\n               Word                      Tag              Synonym                   Antonyms                 Total Frequency")
        word = preprocessing_dataset(sentence_synopsis[len(sentence_synopsis)-1])
        fd = FreqDist(word)
        for word, count in fd.most_common():
            tagged = pt(word)
            for n in tagged:
                tag = ""
                tag = n[1]
            syno = "-"
            anto = "-"
            synsets = wordnet.synsets(word)
            for s in synsets:
                for synonym in s.lemmas():
                    for antonym in synonym.antonyms():
                        syno = synonym.name()
                        anto = antonym.name()

            print("{: >20}    {: >20}   {: >20}       {: >20}    {: >20}".format(word, tag,syno, anto, count))
          
        print("\n")
       
    else:
        print("Please input a synopsis first!\n")

def generate_tree():
    if(len(sentence_synopsis) != 0):
        word = preprocessing_dataset(sentence_synopsis[len(sentence_synopsis)-1])
        tag = pt(word)
        n_chunk = ne_chunk(tag)
        n_chunk.draw()
    else:
        print("Please input a synopsis first!\n")



def create_summary():
    if(len(sentence_synopsis) != 0):
        sentence = sentence_synopsis[len(sentence_synopsis)-1]
        tokens = word_tokenize(sentence)
        stop_words = stopwords.words('english')
        word_frequencies = {}
        for word in tokens:
            if word.lower() not in stop_words:
                if word.lower() not in punctuation:
                    if word not in word_frequencies.keys():
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1

        print(word_frequencies)
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / max_frequency

        sent_token = sent_tokenize(sentence)

        sentence_scores = {}
        for sent in sent_token:
            sentence = sent.split(" ")
            for word in sentence:
                if word.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.lower()]

        print(sentence_scores)
        select_length = int(len(sent_token) * 0.9)
        print(select_length)
        summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
        final_summary = [word for word in summary]
        summary = ' '.join(final_summary)

        print("\nYour summary result is")
        print("======================")
        print(summary)
        print("\n")

    else:
        print("Please input a synopsis first!\n")

choice = -1
while choice !=5:
    choice = -1
    print("Current Synopsis")
    print("-----------------")
    if(len(sentence_synopsis)==0):
        print("No synopis - Please input a synopsis!")
    else:
        print(sentence_synopsis[len(sentence_synopsis)-1])

    print("\nRomance and Fantasy Synopsis Detector")
    print("-------------------------------------")
    print("1. Input New Synopsis")
    print("2. View Synopsis Analysis")
    print("3. Generate Tree")
    print("4. Create Summary")
    print("5. Exit")
    
    while(choice < 1 or choice > 5):
        try:
            choice = int(input("Choose[1-5]: "))
        except:
            print("Input must be number!")
            choice = -1


    if(choice == 1):
        input_new_synopsis()

    elif(choice == 2):
        synopsis_analysis()

    elif(choice == 3):
        generate_tree()

    elif(choice == 4):
        create_summary()
