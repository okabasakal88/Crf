import jpype
import re
import string
from nltk.tokenize import word_tokenize

# JVM / JAR Info
jvm_path = "/usr/lib/jvm/java-8-oracle/jre/lib/amd64/server/libjvm.so"

jvm_path = "C:\\Program files\\Java\\jre7\\bin\\server\\jvm.dll"

# Start JVM
jpype.startJVM(jvm_path, "-ea", "-Djava.class.path=/Users/Osman/PycharmProjects/Zemberek/zemberek-tum-2.0.jar")


Tr = jpype.JClass("net.zemberek.tr.yapi.TurkiyeTurkcesi")
Zemberek = jpype.JClass("net.zemberek.erisim.Zemberek")

tr = Tr()
zemberek = Zemberek(tr)

def Convert(string):
    li = list(string.split(" "))
    return li


def clean_text(text):
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text




with open('data.txt', 'r') as file:
    cumle = file.read().replace('\n', '')



liste = Convert(cumle)
finalliste = []
stopwordliste = []
file = open('C:\\Users\\Osman\\AppData\\Roaming\\nltk_data\\corpora\\conll2002\\deneme2.txt', "w", encoding="utf-8")
for i in range(0,len(liste)):
    try:
        yanit = zemberek.kelimeCozumle(liste[i])
        sonuc = str((yanit[0]))

        start = 'Kok: '
        end = 'tip'
        kok = sonuc[sonuc.find(start)+len(start):sonuc.rfind(end)]
        #print(kok)


        start = 'tip:'
        end = '}'
        sonuc = sonuc[sonuc.find(start)+len(start):sonuc.rfind(end)]
        if(sonuc=="ISIM" or sonuc=="SIFAT"):
            finalliste.append(kok)
        #print("Kelime: ",liste[i],"Tip: ",sonuc)
        s = "%s\t%s\t%s" % (liste[i], sonuc,"DEN")
        file.write(s)
        file.write("\n")
    except:
        #print(liste[i])
        s = "%s\t%s\t%s" % (liste[i],"SAYI","DEN")
        file.write(s)
        file.write("\n")
str1 = ' '.join(finalliste)
file.close()

import matplotlib.pyplot as plt
plt.style.use('ggplot')
# LIB
from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
# from sklearn.cross_validation import cross_val_score
# from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

nltk.corpus.conll2002.fileids()



train_sents = list(nltk.corpus.conll2002.iob_sents('train4.txt'))
test_sents = list(nltk.corpus.conll2002.iob_sents('deneme2.txt'))


print(train_sents[1])




def containDigit(token):
    return any(char.isdigit() for char in token)


def isPercent(token):
    if '%' in token or 'yüzde' in token or 'binde' in token or 'onda' in token:
        return True
    return False


def isTime(token):
    if 'saat' in token:
        return True
    ########################
    elif ':' in token:
        splittedWord = token.split(':')
        if containDigit(splittedWord[0]) and containDigit(splittedWord[1]):
            return True
    #######################
    return False


def checkCase(token):
    if token.islower():
        return 0
    elif token.isupper():
        return 1
    elif token.istitle():
        return 2
    else:
        return 3


def cleanWord(word):
    word = word.replace("#", "")
    word = word.replace("@", "")
    return word


def hasHashtag(word):
    if "#" in word:
        return True
    return False


def hasAt(word):
    if "@" in word:
        return True
    return False


def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

    # Common features for all words
    features = [
        'bias',
        'word.lower=%s' % word.lower(),
        'word[-3:]=%s' % word[-3:],
        'word[-4:]=%s' % word[-4:],
        'word[:4]=%s' % word[:4],
        'word[:5]=%s' % word[:5],
        'word.case=%s' % checkCase(word),
        'word.isnumeric=%s' % word.isnumeric(),
        'word.containDigit=%s' % containDigit(word),
        'postag=%s' % postag,
        'word.isPercent=%s' % isPercent(word),
        'word.isTime=%s' % isTime(word),
        'word.cleaned=%s' % cleanWord(word),
        'word.hashtag=%s' % hasHashtag(word),
        'word.at=%s' % hasAt(word)
    ]

    # Features for words that are not
    # at the beginning of a document
    if i > 1:
        word = doc[i - 2][0]
        postag = doc[i - 2][1]

        features.extend([
            '-2:word.lower=%s' % word.lower(),
            '-2:word[-3:]=%s' % word[-3:],
            '-2:word[-4:]=%s' % word[-4:],
            '-2:word[:4]=%s' % word[:4],
            '-2:word[:5]=%s' % word[:5],
            '-2:word.case=%s' % checkCase(word),
            '-2:word.isnumeric=%s' % word.isnumeric(),
            '-2:word.containDigit=%s' % containDigit(word),
            '-2:postag=%s' % postag,
            '-2:word.isPercent=%s' % isPercent(word),
            '-2:word.isTime=%s' % isTime(word),
            '-2:word.cleaned=%s' % cleanWord(word),
            '-2:word.hashtag=%s' % hasHashtag(word),
            '-2:word.at=%s' % hasAt(word)
        ])

    if i > 0:
        word = doc[i - 1][0]
        postag = doc[i - 1][1]

        features.extend([
            '-1:word.lower=%s' + word.lower(),
            '-1:word[-3:]=%s' % word[-3:],
            '-1:word[-4:]=%s' % word[-4:],
            '-1:word[:4]=%s' % word[:4],
            '-1:word[:5]=%s' % word[:5],
            '-1:word.case=%s' % checkCase(word),
            '-1:word.isnumeric=%s' % word.isnumeric(),
            '-1:word.containDigit=%s' % containDigit(word),
            '-1:postag=%s' % postag,
            '-1:word.isPercent=%s' % isPercent(word),
            '-1:word.isTime=%s' % isTime(word),
            '-1:word.cleaned=%s' % cleanWord(word),
            '-1:word.hashtag=%s' % hasHashtag(word),
            '-1:word.at=%s' % hasAt(word)
        ])

    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    # Features for words that are not
    # at the end of a document

    if i < len(doc) - 2:
        word = doc[i + 2][0]
        postag = doc[i + 2][1]

        features.extend([
            '+2:word.lower=%s' % word.lower(),
            '+2:word[-3:]=%s' % word[-3:],
            '+2:word[-4:]=%s' % word[-4:],
            '+2:word[:4]=%s' % word[:4],
            '+2:word[:5]=%s' % word[:5],
            '+2:word.case=%s' % checkCase(word),
            '+2:word.isnumeric=%s' % word.isnumeric(),
            '+2:word.containDigit=%s' % containDigit(word),
            '+2:postag=%s' % postag,
            '+2:word.isPercent=%s' % isPercent(word),
            '+2:word.isTime=%s' % isTime(word),
            '+2:word.cleaned=%s' % cleanWord(word),
            '+2:word.hashtag=%s' % hasHashtag(word),
            '+2:word.at=%s' % hasAt(word)
        ])

    if i < len(doc) - 1:
        word = doc[i + 1][0]
        postag = doc[i + 1][1]

        features.extend([
            '+1:word.lower=%s' + word.lower(),
            '+1:word[-3:]=%s' % word[-3:],
            '+1:word[-4:]=%s' % word[-4:],
            '+1:word[:4]=%s' % word[:4],
            '+1:word[:5]=%s' % word[:5],
            '+1:word.case=%s' % checkCase(word),
            '+1:word.isnumeric=%s' % word.isnumeric(),
            '+1:word.containDigit=%s' % containDigit(word),
            '+1:postag=%s' % postag,
            '+1:word.isPercent=%s' % isPercent(word),
            '+1:word.isTime=%s' % isTime(word),
            '+1:word.cleaned=%s' % cleanWord(word),
            '+1:word.hashtag=%s' % hasHashtag(word),
            '+1:word.at=%s' % hasAt(word)
        ])

    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


print(sent2features(train_sents[0])[0])


X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
#y_test = [sent2labels(s) for s in test_sents]

words = [sent2tokens(s) for s in test_sents]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

# labels = list(crf.classes_)
# labels.remove('GEN')
# labels

y_pred = crf.predict(X_test)
# print(words)
# print(y_pred)

for i in range(0,len(words[0])):
    print(words[0][i],"---",y_pred[0][i])
#print(y_test)

# metrics.flat_f1_score(y_test, y_pred,
#                       average='weighted', labels=labels)