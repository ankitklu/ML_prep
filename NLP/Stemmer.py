from nltk.stem import PorterStemmer
from nltk.stem import RegexpStemmer
from nltk.stem import SnowballStemmer

stemming= PorterStemmer()



words= ['running', 'ran', 'easily', 'fairness', 'fairly', 'faster', 'fastest', 'fasterer']

for word in words:
    print(word+'----->'+stemming.stem(word))

print(stemming.stem('congratulations'))
## this will output 'congratul' which is not a valid word.
## this is the limitation of stemming, it does not give valid words.



##RegexpStemmer
## this is a class that uses regular expressions to perform stemming.
reg_stemmer= RegexpStemmer('ing|s$|e$|able$', min=4)
words= ['running', 'ran', 'easily', 'fairness', 'fairly', 'faster', 'fastest', 'fasterer']
print(reg_stemmer.stem('eating'))
print(reg_stemmer.stem('ran'))


#### SnowballStemmer
## this is a class that uses the snowball algorithm to perform stemming.
snowball_stemmer= SnowballStemmer('english')

for word in words:
    print(word+'----->'+snowball_stemmer.stem(word))

print(snowball_stemmer.stem('sportingly'))
print(snowball_stemmer.stem('fairly'))
