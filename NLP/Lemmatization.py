# import nltk
# nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

### pos= part of speech
### pos= 'a' for adjective  
### pos= 'n' for noun
### pos= 'r' for adverb
### pos= 'v' for verb 

print(lemmatizer.lemmatize("going", pos='v'))
print(lemmatizer.lemmatize("better", pos='a'))
print(lemmatizer.lemmatize("better", pos='n'))
print(lemmatizer.lemmatize("better", pos='r'))

words= ['running', 'ran', 'easily', 'fairness', 'fairly', 'faster', 'fastest', 'fasterer']
for word in words:
    print(word+'----->'+lemmatizer.lemmatize(word, pos='v'))
 