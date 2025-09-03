import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from nltk.stem import SnowballStemmer
snowballstemmer = SnowballStemmer('english')


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


stemmer= PorterStemmer()


paragraph=[
    "## Speech Of DR APJ Abdul Kalam\n",
    "paragraph = \"\"\"I have three visions for India. In 3000 years of our history, people from all over \n",
    "               the world have come and invaded us, captured our lands, conquered our minds. \n",
    "               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,\n",
    "               the French, the Dutch, all of them came and looted us, took over what was ours. \n",
    "               Yet we have not done this to any other nation. We have not conquered anyone. \n",
    "               We have not grabbed their land, their culture, \n",
    "               their history and tried to enforce our way of life on them. \n",
    "               Why? Because we respect the freedom of others.That is why my \n",
    "               first vision is that of freedom. I believe that India got its first vision of \n",
    "               this in 1857, when we started the War of Independence. It is this freedom that\n",
    "               we must protect and nurture and build on. If we are not free, no one will respect us.\n",
    "               My second vision for India’s development. For fifty years we have been a developing nation.\n",
    "               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world\n",
    "               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.\n",
    "               Our achievements are being globally recognised today. Yet we lack the self-confidence to\n",
    "               see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?\n",
    "               I have a third vision. India must stand up to the world. Because I believe that unless India \n",
    "               stands up to the world, no one will respect us. Only strength respects strength. We must be \n",
    "               strong not only as a military power but also as an economic power. Both must go hand-in-hand. \n",
    "               My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of \n",
    "               space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material.\n",
    "               I was lucky to have worked with all three of them closely and consider this the great opportunity of my life. \n",
    "               I see four milestones in my career\"\"\""
]

sentences = nltk.sent_tokenize(' '.join(paragraph))


## Apply stopwords and filter and then apply stemming

# for i in range(len(sentences)):
#     words = nltk.word_tokenize(sentences[i])
#     words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('ENGLISH'))]
#     sentences[i] =' '.join(words) #Converting all the list of words into sentences

# print(sentences)
# print(stopwords.words('ENGLISH'))

# for i in range(len(sentences)):
#     words = nltk.word_tokenize(sentences[i])
#     words = [snowballstemmer.stem(word) for word in words if word not in set(stopwords.words('ENGLISH'))]
#     sentences[i] =' '.join(words) #Converting all the list of words into sentences

# print(sentences)

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word, pos='v') for word in words if word not in set(stopwords.words('ENGLISH'))]
    sentences[i] =' '.join(words) #Converting all the list of words into sentences

print(sentences)



