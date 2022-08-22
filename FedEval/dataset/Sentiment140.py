import os
import numpy as np
from .FedDataBase import FedData
from nltk.stem.porter import *


def normalize_text(text):
    '''
        Final cleanup of text by removing non-alpha characters like '\n', '\t'... and
        non-latin characters + stripping.


        inputs:
            - text (str):    tweet to be processed

        return:
            - text (str):    preprocessed tweet
    '''

    # constants needed for normalize text functions
    non_alphas = re.compile(u'[^A-Za-z<>]+')
    cont_patterns = [
        ('(W|w)on\'t', 'will not'),
        ('(C|c)an\'t', 'can not'),
        ('(I|i)\'m', 'i am'),
        ('(A|a)in\'t', 'is not'),
        ('(\w+)\'ll', '\g<1> will'),
        ('(\w+)n\'t', '\g<1> not'),
        ('(\w+)\'ve', '\g<1> have'),
        ('(\w+)\'s', '\g<1> is'),
        ('(\w+)\'re', '\g<1> are'),
        ('(\w+)\'d', '\g<1> would'),
    ]
    patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]

    clean = text.lower()
    clean = clean.replace('\n', ' ')
    clean = clean.replace('\t', ' ')
    clean = clean.replace('\b', ' ')
    clean = clean.replace('\r', ' ')
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    return u' '.join([y for y in non_alphas.sub(' ', clean).strip().split(' ')])


def hashtags_preprocess(x):
    '''
        Creating a hashtag token and processing the formatting of hastags, i.e. separate uppercase words
        if possible, all letters to lowercase.


        inputs:
            - x (regex group):        x.group(1) contains the text associated with a hashtag

        return:
            - text (str):             preprocessed text
    '''
    s = x.group(1)
    if s.upper()==s:
        # if all text is uppercase, then tag it with <allcaps>
        return ' <hashtag> '+ s.lower() +' <allcaps> '
    else:
        # else attempts to split words if uppercase starting words (ThisIsMyDay -> 'this is my day')
        return ' <hashtag> ' + ' '.join(re.findall('[A-Z]*[^A-Z]*', s)[:-1]).lower()


def allcaps_preprocess(x):
    '''
        If text/word written in uppercase, change to lowercase and tag with <allcaps>.


        inputs:
            - x (regex group):        x.group() contains the text

        return:
            - text (str):             preprocessed text
    '''
    return x.group().lower()+' <allcaps> '


def glove_preprocess(text):
    '''
        To be consistent with use of GloVe vectors, we replicate most of their preprocessing.
        Therefore the word distribution should be close to the one used to train the embeddings.
        Adapted from https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb


        inputs:
            - text (str):    tweet to be processed

        return:
            - text (str):    preprocessed tweet
    '''
    # for tagging urls
    text = re.sub('(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/|www\.){1}[A-Za-z0-9.\/\\]+[]*', ' <url> ', text)
    # for tagging users
    text = re.sub("\[\[User(.*)\|", ' <user> ', text)
    text = re.sub('@[^\s]+', ' <user> ', text)
    # for tagging numbers
    text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ", text)
    # for tagging emojis
    eyes = "[8:=;]"
    nose = "['`\-]?"
    text = re.sub("<3", ' <heart> ', text)
    text = re.sub(eyes + nose + "[Dd)]", ' <smile> ', text)
    text = re.sub("[(d]" + nose + eyes, ' <smile> ', text)
    text = re.sub(eyes + nose + "p", ' <lolface> ', text)
    text = re.sub(eyes + nose + "\(", ' <sadface> ', text)
    text = re.sub("\)" + nose + eyes, ' <sadface> ', text)
    text = re.sub(eyes + nose + "[/|l*]", ' <neutralface> ', text)
    # split / from words
    text = re.sub("/", " / ", text)
    # remove punctuation
    text = re.sub('[.?!:;,()*]+', ' ', text)
    # tag and process hashtags
    text = re.sub(r'#([^\s]+)', hashtags_preprocess, text)
    # for tagging allcaps words
    text = re.sub("([^a-z0-9()<>' `\-]){2,}", allcaps_preprocess, text)
    # find elongations in words ('hellooooo' -> 'hello <elong>')
    pattern = re.compile(r"(.)\1{2,}")
    text = pattern.sub(r"\1" + " <elong> ", text)
    return text


def tweet2Vec(tweet, word2vectors):
    '''
        Takes in a processed tweet, tokenizes it, converts to GloVe embeddings
        (or zeroes if words are unknown) and applies average pool to obtain one vector for that tweet.


        inputs:
            - tweet (str):             one raw tweet from the dataset
            - word2vectors (dict):     GloVe words mapped to GloVe vectors

        return:
            - embeddings (np.array):   resulting sentence vector (shape: (200,))
    '''
    stemmer = PorterStemmer()
    return np.mean([word2vectors.get(stemmer.stem(t), np.zeros(shape=(200,))) for t in tweet.split(" ")], 0)


class sentiment140(FedData):

    def load_data(self):
        with open(os.path.join(self.data_dir, 'sent140', 'training.1600000.processed.noemoticon.csv'),
                  'r', encoding='latin-1') as f:
            train_data = f.readlines()
        with open(os.path.join(self.data_dir, 'sent140', 'testdata.manual.2009.06.14.csv'),
                  'r', encoding='latin-1') as f:
            test_data = f.readlines()

        def process_data(data):
            # "polarity", "id", "date", "query", "user", "tweet"
            data = [e.strip('\n').strip('"').split('","') for e in data]
            # "polarity", "user", "tweet"
            data = [[int(e[0]) // 4, e[4], e[5]] for e in data if e[0] != '2']
            return data

        assert 0 < self.num_clients <= 50579, \
            f"Sent140 has maximum 50579 clients, received parameter num_clients={self.num_clients}"

        train_data = process_data(train_data)
        test_data = process_data(test_data)

        all_data = train_data + test_data
        all_users = [e[1] for e in all_data]
        user_count = {}
        for user in all_users:
            user_count[user] = user_count.get(user, 0) + 1
        user_count = {key: item for key, item in user_count.items() if item > 5}
        total_num_samples = sum([user_count[e] for e in user_count])
        selected_user_set = set(
            sorted([e for e in user_count], key=lambda e: user_count[e], reverse=True)[:self.num_clients]
        )
        all_data = [e for e in all_data if e[1] in selected_user_set]  # 100
        np.random.shuffle(all_data)
        all_data = sorted(all_data, key=lambda e: e[1])

        # set the identity
        pre_id = all_data[0][1]
        self.identity = [0]
        for i in range(len(all_data)):
            if all_data[i][1] == pre_id:
                self.identity[-1] += 1
            else:
                pre_id = all_data[i][1]
                self.identity.append(1)

        # Get the label
        y = np.array([e[0] for e in all_data], dtype=np.int64)
        y = np.expand_dims(y, axis=-1)

        # Load the glove vector
        word2vectors = {}
        with open(os.path.join(self.data_dir, 'sent140', 'glove.twitter.27B.200d.txt'), 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                word2vectors[word] = np.array(line[1:]).astype(float)

        # Get x
        stemmer = PorterStemmer()
        processed_test = [normalize_text(glove_preprocess(e[-1])) for e in all_data]
        text_vectors = []
        for text in processed_test:
            tmp_vector = [word2vectors.get(stemmer.stem(e), np.zeros(shape=(200,))) for e in text.split(' ')]
            tmp_vector = np.array(tmp_vector, dtype=np.float32)
            if len(tmp_vector) < 25:
                tmp_vector = np.concatenate((np.zeros([25 - len(tmp_vector), 200]), tmp_vector))
            elif len(tmp_vector) > 25:
                tmp_vector = tmp_vector[-25:]
            text_vectors.append(tmp_vector)
        x = np.array(text_vectors, dtype=np.float64)

        print('#' * 40)
        print(f'# Data info, total samples {total_num_samples}, selected clients {self.num_clients}, '
              f'selected samples {sum(self.identity)} Ratio {sum(self.identity) / total_num_samples}')
        print('#' * 40)

        return x, y

