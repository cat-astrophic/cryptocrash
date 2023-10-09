# This script performs sentiment analysis on the harvested cryptobro tweets

# Importing required modules

import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from matplotlib import pyplot as plt

# Where to write the sentiment analysis outputs

writeto = 'D:/cryptobros/data/'

# Read in the tweets data

td = pd.read_csv(writeto + 'data.csv')

# Extract only the relevant data

td = td[['id', 'conversation_id', 'created_at', 'date', 'place', 'tweet', 'language', 'hashtags', 'cashtags',
         'user_id', 'user_id_str', 'username', 'name', 'day', 'hour', 'link', 'urls', 'photos', 'video',
         'thumbnail', 'retweet', 'nlikes', 'nreplies', 'nretweets', 'quote_url', 'search', 'near', 'geo',
         'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to', 'retweet_date', 'period', 'treated']]

# Isolating the most frequent words by treatment status along with the frequencies of tweet appearances

# Splitting the data by treatment status

treated = td[td.treated == 'treated'].reset_index(drop = True)
control = td[td.treated == 'control'].reset_index(drop = True)

# Obtaining word frequencies for each group

# Tokenizing the twitter data and removing stopwords

stop_words = set(stopwords.words('english'))

tweets_t = [str(tweet) for tweet in treated.tweet]
tweets_c = [str(tweet) for tweet in control.tweet]

clean_tweets_t = []
clean_tweets_c = []

for tweet in tweets_t:
    
    word_tokens = word_tokenize(tweet)
    filtered_sentence = []
    
    for w in word_tokens:
        
        if w not in stop_words:
            
            filtered_sentence.append(w)
    
    clean_tweets_t.append(filtered_sentence)

for tweet in tweets_c:
    
    word_tokens = word_tokenize(tweet)
    filtered_sentence = []
    
    for w in word_tokens:
        
        if w not in stop_words:
            
            filtered_sentence.append(w)
    
    clean_tweets_c.append(filtered_sentence)

# Lemmatizing

lemon = WordNetLemmatizer()

for t in range(len(clean_tweets_t)):
    
    res = []
    
    for w in clean_tweets_t[t]:
        
        res.append(lemon.lemmatize(w))
    
    clean_tweets_t[t] = res

for t in range(len(clean_tweets_c)):
    
    res = []
    
    for w in clean_tweets_c[t]:
        
        res.append(lemon.lemmatize(w))
    
    clean_tweets_c[t] = res

# Stemming

ps = PorterStemmer()

for t in range(len(clean_tweets_t)):
    
    res = []
    
    for w in clean_tweets_t[t]:
        
        res.append(ps.stem(w))
        
    clean_tweets_t[t] = res

for t in range(len(clean_tweets_c)):
    
    res = []
    
    for w in clean_tweets_c[t]:
        
        res.append(ps.stem(w))
        
    clean_tweets_c[t] = res

# Remove usernames from @s

for t in range(len(clean_tweets_t)):
    
    ws = clean_tweets_t[t]
    uns = [ws[w] for w in range(1,len(ws)) if ws[w-1] == '@']
    clean_tweets_t[t] = [c for c in clean_tweets_t[t] if c not in uns]

for t in range(len(clean_tweets_c)):
    
    ws = clean_tweets_c[t]
    uns = [ws[w] for w in range(1,len(ws)) if ws[w-1] == '@']
    clean_tweets_c[t] = [c for c in clean_tweets_c[t] if c not in uns]

# Remove symbols

baddies = ['@', '#', '$', '%', '&', '*', ':', ';', '"', '.', ',', '/', '!',
           "'s", 'http', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

for t in range(len(clean_tweets_t)):
    
    clean_tweets_t[t] = [c for c in clean_tweets_t[t] if c not in baddies]

for t in range(len(clean_tweets_c)):
    
    clean_tweets_c[t] = [c for c in clean_tweets_c[t] if c not in baddies]

# Creating a comprehensive list of words for each group

t_bag = []
c_bag = []

for t in clean_tweets_t:
    
    try:
        
        for w in t:
            
            t_bag.append(w)
            
    except:
        
        continue

for t in clean_tweets_c:
    
    try:
        
        for w in t:
            
            c_bag.append(w)
            
    except:
        
        continue

# Get unique words for each group

t_unique = []
c_unique = []

for w in t_bag:
    
    if w not in t_unique:
        
        t_unique.append(w)

for w in c_bag:
    
    if w not in c_unique:
        
        c_unique.append(w)

# Get the frequencies

lt = len(t_bag)
lc = len(c_bag)

t_counts = [t_bag.count(w) for w in t_unique]
c_counts = [c_bag.count(w) for w in c_unique]

t_freqs = [x/lt for x in t_counts]
c_freqs = [x/lc for x in c_counts]

# Look at the distribution of word frequencies

tf_plot = list(sorted(t_freqs))
cf_plot = list(sorted(c_freqs))

plt.plot(tf_plot)
plt.plot(cf_plot)

# Find the most frequent keywords for each treatment status

tdf = pd.concat([pd.Series(t_counts, name = 'Count'), pd.Series(t_freqs, name = 'Freq'), pd.Series([i for i in range(len(t_counts))], name = 'ID')], axis = 1)
cdf = pd.concat([pd.Series(c_counts, name = 'Count'), pd.Series(c_freqs, name = 'Freq'), pd.Series([i for i in range(len(c_counts))], name = 'ID')], axis = 1)

tdf = tdf.sort_values(by = 'Freq', ascending = False)
cdf = cdf.sort_values(by = 'Freq', ascending = False)

t_ids_df = tdf[tdf.Freq > .001]
c_ids_df = cdf[cdf.Freq > .001]

t_ids = list(t_ids_df.ID)
c_ids = list(c_ids_df.ID)

t_words = [t_unique[i] for i in t_ids]
c_words = [c_unique[i] for i in c_ids]

# Which words do not intersect the other list (unique keys)

t_words_only = [w for w in t_words if w not in c_words]
c_words_only = [w for w in c_words if w not in t_words]

# Clean up the lists

t_words_only.remove('\U0001fae1')

c_words_only.remove(']')
c_words_only.remove('[')
c_words_only.remove('a')
c_words_only.remove('‘')
c_words_only.remove('|')
c_words_only.remove('in')
c_words_only.remove('to')

# Waht's left? -- 62 v 19 words with very different implications

# Create and save a dataframe with the uniquely frequent keys

results = pd.concat([pd.Series(t_words_only, name = 'Treated'), pd.Series(c_words_only, name = 'Control')], axis = 1)
results.to_csv(writeto + 'uniquely_frequent.csv', index = False)

# Creating a word cloud for each group from their uniquely frequent keys (stems are extrapolated!)

t_words_only[2] = 'solana'
t_words_only[4] = 'piece'
t_words_only[11] = 'purchase'
t_words_only[17] = 'analytical'
t_words_only[18] = 'realtime'
t_words_only[21] = 'unique'
t_words_only[35] = 'amazing'
t_words_only[38] = 'always'
t_words_only[44] = 'happy'
t_words_only[47] = 'commune'
t_words_only[52] = 'awesome'
t_words_only[53] = 'congrats'

c_words_only[2] = 'quote'
c_words_only[10] = 'business'

twords = ' '.join(t_words_only)
wordcloud = WordCloud(max_font_size = 50, max_words = 100, background_color = 'white', colormap = 'viridis').generate(twords)
plt.figure()
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()

cwords = ' '.join(c_words_only)
wordcloud = WordCloud(max_font_size = 50, max_words = 100, background_color = 'white', colormap = 'viridis').generate(cwords)
plt.figure()
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()

