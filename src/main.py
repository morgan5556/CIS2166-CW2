# Imports the relevant libraries for the program to run
import operator
import pandas as pd
from textblob import TextBlob
from nltk import word_tokenize
from nltk import pos_tag
from nltk import bigrams

# Sets the global variables used within the program
positive_count = 0
negative_count = 0
bigrams_with_frequencies = {}

# This is the main function, this is what is first executed
# when the program is run
def main():
    # Data is loaded from the load data function, variables
    # are defined and columns are added to the data frame
    review_texts = load_data()
    review_texts.columns = ["Index", "Movie ID", "Review"]
    sentiment_list = []
    sentiment_bool_list = []

    # Indicates to use the global variables declared
    global positive_count, negative_count
    
    # Iterates through each of the reviews and performs
    # sentiment analysis
    for index, row in review_texts.iterrows():
        sentiment = sentiment_analysis(str(row['Review']))
        sentiment_list.append(round(sentiment, 2))

        print("Index: " + str(index) + " Sentiment: " + str(round(sentiment, 2)))
        # Dependent on the sentiment, attaches a boolean to 
        # the list which will be added to the dataframe
        if sentiment > 0:
            sentiment_bool_list.append(True)
        else:
            sentiment_bool_list.append(False)
        
    # Adds the sentiment value, and the boolean to the dataframe
    # attached to the review
    review_texts.insert(3, "Sentiment", sentiment_list, True)
    review_texts.insert(4, "Sentiment Boolean", sentiment_bool_list, True)

    # Outputs the total number of positive reviews, and total number
    # of negative reviews
    print("\nSENTIMENT ANALYSIS: \n ----------")
    print("Positive: " + str(len(review_texts[review_texts['Sentiment'] > 0])))
    print("Negative: " + str(len(review_texts[review_texts['Sentiment'] <= 0])))

    # Creates two dataframes from the initial, one for positives
    # and one for negatives
    positive_reviews = review_texts[review_texts['Sentiment'] > 0]
    negative_reviews = review_texts[review_texts['Sentiment'] <= 0]

    # Runs the collocation extraction methods, and passess booleans
    # pased on whether or not to POS filter
    print("\nPOSITIVE COLLOCATIONS WITHOUT POS: \n ----------")
    collocation_extraction(positive_reviews, False)
    print("\nNEGATIVE COLLOCATIONS WITHOUT POS: \n ----------")
    collocation_extraction(negative_reviews, False)
    print("\nPOSITIVE COLLOCATIONS WITH POS: \n ----------")
    collocation_extraction(positive_reviews, True)
    print("\nNEGATIVE COLLOCATIONS WITH POS: \n ----------")
    collocation_extraction(negative_reviews, True)

def load_data():
    # Loads and returns the movie reviews
    review_df = pd.read_csv("data/movie_reviews.csv", sep='\t', header=None, engine='python', quoting=3)
    return review_df

def sentiment_analysis(review):
    sentiment_score = 0
    blob = TextBlob(review)

    # Goes through each sentence in the review, and adds to the sentiment
    # currently stored
    for sentence in blob.sentences:
        sentiment_score = sentiment_score + sentence.sentiment.polarity
 
    return sentiment_score 

def collocation_extraction(reviews, pos_bool):
    # Goes through each of the reviews, and calculate
    # the bigram
    for index, row in reviews.iterrows():
        pos_tagging_bigram(row['Review'], pos_bool)
    
    # Output the frequencies
    output_bigram_freq()

def pos_tagging_bigram(review, pos_bool):
    # This is the POS tag list, for each of the POS tags we want
    pos_tag_list = ["NN", "JJ", "NNP", "VB"]

    # A try statement is used to ensure any errors are passed
    try:
        words = word_tokenize(review)
        words_tags = pos_tag(words)

        bigram_tag = list(bigrams(words_tags))

        for first, second in bigram_tag:
            first_word, first_pos = first[0], first[1]
            second_word, second_pos = second[0], second[1]
            
            # this is done dependent on if POS filtering is used or not
            if pos_bool == True:
                if first_pos in pos_tag_list and second_pos in pos_tag_list:
                    update_freq(first_word, second_word)
            else:
                update_freq(first_word, second_word)
    except:
        pass
    
def update_freq(first_word, second_word):

    # This updates the frequency of the bigram, if it is already present
    if (first_word, second_word) in bigrams_with_frequencies:
        bigrams_with_frequencies[(first_word, second_word)] += 1

    # This adds the frequency of the bigram
    else:
        bigrams_with_frequencies[(first_word, second_word)] = 1

def output_bigram_freq():
    # Sorted the bigram into frequency order
    sorted_list = dict(sorted(bigrams_with_frequencies.items(),
                         key=operator.itemgetter(1), reverse=True))

    # Outputs the top 40 based on frequency
    count = 0
    for bigram in sorted_list:
        print(count, bigram, sorted_list[bigram])
        count = count + 1

        if count == 40:
            break

    # Clears the dictionaries
    bigrams_with_frequencies.clear()
    sorted_list.clear()

# Runs the main function
if __name__ == "__main__":
    main()