# Packages for data reading, manipulation and evaluation
library(here)
library(readr)
library(dplyr)
library(tidyr)
library(ROCR)
library(yardstick)

# Packages for text preprocessing and analysis
library(tm)         # To create and clean corpora
library(textstem)   # For lemmatization
library(textclean)  # For normalization


# Packages for Sentiment Analysis
library(syuzhet)
library(tidytext)
library(sentimentr)
library(SentimentAnalysis)

rm(list=ls())




##################################
#         DATA LOADING           #
##################################
read.imdb <- function(path, polarity = TRUE) {
  # path - string with the path to the data
  # pos - boolean indicating positive/negative
  files.list <- list.files(
    path = paste(path, "/", ifelse(polarity, "pos", "neg"), sep = "") %>% here(), 
    pattern = "\\.txt$", full.names = TRUE
  )
  files.vector <- sapply(files.list, read_file, USE.NAMES = FALSE)
  return(files.vector)
}

# Only reviews from test subset
files.test <- "data/imdb-test"

imdb.test.pos <- read.imdb(files.test, TRUE)
imdb.test.neg <- read.imdb(files.test, FALSE)

imdb <- c(imdb.test.pos, imdb.test.neg)
imdb.true <- c(
  rep("positive", length(imdb.test.pos)), 
  rep("negative", length(imdb.test.neg))
) %>% as.factor()









##################################
#          PREPROCESSING         #
##################################

imdb.preprocess <- function(imdb) {
  # Preprocessing:
  # 1. Remove HTML tags
  # 2. Convert to lowercase
  # 3. Expand contractions
  # 4. Lemmatization
  imbd.corpus <- Corpus(VectorSource(imdb))
  imbd.corpus <- tm_map(imbd.corpus, content_transformer(function(x) gsub("<.*?>", " ", x)))
  imbd.corpus <- tm_map(imbd.corpus, content_transformer(tolower))
  imbd.corpus <- tm_map(imbd.corpus, content_transformer(function(x) replace_contraction(x)))
  imbd.corpus <- tm_map(imbd.corpus, content_transformer(function(x) lemmatize_strings(x)))
  return(sapply(imbd.corpus, as.character))
}

imdb <- imdb.preprocess(imdb)



###################################
#        Sentiment Analysis       #
###################################

# For each of the packages that will be used, the following
# objects are defined:
#
# 1. {$package}.dicts - Array or list with the dictionaries that will be used 
#                         for that package.
# 2. {$package}.results - Data frame that stores the sentiment evaluations with
#                         every dictionary for that package.
# 3. {$package}.imdb (only for tidytext and sentimentr) - Additional 
#                         preprocessing for the reviews



###################################
# Sentiment Analysis with syuzhet #
###################################

syuzhet.dicts <- c("Bing", "AFINN", "NRC", "Syuzhet")
syuzhet.results <- matrix(
  0, nrow = length(imdb), ncol = length(syuzhet.dicts), 
  dimnames = list(NULL, syuzhet.dicts)
) %>% data.frame()


for(dictionary in syuzhet.dicts) {
  syuzhet.results[[dictionary]] <- get_sentiment(
    imdb, method = dictionary %>% tolower()
  )
}





####################################
# Sentiment Analysis with tidytext #
####################################

tidytext.dicts <- list(
  "Bing" = get_sentiments("bing"), 
  "AFINN" = get_sentiments("afinn"),
  "NRC" = get_sentiments("nrc") %>% 
    filter(sentiment %in% c("positive", "negative"))
)
tidytext.imdb <- tibble(
    review.number = 1:length(imdb), 
    review = imdb
  ) %>%
  unnest_tokens(word, review)
tidytext.results <- matrix(
  0, nrow = length(imdb), ncol = length(tidytext.dicts), 
  dimnames = list(NULL, names(tidytext.dicts))
) %>% data.frame()


for(dictionary in c("Bing", "NRC")) {
  tidytext.results[[dictionary]] <- tidytext.imdb %>%
    left_join(tidytext.dicts[[dictionary]], by = "word") %>%
    count(review.number, sentiment) %>%
    pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
    mutate(sentiment.score = positive - negative) %>%
    pull(sentiment.score)
}
tidytext.results$AFINN <- tidytext.imdb %>%
  left_join(tidytext.dicts[["AFINN"]], by = "word") %>%
  group_by(review.number) %>%
  summarise(sentiment.score = sum(value, na.rm = T)) %>%
  pull(sentiment.score)





######################################
# Sentiment Analysis with sentimentr #
######################################
# preprocessing: get sentences in    #
#     each review and add their      #
#     polarities                     #
######################################

sentimentr.imdb <- imdb %>% get_sentences()
sentimentr.dicts <- list(
  "Syuzhet" = lexicon::hash_sentiment_jockers,
  "Bing" = lexicon::hash_sentiment_huliu,
  "Loughran" = lexicon::hash_sentiment_loughran_mcdonald,
  "NRC" = lexicon::hash_sentiment_nrc,
  "SentiWordNet" = lexicon::hash_sentiment_sentiword,
  "Jockers-Rinker" = lexicon::hash_sentiment_jockers_rinker
)
sentimentr.results <- matrix(
  0, nrow = length(imdb), ncol = length(sentimentr.dicts), 
  dimnames = list(NULL, names(sentimentr.dicts))
) %>% data.frame()


for(dictionary in names(sentimentr.dicts)) {
  sentimentr.results[[dictionary]] <- 
    sentiment(sentimentr.imdb, sentimentr.dicts[[dictionary]]) %>%
      group_by(element_id) %>%
      summarise(sentiment = sum(sentiment)) %>%
      pull(sentiment)
}



#############################################
# Sentiment Analysis with SentimentAnalysis #
#############################################

sa.dicts <- c("GI" = 2, "Henry" = 5, "Loughran" = 8, "QDAP" = 12)


sa.results <- rbind(
  analyzeSentiment(imdb[1:6250]),
  analyzeSentiment(imdb[6251:12500]),
  analyzeSentiment(imdb[12501:18750]),
  analyzeSentiment(imdb[18751:25000])
  
)
sa.results[is.na(sa.results)] <- 0





##################################
#     PERFORMANCE EVALUATION     #
#      Accuracy and F1-Score     #
##################################

dictionaries <- c(
  "GI", "Bing", "AFINN", "NRC", "Syuzhet",
  "QDAP", "SentiWordNet", "Henry", "Loughran",
  "Jockers-Rinker"
)
packages <- c(
  "syuzhet", "tidytext",
  "sentimentr", "SentimentAnalysis"
)

# One able for each evaulation metric
f1 <- accuracy <- matrix(
  NA, 
  nrow = length(dictionaries), 
  ncol = length(packages),
  dimnames = list(dictionaries, packages)
)


for(dictionary in syuzhet.dicts) {
  pred <- ifelse(syuzhet.results[[dictionary]] > 0, "positive", "negative") %>% as.factor()
  
  accuracy[dictionary, "syuzhet"] <- accuracy_vec(imdb.true, pred)
  f1[dictionary, "syuzhet"] <- f_meas_vec(imdb.true, pred)
}
for(dictionary in names(tidytext.dicts)) {
  pred <- ifelse(tidytext.results[[dictionary]] > 0, "positive", "negative") %>% as.factor()
  
  accuracy[dictionary, "tidytext"] <- accuracy_vec(imdb.true, pred)
  f1[dictionary, "tidytext"] <- f_meas_vec(imdb.true, pred)
}
for(dictionary in names(sentimentr.dicts)) {
  pred <- ifelse(sentimentr.results[[dictionary]] > 0, "positive", "negative") %>% as.factor()
  
  accuracy[dictionary, "sentimentr"] <- accuracy_vec(imdb.true, pred)
  f1[dictionary, "sentimentr"] <- f_meas_vec(imdb.true, pred)
}
for(dictionary in names(sa.dicts)) {
  pred <- ifelse(sa.results[[sa.dicts[dictionary]]] > 0, "positive", "negative") %>% as.factor()
  
  accuracy[dictionary, "SentimentAnalysis"] <- accuracy_vec(imdb.true, pred)
  f1[dictionary, "SentimentAnalysis"] <- f_meas_vec(imdb.true, pred)
}







#####################################
#              RESULTS              #
#####################################

round(accuracy, 4)
round(f1, 4)


