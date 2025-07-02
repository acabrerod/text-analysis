# Packages for data reading and manipulation
library(here)
library(readr)
library(dplyr)


# Packages for text preprocessing
library(tm)
library(textclean)
library(textstem)
library(stringi)
library(text2vec)


# Packages for visualization
library(wordcloud)
library(igraph)
library(ggplot2)

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

files.train <- "data/imdb-train"
files.test <- "data/imdb-test"

imdb.train.pos <- read.imdb(files.train, TRUE)
imdb.train.neg <- read.imdb(files.train, FALSE)
imdb.test.pos <- read.imdb(files.test, TRUE)
imdb.test.neg <- read.imdb(files.test, FALSE)

imdb.corpus.pos <- Corpus(VectorSource(c(imdb.train.pos, imdb.test.pos)))
imdb.corpus.neg <- Corpus(VectorSource(c(imdb.train.neg, imdb.test.neg)))




##################################
#          PREPROCESSING         #
##################################

imdb.preprocess <- function(imbd.corpus) {
  # Preprocessing:
  # 1. Remove HTML tags
  # 2. Convert to lowercase
  # 3. Expand contractions
  # 4. Lemmatization
  # 5. Stop words removal
  imbd.corpus <- tm_map(imbd.corpus, content_transformer(function(x) gsub("<.*?>", " ", x)))
  imbd.corpus <- tm_map(imbd.corpus, content_transformer(tolower))
  imbd.corpus <- tm_map(imbd.corpus, content_transformer(function(x) replace_contraction(x)))
  imbd.corpus <- tm_map(imbd.corpus, content_transformer(function(x) lemmatize_strings(x)))
  imbd.corpus <- tm_map(imbd.corpus, removeWords, stopwords("en"))
  return(imbd.corpus)
}

imdb.corpus.pos <- imdb.preprocess(imdb.corpus.pos)
imdb.corpus.neg <- imdb.preprocess(imdb.corpus.neg)




##################################
#          WORD CLOUDS           #
##################################
scale <- c(5, 0.5)
color <- brewer.pal(8, "Dark2")
max.words <- 60


set.seed(456)
wordcloud(imdb.corpus.pos, scale = scale, max.words = max.words, random.order = FALSE, colors = color)
wordcloud(imdb.corpus.neg, scale = scale, max.words = max.words, random.order = FALSE, colors = color)




##################################
#     CO-OCCURRENCE NETWORKS     #
##################################

# TOKENIZATION
imdb.tokens.pos <- sapply(imdb.corpus.pos, as.character) %>% word_tokenizer()
imdb.tokens.neg <- sapply(imdb.corpus.neg, as.character) %>% word_tokenizer()

# CO-OCCURRENCE MATRIX
imdb.it.pos <- itoken(imdb.tokens.pos, progressbar = FALSE)
imdb.it.neg <- itoken(imdb.tokens.neg, progressbar = FALSE)

imdb.vocab.pos <- create_vocabulary(imdb.it.pos)
imdb.vocab.neg <- create_vocabulary(imdb.it.neg)

imdb.tcm.pos <- create_tcm(imdb.it.pos, 
  vectorizer = vocab_vectorizer(imdb.vocab.pos[order(-imdb.vocab.pos$term_count), ][1:15, ]),
  skip_grams_window = 5
)
imdb.tcm.neg <- create_tcm(imdb.it.neg, 
  vectorizer = vocab_vectorizer(imdb.vocab.neg[order(-imdb.vocab.neg$term_count), ][1:15, ]),
  skip_grams_window = 5
)


# CO-OCCURRENCE GRAPH
imdb.cooc.plot <- function(tcm, vocab,
    weight.cutoff = 800,
    seed = 45,
    layout = layout_with_fr
  ) {
  g <- graph_from_adjacency_matrix(as.matrix(tcm), mode = "undirected", weighted = TRUE, diag = FALSE)
  g <- delete_edges(g, E(g)[weight < weight.cutoff])
  set.seed(seed)
  plot(g,
       vertex.label.cex = 1.2,
       vertex.label.family = "sans",
       vertex.label.color = "black",
       vertex.size = vocab$term_count[match(V(g)$name, vocab$term)] %>% sqrt() / 10,
       edge.width = E(g)$weight / 800,
       layout = layout_with_fr 
  )
}

# Other layouts: layout_in_circle, layout_with_fr, layout_nicely
imdb.cooc.plot(imdb.tcm.pos, imdb.vocab.pos)
imdb.cooc.plot(imdb.tcm.neg, imdb.vocab.neg)





##################################
#          REVIEW LENGTH         #
##################################

review.lengths <- sapply(
  strsplit(c(imdb.train.pos, imdb.train.neg, imdb.test.pos, imdb.test.neg), "\\s+"),
  length
)

summary(review.lengths)
boxplot(review.lengths, horizontal = TRUE)

review.lengths.df <- data.frame(length = review.lengths)

# Histogram
ggplot(review.lengths.df, aes(x = length)) +
  geom_histogram(binwidth = 15, fill = "steelblue", color = "white") +
  scale_x_continuous(limits = c(0, 1000)) +
  labs(
    title = "Distribution of IMDb Review Length in Number of Words", 
    x = "Number of words",
    y = "Frequency"
  ) +
  theme_bw(base_size = 16)



