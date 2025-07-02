# Packages for data reading, manipulation and evaluation
library(here)
library(readr)
library(dplyr)
library(tidyr)
library(ROCR)
library(yardstick)		
library(parallel)
library(fpc)
library(lubridate)
library(Matrix)


# Packages for text preprocessing and analysis
library(tm)         # To create and clean corpora
library(textstem)   # For lemmatization
library(textclean)  # For normalization
library(text2vec)		# For BoW, TF-IDF, LSA and GloVe representations
library(word2vec)		# For Word2Vec representation
library(proxy) 		  # For cosine distances calculation
library(stringi)
library(stringr)	


# Packages for result graphs
library(ggplot2)
library(ggrepel)
library(scales)



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
  # 3. Number elimination
  # 4. Stopword elimination
  # 5. Expand contractions
  # 6. Lemmatization
  imbd.corpus <- Corpus(VectorSource(imdb))
  imbd.corpus <- tm_map(imbd.corpus, content_transformer(function(x) gsub("<.*?>", " ", x)))
  imbd.corpus <- tm_map(imbd.corpus, content_transformer(tolower))
  imbd.corpus <- tm_map(imbd.corpus, removeNumbers)
  imbd.corpus <- tm_map(imbd.corpus, removeWords, stopwords("en"))
  imbd.corpus <- tm_map(imbd.corpus, content_transformer(function(x) replace_contraction(x)))
  imbd.corpus <- tm_map(imbd.corpus, content_transformer(function(x) lemmatize_strings(x)))
  return(sapply(imbd.corpus, as.character))
}

imdb <- imdb.preprocess(imdb)

save(imdb, file = "preprocessed-imdb.RData")
load("preprocessed-imdb.RData")





###################################
# COMMON FUNCTIONS FOR CLUSTERING #
###################################

# Prints partial results for clustering and evaluation when a test ends
clustering.log.results <- function(performance.index, start, symbolic, validation) {
  # performance.index - index in the performance table, to obtain test parameters
  # start - timestamp that indicates when the algorithm started
  # symbolic - boolean to indicate whether the representation is symbolic
  # validation - list with results if the calculation that ended was validation
  end <- Sys.time()
  elapsed <- as.numeric(difftime(end, start, units = "secs"))
  hours <- floor(elapsed / 3600)
  minutes <- floor((elapsed %% 3600) / 60)
  seconds <- round(elapsed %% 60)
  
  representation <- as.character(performance$representation[performance.index])
  log.entry <- paste(
    "Start: ", format(start, "%H:%M:%S"), " End: ", format(end, "%H:%M:%S"), "\n",
    "Elapsed: ", sprintf("%02d:%02d:%02d", hours, minutes, seconds), "\n",
    "Representation: ", representation, "\n", sep = ""
  )
  if(symbolic) {
    fe.method <- as.character(performance$feature.extraction.method[performance.index])
    fe.amount <- performance$feature.extraction.amount[performance.index]
    log.entry <- paste(log.entry, 
      "Feature extraction: ", fe.method, " (", fe.amount, ")\n", sep = ""
    )
  } else {
    dimensions <- performance$dimensions[performance.index]
    log.entry <- paste(log.entry, 
      "Dimensions: ", dimensions, "\n", sep = ""
    )
  }
  if(length(validation) == 0) {
    write(paste(log.entry, "\n\n"), file = "clustering-log.txt", append = TRUE)
    return()
  }
  for(metric in names(validation)){
    log.entry <- paste(log.entry, 
      metric, ": ", validation[[metric]], "\n", sep = ""
    )
  }
  write(paste(log.entry, "\n\n"), file = "clustering-validation-log.txt", append = TRUE)
}

# Executes k-means algorithm
clustering.with.kmeans <- function(performance.index, symbolic = TRUE, log = TRUE) {
  # Returns a list with the index for the performance table and the object
  #     returned by kmeans
  start <- Sys.time()
  x <- clustering.get.data(performance.index) # obtains data matrix according to the test parameters
  
  # Clustering with k-means
  set.seed(clustering.seed)
  clustering.result <- kmeans(
    x, centers = 2, nstart = clustering.nstart, iter.max = clustering.max.iter
  )
  
  if(log) { clustering.log.results(performance.index, start, symbolic, list()) }
  
  return(list(
    index = performance.index,
    result = clustering.result
  ))
}

# Calculates validation metrics
clustering.validation <- function(res, symbolic = TRUE, log = TRUE) {
  # Estimates internal and external validation metrics for clustering
  # - interal: Dunn and Silhouette indices (estimated)
  # - external: Accuracy (exact calculation)
  #
  # Takes samples of reviews and computes a distance matrix, from which the
  # validation metrics are calculated. The process is repeated several times.
  #
  # res is a list with the following content:
  # - index: index in the performance table, to obtain test parameters
  # - result: object returned by kmeans
  
  index <- res$index
  n <- length(res$result$cluster)
  
  # Stores results for every sample
  validation.dunn <- numeric(clustering.validation.sample.amount)
  validation.silhouette <- numeric(clustering.validation.sample.amount)
  
  start <- Sys.time()
  x <- clustering.get.data(index)
  
  set.seed(clustering.seed)
  # Internal validation
  for(sample.index in 1:clustering.validation.sample.amount) {
    validation.sample <- sample(1:n, clustering.validation.sample.size)
    validation.dist.matrix <- proxy::dist(as.matrix(x[validation.sample, ]), method = "cosine")
    
    validation.stats <- cluster.stats(
      validation.dist.matrix, 
      res$result$cluster[validation.sample],
      wgap = FALSE, sepindex = FALSE, 
    )
    validation.dunn[sample.index] <- validation.stats$dunn
    validation.silhouette[sample.index] <- validation.stats$avg.silwidth
  }
  
  # External validation
  validation.accuracy <- accuracy_vec(
    imdb.true, 
    factor(ifelse(res$result$cluster == 1, "positive", "negative"))
  )
  if (validation.accuracy < 0.5) validation.accuracy <- 1 - validation.accuracy
  
  validation <- list(
    "Dunn index" = round(median(validation.dunn), 4),
    "Silhouette index" = round(median(validation.silhouette), 4),
    "Accuracy" = validation.accuracy
  )
  if(log) { clustering.log.results(index, start, symbolic, validation) }
  
  return(list(
    index = index,
    dunn = validation.dunn,
    silhouette = validation.silhouette,
    accuracy = validation.accuracy
  ))
}

