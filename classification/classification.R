# Packages for data reading, manipulation and evaluation
library(here)
library(readr)
library(dplyr)
library(tidyr)
library(ROCR)
library(yardstick)		
library(parallel)
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
library(scales)


# Package for classification
library(LiblineaR)





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

files.train <- "datos/imdb-train"
files.test <- "datos/imdb-test"

imdb.train.pos <- read.imdb(files.train, TRUE)
imdb.train.neg <- read.imdb(files.train, TRUE)
imdb.test.pos <- read.imdb(files.test, FALSE)
imdb.test.neg <- read.imdb(files.test, FALSE)

imdb.train <- c(imdb.train.pos, imdb.train.neg)
imdb.test <- c(imdb.test.pos, imdb.test.neg)
imdb.true <- c(
  rep("positive", length(imdb.test.pos)), 
  rep("negative", length(imdb.test.neg))
) %>% as.factor()





##################################
#        PREPROCESAMIENTO        #
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

imdb.train <- imdb.preprocess(imdb.train)
imdb.test <- imdb.preprocess(imdb.test)

save(imdb.train, imdb.test, file = "preprocessed-imdb.RData")

load("preprocessed-imdb.RData")





##################################
#         REPRESENTATION         #
#         BoW and TF-IDF         #
##################################

# Train subset
tokens.train <- word_tokenizer(imdb.train)
it.train <- itoken(tokens.train, progressbar = FALSE)
vocab <- create_vocabulary(it.train)
vocab <- prune_vocabulary(vocab, term_count_min = 4) # -> 25,788 terms
vectorizer <- vocab_vectorizer(vocab)

tcm.train <- create_tcm(it.train, vectorizer, skip_grams_window = 5)
dtm.train <- create_dtm(it.train, vectorizer)
tfidf.model <- TfIdf$new()
tfidf.train <- tfidf.model$fit_transform(dtm.train)


# Test subset
tokens.test <- word_tokenizer(imdb.test)
it.test <- itoken(tokens.test, progressbar = FALSE)

dtm.test <- create_dtm(it.test, vectorizer)
tfidf.test <- tfidf.model$transform(dtm.test)



save(
  dtm.train, dtm.test, tcm.train,  
  tfidf.train, tfidf.test,
  file = "representations-symbolic.RData"
) # -> 84 MB

load("representations-symbolic.RData")





#################################
#       FEATURE EXTRACTION      #
#         CHI-2 STATISTIC       #
#################################

calculate.chi2.for.term <- function(bow.term.column) {
  # Calculates the chi-2 statistic to test for lack of independence
  # between a given term and a given class
  class.term.contigency.table <- table(bow.term.column > 0, imdb.true)
  
  # Check that the term appears in more than one class
  if(!all(dim(class.term.contigency.table) == c(2, 2))) return(0)
  return(chisq.test(class.term.contigency.table)$statistic)
}

# For binary classification, only one statistic per term is required
feature.extraction.chi2 <- apply(dtm.train, 2, calculate.chi2.for.term)
feature.extraction.chi2 <- feature.extraction.chi2[order(feature.extraction.chi2, decreasing = TRUE)]


save(
  feature.extraction.chi2,
  file = "representations-symbolic-chi2.RData"
)

load("representations-symbolic-chi2.RData")





##################################
#   DISTRIBUTED REPRESENTATIONS  #
##################################

dimensions <- c(5, 10, 25, 50, 100, 250, 500)



##################################
#         REPRESENTATION         #
#              LSA               #
##################################

get.lsa.embeddings <- function(d) {
  lsa.model <- LatentSemanticAnalysis$new(n_topics = d)
  lsa.embeddings.train <- lsa.model$fit_transform(tfidf.train)
  lsa.embeddings.test <- lsa.model$transform(tfidf.test)
  return(list(train = lsa.embeddings.train, test = lsa.embeddings.test))
}

set.seed(456)
lsa.embeddings.list <- lapply(dimensions[dimensions < 250], get.lsa.embeddings)
save(lsa.embeddings.list, file = "representations-distributed-LSA.RData") # -> 71 MB

load("representations-distributed-LSA.RData")






##################################
#        REPRESENTATION          #
#       Word2Vec y GloVe         #
##################################

# Filter the terms eliminated from the dtm.train (min. count > 4)
valid.words <- colnames(dtm.train)
tokens.train.filtered <- lapply(tokens.train, function(x) x[x %in% valid.words])


# Word vectors for Word2Vec
get.word2vec.wordvectors <- function(d) {
  word2vec.model <- word2vec(
    tokens.train.filtered, type = "cbow", dim = d,
    iter = 20, threads = detectCores() - 1
  )
  word2vec.wv <- as.matrix(word2vec.model)
  return(word2vec.wv)
}
# Word vectors for GloVe
get.glove.wordvectors <- function(d) {
  glove.model <- GlobalVectors$new(rank = d, x_max = 10, learning_rate = 0.05)
  glove.fit <- glove.model$fit_transform(tcm.train, n_iter = 20, n_threads = detectCores() - 1)
  glove.wv = glove.fit + t(glove.model$components)
  return(glove.wv)
}

# Calculates an embedding matrix for all the reviews for a number of dimensions
#     d by taking the average wordvector for all the terms in every review. 
#     Accepts models Word2Vec and GloVe.
get.embeddings <- function(d, model) {
  wv <- if (model == "Word2Vec") { get.word2vec.wordvectors(d) }
  else if (model == "GloVe") { get.glove.wordvectors(d) }
  
  # Matrices dtm.train and dtm.test have the same columns
  valid.words <- rownames(wv)[rownames(wv) %in% colnames(dtm.train)]
  
  embeddings.train.sum <- as.matrix(dtm.train[, valid.words] %*% wv[valid.words, ])
  embeddings.test.sum <- as.matrix(dtm.test[, valid.words] %*% wv[valid.words, ])
  
  return(list(
    train = sweep(embeddings.train.sum, 1, rowSums(dtm.train), FUN = "/"), 
    test = sweep(embeddings.test.sum, 1, rowSums(dtm.test), FUN = "/")
  ))
}

set.seed(456)

# Word2Vec
word2vec.embeddings.list <- lapply(dimensions, function(dim) get.embeddings(dim, "Word2Vec"))
save(word2vec.embeddings.list, file = "representations-distributed-Word2Vec.RData") # -> 325 MB


# GloVe
glove.embeddings.list <- lapply(dimensions, function(dim) get.embeddings(dim, "GloVe"))
save(glove.embeddings.list, file = "representations-distributed-GloVe.RData") # -> 347 MB


load("representations-distributed-Word2Vec.RData")
load("representations-distributed-GloVe.RData")









##################################
#       PERFORMANCE TABLE        #
##################################

classification.algorithm <- c("SVM", "Logistic")
feature.extraction.amount <- c(10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)

performance <- rbind(
  expand.grid(
    classification.algorithm = classification.algorithm,
    representation = c("BoW", "TF-IDF"),
    dimensions = feature.extraction.amount,
    stringsAsFactors = FALSE
  ),
  expand.grid(
    classification.algorithm = classification.algorithm,
    representation = c("LSA", "Word2Vec", "GloVe"),
    dimensions = dimensions,
    stringsAsFactors = FALSE
  )
)
performance <- performance[
  -which(performance$representation == "LSA" & performance$dimensions >= 250),
]
performance$dimensions <- as.numeric(performance$dimensions)
performance$accuracy <- NA
performance$f1.score <- NA

dim(performance) # -> 78, 5







#####################################
#          CLASSIFICATION           #
#####################################

# Classification parameters
classification.seed <- 456
classification.kfold.k <- 10


classification.get.data <- function(performance.index, train = TRUE) {
  # Returns the feature matrix for all representation models depending on
  #     the number of dimensions specified in the performance table by 
  #     performance.index
  representation <- as.character(performance$representation[performance.index])
  d <- performance$dimensions[performance.index]
  
  # Feature extraction
  if(representation %in% c("BoW", "TF-IDF")) {
    extracted.features <- names(feature.extraction.chi2)[1:d]
  } else { 
    list.index <- which(sapply(glove.embeddings.list, function(l) ncol(l$train)) == d)
  }
  
  # Feature matrix
  if(train) {
    x.train <- if (representation == "LSA") {lsa.embeddings.list[[list.index]]$train}
    else if (representation == "Word2Vec") {word2vec.embeddings.list[[list.index]]$train}
    else if (representation == "GloVe") {glove.embeddings.list[[list.index]]$train}
    else if (representation == "BoW") {dtm.train[, extracted.features]} 
    else if (representation == "TF-IDF") {tfidf.train[, extracted.features]}
    else { stop("Error: non-valid representation method") }
    return(x.train)
  }
  x.test <- if (representation == "LSA") {lsa.embeddings.list[[list.index]]$test}
  else if (representation == "Word2Vec") {word2vec.embeddings.list[[list.index]]$test}
  else if (representation == "GloVe") {glove.embeddings.list[[list.index]]$test}
  else if (representation == "BoW") {dtm.test[, extracted.features]} 
  else if (representation == "TF-IDF") {tfidf.test[, extracted.features]}
  else { stop("Error: non-valid representation method") }
  return(x.test)
  
}

# Entrena y evalua al clasificador
train.and.eval.classifier <- function(performance.index) {
  # Trains the classifier specified in the performance table by 
  #       performance.index, and returns the evaluation results with the
  #       metric accuracy and F1-Score. If executed in the same thread, prints
  #       progress in the console.
  classification.algorithm <- as.character(performance$classification.algorithm[performance.index])
  train.data <- classification.get.data(performance.index, train = TRUE)
  test.data <- classification.get.data(performance.index, train = FALSE)
  
  # Train classifier
  set.seed(classification.seed)
  classifier.type <- c("Logistic" = 0, "SVM" = 2)[classification.algorithm]
  best.cost <- LiblineaR(
    as.matrix(train.data), imdb.true, type = classifier.type, 
    cost = 1, cross = classification.kfold.k, findC = TRUE
  )
  classifier <- LiblineaR(
    as.matrix(train.data), imdb.true, 
    type = classifier.type, cost = best.cost
  )
  
  # Evaluation
  pred.train <- predict(classifier, train.data)
  pred.test <- predict(classifier, test.data)
  accuracy.test <- accuracy_vec(imdb.true, pred.test[[1]])
  accuracy.train <- accuracy_vec(imdb.true, pred.train[[1]])
  f1.score.test <- f_meas_vec(imdb.true, pred.test[[1]])
  f1.score.train <- f_meas_vec(imdb.true, pred.train[[1]])
  
  # Print results on console
  representation <- as.character(performance$representation[performance.index])
  dimensions <- performance$dimensions[performance.index]
  cat(
    "** Classifier **\n",
    "Algorithm: ", classification.algorithm, "\n",
    "Representation: ", representation, " (", dimensions, ")", "\n",
    "Accuracy (train): ", round(accuracy.test, 3), " (", round(accuracy.train, 3), ")", "\n",
    "F1-Score (train): ", round(f1.score.test, 3), " (", round(f1.score.train, 3), ")", "\n",
    "\n\n", sep = ""
  )
  return(list(
    index = performance.index,
    classifier = classifier,
    accuracy = accuracy.test,
    f1.score = f1.score.test
  ))
}


# Optional filters to exclude tests from execution
performance.indices <- which(
  # performance$representation %in% c("BoW", "TF-IDF") & 
  # performance$representation %in% c("LSA", "Word2Vec", "GloVe")
  performance$dimensions > 0
)

# For same-thread execution:
# classification.results <- sapply(performance.indices, train.and.eval.classifier)

# In parallel
cl <- makeCluster(3)
clusterExport(cl, varlist = c(
  "classification.get.data",
  "performance", "feature.extraction.chi2",
  "dtm.train", "dtm.test", "tfidf.train", "tfidf.test",
  "lsa.embeddings.list", "word2vec.embeddings.list", "glove.embeddings.list",
  "classification.seed", "classification.kfold.k", "imdb.true"
))
clusterEvalQ(cl, library(Matrix))
clusterEvalQ(cl, library(yardstick))
clusterEvalQ(cl, library(LiblineaR))

# Execute classification
classification.results <- parLapply(cl, performance.indices, train.and.eval.classifier)
save(classification.results, file = "classification-results.RData")
load("classification-results.RData")

stopCluster(cl)


# Save results
for (res in classification.results) {
  performance$accuracy[res$index] <- res$accuracy
  performance$f1.score[res$index] <- res$f1.score
}
save(performance, file = "performance.RData")

load("performance.RData")





#####################################
#           RESULT GRAPHS           #
#####################################

performance.long <- pivot_longer(
  performance,
  cols = c(accuracy, f1.score),
  names_to = "metric",
  values_to = "score"
)


graph.params.list <- list(
  list(
    representations.filter = c("BoW", "TF-IDF"),
    title = "Classification performance for symbolic representations",
    x.label = "Number of extracted features (log scale)",
    y.limits = c(0.7, 0.88),
    manual.color.values = c("TF-IDF" = "#0072B2", "BoW" = "#E69F00")
  ),
  list(
    representations.filter = c("LSA", "Word2Vec", "GloVe"),
    title = "Classification performance for distributed representations",
    x.label = "Number of dimensions (log scale)",
    y.limits = c(0.65, 0.87),
    manual.color.values = c("LSA" = "#E69F00", "Word2Vec" = "#0072B2", "GloVe" = "#009E73")
  )
)
for (graph.params in graph.params.list) {
  p <- performance.long %>%
    filter(representation %in% graph.params$representations.filter) %>% 
    mutate(group = paste(representation, classification.algorithm, sep = "_")) %>%
    ggplot(aes(
      x = dimensions, y = score, 
      color = representation, group = group
    )) +
    geom_line(aes(color = representation), linewidth = 1.2) +
    geom_point(aes(shape = classification.algorithm), size = 3) +
    scale_color_manual(
      name = "Representation",
      values = graph.params$manual.color.values
    ) + 
    scale_shape_manual(
      name = "Algorithm",
      values = c("SVM" = 16, "Logistic" = 17),
    ) +
    scale_x_log10() +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = graph.params$y.limits
    ) +
    facet_wrap(
      ~ metric, scales = "free_y", 
      labeller = labeller(metric = c(
        accuracy = "Accuracy", f1.score = "F1-Score"
      ))
    ) +
    labs(
      title = graph.params$title,
      x = graph.params$x.label,
      y = "Evaluation metric value"
    ) +
    theme_bw(base_size = 16) +
    theme(legend.position = "bottom")
  
  print(p)
}




#####################################
#    EXTRACTION OF BEST RESULTS     #
#     for any chosen crierion       #
#####################################

performance %>%
  #filter(representation == "GloVe") %>% 
  filter(dimensions == 100)  %>% 
  mutate(
    accuracy = round(accuracy, 3), 
    f1.score = round(f1.score, 3),
    rank.acc = dense_rank(desc(accuracy)),
    rank.f1 = dense_rank(desc(f1.score)),
    best.acc = case_when(rank.acc == 1 ~ "***", rank.acc == 2 ~ "+++", TRUE ~ ""),
    best.f1 = case_when(rank.f1 == 1 ~ "***", rank.f1 == 2 ~ "+++", TRUE ~ "")
  ) %>%
  select(-rank.acc, -rank.f1)
  





