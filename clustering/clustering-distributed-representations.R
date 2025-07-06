

# Requires the libraries, functions and preprocessed data from the
#   file clustering-common.R

# Requires the matrices tdm, tdidf, and tcm from the file
#   clustering-symbolic-representations.R

dimensions <- c(5, 10, 25, 50, 100, 250, 500)



##################################
#         REPRESENTATION         #
#              LSA               #
##################################

get.lsa.embeddings <- function(d) {
  lsa.model <- LatentSemanticAnalysis$new(n_topics = d)
  lsa.embeddings <- lsa.model$fit_transform(tfidf)
  return(lsa.embeddings)
}

set.seed(456)
lsa.embeddings.list <- sapply(dimensions[dimensions < 250], get.lsa.embeddings)
save(lsa.embeddings.list, file = "representations-distributed-LSA.RData") # -> 35 MB

load("representations-distributed-LSA.RData")
sapply(lsa.embeddings.list, dim)




##################################
#        REPRESENTACION          #
#       Word2Vec y GloVe         #
##################################

# Filter the terms eliminated from the dtm (min. count > 4)
tokens <- word_tokenizer(imdb)
valid.words <- colnames(dtm)
tokens.filtered <- lapply(tokens, function(x) x[x %in% valid.words])


# Word vectors for Word2Vec
get.word2vec.wordvectors <- function(d) {
  word2vec.model <- word2vec(tokens.filtered, type = "cbow", dim = d, iter = 20, threads = detectCores() - 1)
  word2vec.wv <- as.matrix(word2vec.model)
  return(word2vec.wv)
}
# Word vectors for GloVe
get.glove.wordvectors <- function(d) {
  glove.model <- GlobalVectors$new(rank = d, x_max = 10, learning_rate = 0.05)
  glove.fit <- glove.model$fit_transform(tcm, n_iter = 20, n_threads = detectCores() - 1)
  glove.wv = glove.fit + t(glove.model$components)
  return(glove.wv)
}

# Calculates an embedding matrix for all the reviews for a number of dimensions
#     d by taking the average wordvector for all the terms in every review. 
#     Accepts models Word2Vec and GloVe.
get.embeddings <- function(d, model) {
  wv <- if (model == "Word2Vec") { get.word2vec.wordvectors(d) }
  else if (model == "GloVe") { get.glove.wordvectors(d) }
  
  valid.words <- rownames(wv)[rownames(wv) %in% colnames(dtm)]
  embeddings <- as.matrix(dtm[, valid.words] %*% wv[valid.words, ])
  return(sweep(embeddings, 1, rowSums(dtm), FUN = "/"))
}

set.seed(456)

# Word2Vec
word2vec.embeddings.list <- sapply(dimensions, function(dim) get.embeddings(dim, "Word2Vec"))
save(word2vec.embeddings.list, file = "representations-distributed-Word2Vec.RData") # -> 162 MB


# GloVe
glove.embeddings.list <- sapply(dimensions, function(dim) get.embeddings(dim, "GloVe"))
save(glove.embeddings.list, file = "representations-distributed-GloVe.RData") # -> 173 MB


load("representations-distributed-Word2Vec.RData")
load("representations-distributed-GloVe.RData")
sapply(word2vec.embeddings.list, dim)
sapply(glove.embeddings.list, dim)





##################################
#       PERFORMANCE TABLE        #
##################################

representation <- c("LSA", "Word2Vec", "GloVe")
performance <- expand.grid(
  dimensions = dimensions,
  representation = representation,
  stringsAsFactors = TRUE
)
performance <- performance[
  -which(performance$representation == "LSA" & performance$dimensions >= 250),
]
performance$dimensions <- as.numeric(performance$dimensions)
performance$dunn <- NA
performance$silhouette <- NA
performance$accuracy <- NA
dim(performance) # -> 19, 5







#####################################
#      CLUSTERING WITH K-MEANS      #
#####################################

# Clustering parameters
clustering.seed <- 456
clustering.nstart <- 10
clustering.max.iter <- 100
clustering.validation.sample.size <- 2000
clustering.validation.sample.amount <- 10


# New function clustering.get.data
clustering.get.data <- function(performance.index) {
  # Returns the feature matrix for LSA, Word2Vec and GloVe models depending on
  #     the representation and number of dimensions specified in the performance
  #     table by performance.index
  representation <- performance$representation[performance.index]
  dim <- performance$dimensions[performance.index]
  
  list.index <- which(sapply(glove.embeddings.list, ncol) == dim)
  
  x <- if (representation == "LSA") {lsa.embeddings.list[[list.index]]}
  else if (representation == "Word2Vec") {word2vec.embeddings.list[[list.index]]}
  else if (representation == "GloVe") {glove.embeddings.list[[list.index]]}
  
  return(x)
}

performance.indexes <- 1:nrow(performance)


cl <- makeCluster(8)
clusterExport(cl, varlist = c(
  "clustering.with.kmeans", "clustering.validation", "clustering.get.data", "clustering.log.results",
  "performance", "lsa.embeddings.list", "word2vec.embeddings.list", "glove.embeddings.list",
  "clustering.nstart", "clustering.seed", "clustering.max.iter",
  "clustering.validation.sample.size", "clustering.validation.sample.amount", "imdb.true"
))
clusterEvalQ(cl, library(Matrix))
clusterEvalQ(cl, library(yardstick))
clusterEvalQ(cl, library(proxy))
clusterEvalQ(cl, library(fpc))

# Launch clustering
file.create("clustering-log.txt")
clustering.results <- parLapply(cl, performance.indexes, function(idx) clustering.with.kmeans(idx, FALSE, TRUE))
save(clustering.results, file = "representations-distributed-cluster.RData")

load("representations-distributed-cluster.RData")
sapply(clustering.results, function(res) table(res$result$cluster))


# Launch validation
file.create("clustering-validation-log.txt")
clustering.validation.results <- parLapply(cl, clustering.results, function(res) clustering.validation(res, FALSE, TRUE))

stopCluster(cl)


# Save results
for (res in clustering.validation.results) {
  performance$dunn[res$index] <- mean(res$dunn)
  performance$silhouette[res$index] <- mean(res$silhouette)
  performance$accuracy[res$index] <- res$accuracy
}
save(performance, clustering.validation.results, file = "representations-distributed-performance.RData")

load("representations-distributed-performance.RData")








#####################################
#           RESULT GRAPHS           #
#####################################

performance.long <- pivot_longer(
  performance,
  cols = c(silhouette, dunn, accuracy),
  names_to = "metric",
  values_to = "score"
)

ggplot(performance.long, aes(x = dimensions, y = score, group = representation)) +
  geom_line(aes(color = representation), linewidth = 1.2) +
  geom_point(aes(color = representation, shape = representation), size = 3) +
  scale_color_manual(name = "Legend",
    values = c("LSA" = "#E69F00", "Word2Vec" = "#0072B2", "GloVe" = "#009E73")
  ) + 
  scale_shape_manual(name = "Legend",
    values = c("LSA" = 16, "Word2Vec" = 17, "GloVe" = 4)
  ) +
  scale_x_log10(breaks = unique(performance.long$dimensions)) +
  facet_wrap(
    ~ metric, scales = "free_y", 
    labeller = labeller(metric = c(
      silhouette = "Average silhouette index",
      dunn = "Dunn index",
      accuracy = "Accuracy"
    ))
  ) +
  labs(
    title = "Clustering performance for distributed representations",
    x = "Number of dimensions (log scale)", y = "Evaluation metric value"
  ) +
  theme_bw(base_size = 16) +
  theme(legend.position = "bottom")




