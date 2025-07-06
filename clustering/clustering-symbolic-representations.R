
# Requires the libraries, functions and preprocessed data from the
#   file clustering-common.R


##################################
#         REPRESENTATION         #
#         BoW and TF-IDF         #
##################################

tokens <- word_tokenizer(imdb)
it <- itoken(tokens, progressbar = TRUE)
vocab <- create_vocabulary(it)
vocab <- prune_vocabulary(vocab, term_count_min = 4)
vectorizer <- vocab_vectorizer(vocab)
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)
dtm <- create_dtm(it, vectorizer)

tfidf.model <- TfIdf$new()
tfidf <- tfidf.model$fit_transform(dtm)


save(dtm, tfidf, file = "representations-symbolic-Bow-TFIDF.RData")
save(tcm, file = "representations-symbolic-tcm.RData") # for GloVe

load("representations-symbolic-Bow-TFIDF.RData")





##################################
#       FEATURE EXTRACTION       #
#     DOCUMENT FREQUENCY (DF)    #
##################################

document.frequency <- vocab$doc_count
names(document.frequency) <- vocab$term
document.frequency <- document.frequency[order(document.frequency, decreasing = TRUE)]




##################################
#       FEATURE EXTRACTION       #
#       TERM STRENGTH (TS)       #
##################################

# distances among reviews with cosine (sample of 5.000)
set.seed(456)
ts.sample <- sample(1:25000, 5000)
tfidf.ts.sample <- tfidf[ts.sample, ]
tfidf.similarity <- sim2(tfidf.ts.sample, method = "cosine", norm = "l2")

ts.tokens <- word_tokenizer(imdb[ts.sample])
ts.it <- itoken(ts.tokens, progressbar = TRUE)
ts.vocab <- create_vocabulary(ts.it)
# remove terms that were deleted for dtm
ts.vocab <- ts.vocab[ts.vocab$term %in% vocab$term, ]
ts.vectorizer <- vocab_vectorizer(ts.vocab)
ts.dtm <- dtm[ts.sample, ]
length(ts.vocab$term) # -> 21.186

# 80.000 related reviews
ts.cutoff <- quantile(tfidf.similarity, probs = 0.96)
ts.related.reviews <- as.matrix(tfidf.similarity) > ts.cutoff
ts.related.reviews <- which(ts.related.reviews, arr.ind = TRUE)
ts.related.reviews <- ts.related.reviews[ts.related.reviews[,1] < ts.related.reviews[,2], ]



calculate.ts.for.term <- function(term, ts.dtm, ts.related.reviews) {
  # Calculates TS for term, given pairs of related reviews in 
  #   ts.related.reviews
  term.index <- which(colnames(ts.dtm) == term)
  term.present <- ts.dtm[, term.index] > 0
  
  in.first  <- term.present[ts.related.reviews[, 1]]
  in.second <- term.present[ts.related.reviews[, 2]]
  
  count.in.first <- sum(in.first)
  count.in.both  <- sum(in.first & in.second)
  
  return(count.in.both / count.in.first)
}

term.strength <- numeric(length(ts.vocab$term))
names(term.strength) <- ts.vocab$term

cl <- makeCluster(detectCores() - 1)
clusterExport(cl, varlist = c("ts.dtm", "ts.related.reviews", "calculate.ts.for.term"), envir = environment())
start <- Sys.time()
term.strength <- parSapply(cl, ts.vocab$term,
  function(term) calculate.ts.for.term(term, ts.dtm, ts.related.reviews)
)
end <- Sys.time()
end - start
stopCluster(cl)

term.strength[is.nan(term.strength)] <- 0
term.strength <- term.strength[order(term.strength, decreasing = TRUE)]




##################################
#       FEATURE EXTRACTION       #
#  ENTROPY BASED RANKING (EBR)   #
##################################

# Sample of reviews - a distance matrix is computed #terms times
set.seed(4567)
ebr.sample <- sample(1:25000, 200)
ebr.tokens <- word_tokenizer(imdb[ebr.sample])
ebr.it <- itoken(ebr.tokens, progressbar = TRUE)
ebr.vocab <- create_vocabulary(ebr.it)
# remove terms that were deleted for dtm
ebr.vocab <- ebr.vocab[ebr.vocab$term %in% vocab$term, ]
ebr.vectorizer <- vocab_vectorizer(ebr.vocab)
ebr.dtm <- create_dtm(ebr.it, ebr.vectorizer)
length(ebr.vocab$term) # -> 5.360

calculate.ebr.for.term <- function(term, ebr.dtm) {
  # Calculates EBR for term, calculating distances with BoW
  term.index <- which(colnames(ebr.dtm) == term)
  dtm.without.term <- ebr.dtm[, -term.index, drop=FALSE]
  
  dist.mat <- proxy::dist(as.matrix(dtm.without.term), method = "cosine")
  dist.mat <- as.matrix(dist.mat)
  
  mean.dist <- mean(dist.mat[lower.tri(dist.mat)])
  S <- 2^(-dist.mat / mean.dist)
  eps <- 1e-10
  S <- pmin(pmax(S, eps), 1 - eps)
  
  entropy <- S * log2(S) + (1 - S) * log2(1 - S)
  return(-sum(entropy))
}

entropy.based.ranking <- numeric(length(ebr.vocab$term))
names(entropy.based.ranking) <- ebr.vocab$term

cl <- makeCluster(detectCores() - 1)
clusterExport(cl, varlist = c("ebr.dtm", "calculate.ebr.for.term"), envir = environment())
clusterEvalQ(cl, library(proxy))
clusterEvalQ(cl, library(Matrix))
start <- Sys.time()
entropy.based.ranking <- parSapply(cl, ebr.vocab$term,
  function(t) calculate.ebr.for.term(t, ebr.dtm)
)
end <- Sys.time()
end - start
stopCluster(cl)

# Sign change to sort them from best to worst
entropy.based.ranking <- -entropy.based.ranking
entropy.based.ranking <- entropy.based.ranking[order(entropy.based.ranking, decreasing = TRUE)]




##################################
#       FEATURE EXTRACTION       #
#     TERM CONTRIBUTION (TC)     #
##################################

calculate.tc.for.term <- function(term, tfidf) {
  # Calculates TC for term, leveraging on:
  #     Dot product: sum(vec[i] * vec[j]) for i != j
  #     is equivalent to: (sum(vec)^2 - sum(vec^2))
  term.index <- which(colnames(tfidf) == term)
  term.vector <- tfidf[, term.index]
  return(sum(term.vector)^2 - sum(term.vector^2))
}

term.contribution <- numeric(length(vocab$term))
names(term.contribution) <- vocab$term

cl <- makeCluster(detectCores() - 1)
clusterExport(cl, varlist = c("tfidf", "calculate.tc.for.term"), envir = environment())
start <- Sys.time()
term.contribution <- parSapply(cl, vocab$term, function(t) calculate.tc.for.term(t, tfidf))
end <- Sys.time()
end - start
stopCluster(cl)

term.contribution <- term.contribution[order(term.contribution, decreasing = TRUE)]




##################################
#   STORE / LOAD F.E. METRICS    #
##################################

# Number of terms evaluated by each F.E. metric
cat(paste(
  "Document Frecuency (DF):\t", length(document.frequency), "\n",
  "Term Strength (TS):\t\t", length(term.strength), "\n",
  "Entropy Based Ranking (EBR):\t", length(entropy.based.ranking), "\n",
  "Term Contibution (TC):\t\t", length(term.contribution), "\n", sep = ""
))

save(
  document.frequency,
  term.strength,
  entropy.based.ranking,
  term.contribution,
  file = "representations-symbolic-features.RData"
)

load("representations-symbolic-features.RData")









##################################
#       PERFORMANCE TABLE        #
##################################

feature.extraction.method <- c(
  "Document Frequency", "Term Strength", 
  "Entropy-based Ranking", "Term Contribution"
)
feature.extraction.amount <- c(10, 25, 50, 100, 250, 500, 1000, 2500)
representation <- c("BoW", "TF-IDF")
performance <- expand.grid(
  clustering.algorithm = "k-medias",
  representation = representation,
  feature.extraction.method = feature.extraction.method,
  feature.extraction.amount = feature.extraction.amount,
  stringsAsFactors = FALSE
)
performance$feature.extraction.amount <- as.numeric(performance$feature.extraction.amount)
performance$dunn <- NA
performance$silhouette <- NA
performance$accuracy <- NA
dim(performance) # -> 64, 7






#####################################
#      CLUSTERING WITH K-MEANS      #
#####################################

# Clustering parameters
clustering.seed <- 456
clustering.nstart <- 10
clustering.max.iter <- 100
clustering.validation.sample.size <- 2000
clustering.validation.sample.amount <- 10


clustering.get.data <- function(performance.index) {
  # Returns the feature matrix for BoW or TF-IDF models depending on
  #     the F.E. method and amount of features specified in the performance
  #     table by performance.index
  representation <- as.character(performance$representation[performance.index])
  fe.method <- as.character(performance$feature.extraction.method[performance.index])
  fe.amount <- performance$feature.extraction.amount[performance.index]
  
  # Feature extraction
  fe.criterion <- switch(
    fe.method,
    "Document Frequency" = document.frequency,
    "Term Strength" = term.strength,
    "Entropy-based Ranking" = entropy.based.ranking,
    "Term Contribution" = term.contribution
  )
  extracted.features <- names(fe.criterion)[1:fe.amount]
  
  x <- if (representation == "BoW") { dtm[, extracted.features] } 
  else { tfidf[, extracted.features] }
  return(x)
}


# Optional filters to exclude tests from execution
performance.indexes <- which(
  performance$clustering.algorithm == "k-medias" # &
  # performance$feature.extraction.method == "Entropy-based Ranking" &
  # performance$feature.extraction.amount < 100
)

cl <- makeCluster(8)
clusterExport(cl, varlist = c(
  "clustering.get.data", "clustering.log.results",
  "performance", "dtm", "tfidf",
  "document.frequency", "term.strength", "entropy.based.ranking", "term.contribution",
  "clustering.nstart", "clustering.seed", "clustering.max.iter",
  "clustering.validation.sample.size", "clustering.validation.sample.amount", "imdb.true"
))
clusterEvalQ(cl, library(Matrix))
clusterEvalQ(cl, library(yardstick))
clusterEvalQ(cl, library(proxy))
clusterEvalQ(cl, library(fpc))

# Launch clustering
file.create("clustering-log.txt")
clustering.results <- parLapply(cl, performance.indexes, clustering.with.kmeans)
save(clustering.results, file = "representations-symbolic-cluster.RData")

load("representations-symbolic-cluster.RData")
sapply(clustering.results, function(res) table(res$result$cluster))


# Launch validation
file.create("clustering-validation-log.txt")
clustering.validation.results <- parLapply(cl, clustering.results, clustering.validation)

stopCluster(cl)


# Save results
for (res in clustering.validation.results) {
  performance$dunn[res$index] <- mean(res$dunn)
  performance$silhouette[res$index] <- mean(res$silhouette)
  performance$accuracy[res$index] <- res$accuracy
}
save(performance, clustering.validation.results, file = "representations-symbolic-performance.RData")

load("representations-symbolic-performance.RData")







#####################################
#           RESULT GRAPHS           #
#####################################
# 2 types of graphs are made:       #
#     1st. with facet wrap          #
#     2nd. independent for every    #
#          evaluation metric        #
#####################################
performance.long <- pivot_longer(
  performance,
  cols = c(silhouette, dunn, accuracy),
  names_to = "metric",
  values_to = "score"
)

# Graph with facet wrap
performance.long <- performance.long %>%
  mutate(
    group = paste(representation, feature.extraction.method, sep = " | "),
    is.ebr = ifelse(feature.extraction.method == "Entropy-based Ranking", "EBR", "Resto")
  )

ggplot(performance.long, aes(
    x = feature.extraction.amount, y = score, 
    color = representation, group = group, 
  )) +
  geom_line(aes(color = representation), linewidth = 1, alpha = 0.9) +
  geom_point(aes(shape = is.ebr), size = 3) +
  scale_color_manual(
    name = "Representation",
    values = c("TF-IDF" = "#0072B2", "BoW" = "#E69F00")
  ) + 
  scale_shape_manual(
    name = "Feature extraction\nmethod",
    values = c("EBR" = 16),
    labels = c("Entropy-based Ranking")
  ) +
  scale_x_log10() +
  facet_wrap(
    ~ metric, scales = "free_y", 
    labeller = labeller(metric = c(
      silhouette = "Average silhouette index",
      dunn = "Dunn index",
      accuracy = "Accuracy"
    ))
  ) +
  labs(
    title = "Clustering performance for symbolic representations",
    subtitle = "BoW and TF-IDF are combined with the F.E. methods DF, TS, EBR, and TC",
    x = "Number of extracted features (log scale)",
    y = "Evaluation metric value"
  ) +
  theme_bw(base_size = 16) +
  theme(legend.position = "bottom")



# Individual graphs
graph.params.list <- list(
  list(
    metric = "accuracy", name = "Accuracy",
    scale.y.limits = c(0.50, max(performance$accuracy)),
    scale.y.labels = scales::percent_format(accuracy = 1)
  ),
  list(
    metric = "dunn", name = "Dunn index",
    scale.y.limits = c(-0.01, 0.45),
    scale.y.labels = waiver()
  ),
  list(
    metric = "silhouette", name = "Average silhouette index",
    scale.y.limits = c(-0.1, 0.2),
    scale.y.labels = waiver()
  )
)
for (graph.params in graph.params.list) {
  label.data <- performance.long %>% 
    filter(metric == graph.params$metric) %>%
    group_by(group) %>%
    filter(feature.extraction.amount == max(feature.extraction.amount)) %>%
    ungroup()

  set.seed(456)
  p <- performance.long %>% 
    filter(metric == graph.params$metric) %>%
    ggplot(aes(
      x = feature.extraction.amount, y = score, 
      color = representation, group = group
    )) +
    geom_line(aes(color = representation), linewidth = 1, alpha = 0.9) +
    geom_point(size = 1.75) +
    geom_text_repel(
      data = label.data,
      aes(label = group, color = representation),
      size = 3.5, direction = "y", hjust = 0,
      nudge_x = 0.4, nudge_y = 0
    ) +
    scale_x_log10(
      breaks = c(10, 100, 1000),
      labels = scales::label_number(),
      expand = expansion(mult = c(0.01, 0.35))
    ) +
    scale_y_continuous(
      limits = graph.params$scale.y.limits,
      labels = graph.params$scale.y.labels
    ) +
    scale_color_manual(
      name = "Legend",
      values = c("TF-IDF" = "#0072B2", "BoW" = "#E69F00")
    ) + 
    theme_bw(base_size = 16) +
    theme(legend.position = "none") +
    labs(
      title = paste(graph.params$name, "for symbolic representations"),
      x = "Number of extracted features (log scale)",
      y = graph.params$name,
      color = "Line Type",
      size = "Line Type"      
    )
  
  print(p)
}
