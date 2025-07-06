# Sentiment Analysis

This folder contains the R code and visualizations used to evaluate different sentiment analysis lexicon-based approaches applied to the IMDb Movie Reviews dataset (25.000 reviews from test subset).



## 1. Methodology Overview

All of the available dictionaries from the following R packages were used:

 - `syuzhet`
 - `tidytext`
 - `sentimentr`
 - `SentimentAnalysis`



## 2. Results

To evaluate the performance of the sentiment analysis approaches, the metrics Accuracy and F1-Score are employed.

The following tables show the performance of every method, where the rows represent a given dictionary, and the columns the R package.


```r
> round(accuracy, 4)
               syuzhet tidytext sentimentr SentimentAnalysis
GI                  NA       NA         NA            0.6410
Bing            0.7279   0.7313     0.7583                NA
AFINN           0.7103   0.6904         NA                NA
NRC             0.6567   0.6584     0.6925                NA
Syuzhet         0.6887       NA     0.7267                NA
QDAP                NA       NA         NA            0.6626
SentiWordNet        NA       NA     0.6331                NA
Henry               NA       NA         NA            0.5718
Loughran            NA       NA     0.6853            0.6514
Jockers-Rinker      NA       NA     0.7355                NA

> round(f1, 4)
               syuzhet tidytext sentimentr SentimentAnalysis
GI                  NA       NA         NA            0.5336
Bing            0.7492   0.7389     0.7554                NA
AFINN           0.6683   0.6262         NA                NA
NRC             0.5840   0.5818     0.6345                NA
Syuzhet         0.6356       NA     0.6857                NA
QDAP                NA       NA         NA            0.5762
SentiWordNet        NA       NA     0.5762                NA
Henry               NA       NA         NA            0.4332
Loughran            NA       NA     0.6864            0.7028
Jockers-Rinker      NA       NA     0.7084                NA
```