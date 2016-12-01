setwd("~/Desktop/Other/MS_Courses/Capstone/DS-GA-1006-Project")
library(lda)
library(Metrics)

#Read in data
vocab = scan("modeling/vocab.txt", what="character", sep="\n")
train_price = as.numeric(scan("modeling/train_price.csv", what="character", sep="\n"))
test_price = as.numeric(scan("modeling/test_price.csv", what="character", sep="\n"))
train_docs = scan("modeling/train_docs.csv", what="character", sep="\n")
test_docs = scan("modeling/test_docs.csv", what="character", sep="\n")

#Lex the train/test docs and drop empty docs (after stop word filtering)
train_counts = lexicalize(train_docs, sep = " ", lower = TRUE, count = 1L, vocab = vocab)
empty_train_docs = c()
for (i in seq(1,length(train_counts))) {
  if (dim(train_counts[[i]])[2] == 0){
    empty_train_docs = append(empty_train_docs,i)
  }
}
train_counts = train_counts[-empty_train_docs]
train_price = train_price[-empty_train_docs]

test_counts = lexicalize(test_docs, sep = " ", lower = TRUE, count = 1L, vocab = vocab)

empty_test_docs = c()
for (i in seq(1,length(test_counts))) {
  if (dim(test_counts[[i]])[2] == 0){
    empty_test_docs = append(empty_test_docs,i)
  }
}
test_counts = test_counts[-empty_test_docs]
test_price = test_price[-empty_test_docs]

#Function to optimize K
get_mse_for_k = function(num_topics) {
  #Initial randomcoefficients in linear model for response
  params = sample(c(-1, 1), num_topics, replace=TRUE)
  cat(sprintf("Testing K = %d\n", num_topics))
  cat('Fitting model...\n')
  model = slda.em(documents=train_counts,
                  K=num_topics,
                  vocab=vocab,
                  num.e.iterations=10,
                  num.m.iterations=4,
                  alpha=1.0, eta=0.1,
                  train_price,
                  params,
                  variance=var(train_price), #Empirical variance of response
                  logistic=FALSE,
                  method="sLDA")
  
  cat('Making predictions...\n')
  predictions = slda.predict(test_counts,
                             model$topics,
                             model$model,
                             alpha = 1.0,
                             eta=0.1)
  
  cat('Finding MSE...\n')
  MSE = mse(predictions, test_price)
  return(MSE)
}

#Search for K over grid
test_Ks = seq(3,13,2)
MSEs = c()
for (K in test_Ks){
  MSE = get_mse_for_k(K)
  MSEs = append(MSEs, MSE)
}

#Visualizations- first plot adapted from demo(slda)
#Second plot uses LDAvis to explore topics

#Visual 1
library(ggplot2)
Topics = apply(top.topic.words(model$topics, 5, by.score=TRUE), 2, paste, collapse=" ")
coefs = data.frame(coef(summary(model$model)))
theme_set(theme_bw())
coefs = cbind(coefs, Topics=factor(Topics, Topics[order(coefs$Estimate)]))
coefs = coefs[order(coefs$Estimate),]
qplot(Topics, Estimate, colour=Estimate, data=coefs) + geom_errorbar(width=0.5, aes(ymin=Estimate-Std..Error, ymax=Estimate+Std..Error)) + coord_flip()

#Visual 2
library(LDAvis)
theta = t(apply(model$document_sums + 1, 2, function(x) x/sum(x)))
phi = t(apply(t(model$topics) + 0.1, 2, function(x) x/sum(x)))

list_for_LDA = list(phi = phi,
                    theta = theta,
                    doc.length = document.lengths(train_counts),
                    vocab = vocab,
                    term.frequency = word.counts(train_counts))

json = createJSON(phi = list_for_LDA$phi, 
                   theta = list_for_LDA$theta, 
                   doc.length = list_for_LDA$doc.length, 
                   vocab = list_for_LDA$vocab, 
                   term.frequency = list_for_LDA$term.frequency)

serVis(json, out.dir = 'vis', open.browser = TRUE)
