setwd("~/Desktop/Other/MS_Courses/Capstone/DS-GA-1006-Project/modeling/")
library(lda)
library(Metrics)
#----------------------------------------------------#
### First build and evaluate model using train/val sets,
### saving topic weights to predict using GBR (for comparison)
### with straight LDA from sklearn
#----------------------------------------------------#

#Read in data
vocab = scan("vocab.txt", what="character", sep="\n")
train_price = as.numeric(scan("train_price.csv", what="character", sep="\n"))
test_price = as.numeric(scan("test_price.csv", what="character", sep="\n"))
train_docs = scan("train_docs.csv", what="character", sep="\n")
test_docs = scan("test_docs.csv", what="character", sep="\n")

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

cat('Fitting model...\n')
params = sample(c(-1, 1), 10, replace=TRUE)
model = slda.em(documents=train_counts,
                  K=10,
                  vocab=vocab,
                  num.e.iterations=10,
                  num.m.iterations=4,
                  alpha=1.0, eta=0.1,
                  train_price,
                  params,
                  variance=var(train_price), #Empirical variance of response
                  logistic=FALSE,
                  method="sLDA")

cat('Getting docsums...\n')

train_docsums = slda.predict.docsums(train_counts,
                                    model$topics,
                                    alpha = 1.0,
                                    eta=0.1)

train_topic_weights = apply(train_docsums, 2, function(x) x / sum(x))

test_docsums = slda.predict.docsums(test_counts,
                             model$topics,
                             alpha = 1.0,
                             eta=0.1)

test_topic_weights = apply(test_docsums, 2, function(x) x / sum(x))

write.table(train_topic_weights, file="sLDA_10_train.csv", sep=",", row.names = FALSE, col.names = FALSE)
write.table(test_topic_weights, file="sLDA_10_val.csv", sep=",", row.names = FALSE, col.names = FALSE)
write.table(train_price, file="sLDA_10_train_price.csv", sep=",", row.names = FALSE, col.names = FALSE)
write.table(test_price, file="sLDA_10_val_price.csv", sep=",", row.names = FALSE, col.names = FALSE)

#----------------------------------------------------#
### Next build model using train/test sets for final featurization
#----------------------------------------------------#

all_train_data = read.csv('all_train_data.csv', header = TRUE)
train_counts = lexicalize(all_train_data$description, sep = " ", lower = TRUE, count = 1L, vocab = vocab)
empty_train_docs = c()
for (i in seq(1,length(train_counts))) {
  if (dim(train_counts[[i]])[2] == 0){
    empty_train_docs = append(empty_train_docs,i)
  }
}
train_ids = all_train_data$saleid[-empty_train_docs]
train_counts = train_counts[-empty_train_docs]
train_price = all_train_data$price[-empty_train_docs]

all_test_data = read.csv('all_test_data.csv', header = TRUE)
test_counts = lexicalize(all_test_data$description, sep = " ", lower = TRUE, count = 1L, vocab = vocab)
empty_test_docs = c()
for (i in seq(1,length(test_counts))) {
  if (dim(test_counts[[i]])[2] == 0){
    empty_test_docs = append(empty_test_docs,i)
  }
}
test_ids = all_test_data$saleid[-empty_test_docs]
test_counts = test_counts[-empty_test_docs]
test_price = all_test_data$price[-empty_test_docs]

cat('Fitting model...\n')
params = sample(c(-1, 1), 10, replace=TRUE)
model = slda.em(documents=train_counts,
                K=10,
                vocab=vocab,
                num.e.iterations=10,
                num.m.iterations=4,
                alpha=1.0, eta=0.1,
                train_price,
                params,
                variance=var(train_price), #Empirical variance of response
                logistic=FALSE,
                method="sLDA")

cat('Getting docsums...\n')
train_docsums = slda.predict.docsums(train_counts,
                                    model$topics,
                                    alpha = 1.0,
                                    eta=0.1)
train_topic_weights = apply(train_docsums, 2, function(x) x / sum(x))
train_topic_weights = data.frame(t(train_topic_weights))
colnames(train_topic_weights) = c('topic1','topic2','topic3','topic4','topic5','topic6','topic7','topic8','topic9','topic10')

test_docsums = slda.predict.docsums(test_counts,
                                    model$topics,
                                    alpha = 1.0,
                                    eta=0.1)
test_topic_weights = apply(test_docsums, 2, function(x) x / sum(x))
test_topic_weights = data.frame(t(test_topic_weights))
colnames(test_topic_weights) = c('topic1','topic2','topic3','topic4','topic5','topic6','topic7','topic8','topic9','topic10')


train_final = cbind(train_ids, train_topic_weights)
colnames(train_final)[1]= "saleid"
test_final = cbind(test_ids, test_topic_weights)
colnames(test_final)[1]= "saleid"

write.table(train_ids, file="final_train_ids.txt", sep=",", row.names = FALSE, col.names = FALSE, quote=FALSE)
write.table(test_ids, file="final_test_ids.txt", sep=",", row.names = FALSE, col.names = FALSE, quote=FALSE)
write.table(rbind(train_final,test_final), file="final_sLDA_weights.csv", sep=",", row.names = FALSE, quote=FALSE)

#----------------------------------------------------#
### Finally create final LDA visualizations
#----------------------------------------------------#

#Visual 1
library(ggplot2)
jpeg(file = 'figs/topic_weights.jpeg', quality=100)
Topics = apply(top.topic.words(model$topics, 5, by.score=TRUE), 2, paste, collapse=" ")
coefs = data.frame(coef(summary(model$model)))
theme_set(theme_bw())
coefs = cbind(coefs, Topics=factor(Topics, Topics[order(coefs$Estimate)]))
coefs = coefs[order(coefs$Estimate),]
qplot(Topics, Estimate, colour=Estimate, data=coefs, main='Topic weights- predicted\nlog10(price) for each topic') + coord_flip()
dev.off()

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

serVis(json, out.dir = 'vis', open.browser = FALSE)