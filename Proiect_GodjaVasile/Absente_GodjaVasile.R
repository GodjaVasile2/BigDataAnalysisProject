library(rpart)
library(rpart.plot)
library(tidyverse)
library(rsample)
library(caret)
library(partykit)
library(tree)
library(ipred)
library(ranger)
library(pROC)


absente<- read_csv("C:\\Users\\user\\Desktop\\Big DATA\\absente_cleaned.csv")

#extragem denumirea coloanelor
names(absente)
view(absente)



# analizam graficele pentru atributele numerice
absente %>%
  select_if(is.numeric) %>%
  gather(metric,value) %>%
  ggplot(aes(value, fill=metric)) +
  geom_density(show.legend = FALSE) +
  facet_wrap(~metric, scales = "free")

absente <- absente %>% 
  mutate(ReasonForAbsence = factor(ReasonForAbsence),
         MonthOfAbsence = factor(MonthOfAbsence),
         DayOfTheWeek = factor(DayOfTheWeek)
         ,Education = factor(Education)
         ,DayOfTheWeek = factor(DayOfTheWeek)
         ,SocialSmoker = factor(SocialSmoker)
         ,SocialDrinker = factor(SocialDrinker)
         ,DisciplinaryFailure = factor(DisciplinaryFailure))


ggplot(data = absente, aes(x = AbsenteeismTimeInHours)) +
  geom_density(aes(y = ..density..), color = "black", fill = "lightblue") +
  labs(title = "Distribuția absenteism",
       x = "Ore lipsite",
       y = "Densitate") +
  theme_minimal()

mean_absenteeism <- mean(absente$AbsenteeismTimeInHours)
mean_absenteeism

absente <- absente %>% 
  mutate(AbsenteeismTimeInHours = ifelse(AbsenteeismTimeInHours <= 7, "Low", "High"))
absente <- absente %>% mutate(AbsenteeismTimeInHours = factor(AbsenteeismTimeInHours))
table(absente$AbsenteeismTimeInHours)



#generare numere aleatoare 
set.seed(123)
#se fac spliturile bazate pe proportia initiala 
absente_split <- initial_split(absente, prop = 0.7, strata = "AbsenteeismTimeInHours")
absente_train <- training(absente_split)
absente_test <- testing(absente_split)

# Verificați numărul de înregistrări pentru fiecare categorie în seturile de antrenare și testare
table(absente_train$AbsenteeismTimeInHours)
table(absente_test$AbsenteeismTimeInHours)



#1 ARBORE

set.seed(123)
arb1 = rpart(
  formula = AbsenteeismTimeInHours ~. ,
  data = absente_train,
  method = "class"
)
arb1
summary(arb1)
rpart.plot(arb1)
plotcp(arb1)

#predictie cu arborele m1  pe setul de test 
pred_arb1 <- predict(arb1, newdata = absente_test, target = "class")

pred_arb1 <- as_tibble(pred_arb1) %>% mutate(class = ifelse(Low >= High, "Low", "High"))
table(pred_arb1$class, absente_test$AbsenteeismTimeInHours)
confusionMatrix(factor(pred_arb1$class), factor(absente_test$AbsenteeismTimeInHours))



#2 ARBORE PRUNED
arb1$cptable

set.seed(123)
arb1_pruned <- prune(arb1, cp = 0.02579)
arb1_pruned
summary(arb1_pruned)
rpart.plot(arb1_pruned)
plotcp(arb1_pruned)

pred_arb1_pruned <- predict(arb1_pruned, newdata = absente_test, target = "class")
pred_arb1_pruned <- as_tibble(pred_arb1_pruned) %>% 
  mutate(class = ifelse(Low >= High, "Low", "High"))
table(pred_arb1_pruned$class,absente_test$AbsenteeismTimeInHours)
confusionMatrix(factor(pred_arb1_pruned$class), factor(absente_test$AbsenteeismTimeInHours))


#3 ARBORE CP 0
set.seed(123)
arb2 <- rpart(AbsenteeismTimeInHours ~., 
                 data = absente_train,
                 method = "class",
                 control = list(cp=0))
arb2
summary(arb2)
rpart.plot(arb2)
plotcp(arb2)
arb2$cptable  # afisarea parametrilor alpha 


pred_arb2 <- predict(arb2, newdata = absente_test, target = "class")
pred_arb2 <- as_tibble(pred_arb2) %>% 
  mutate(class = ifelse(Low >= High, "Low", "High"))
table(pred_arb2$class,absente_test$AbsenteeismTimeInHours)
confusionMatrix(factor(pred_arb2$class), factor(absente_test$AbsenteeismTimeInHours))


#4 ARBORE TREE GINI

set.seed(123)
arb_gini <- tree(AbsenteeismTimeInHours ~., data = absente_train, split="gini") # works with Gini index
arb_gini
summary(arb_gini)

pred_arb_gini <- predict(arb_gini, newdata = absente_test, target = "class")
pred_arb_gini <- as_tibble(pred_arb_gini) %>% mutate(class = ifelse(Low >= High, "Low", "High"))
confusionMatrix(factor(pred_arb_gini$class), factor(absente_test$AbsenteeismTimeInHours))


#5 BOOTSTRAP AGGREGATIONG(BAGGING)

set.seed(123)
arb_bagg <- bagging(AbsenteeismTimeInHours ~ .,
                     data = absente_train, coob = TRUE)
arb_bagg
summary(arb_bagg)
pred_arb_bagg <- predict(arb_bagg, newdata = absente_test, target = "class")
confusionMatrix(pred_arb_bagg, factor(absente_test$AbsenteeismTimeInHours))


ntree <- seq(10, 50, by = 1)
misclassification <- vector(mode = "numeric", length = length(ntree))
for (i in seq_along(ntree)) {
  set.seed(123)
  model <- bagging( 
    AbsenteeismTimeInHours ~.,
    data = absente_train,
    coob = TRUE,
    nbag = ntree[i])
  misclassification[i] = model$err
}

plot(ntree, misclassification, type="l", lwd="2", xaxt = "n")  
axis(side = 1, at = seq(10, 50, by = 1))  


#48 BAGS PETRU A STABILIZA ERROR RATE
arb_bagg_48 <- bagging(AbsenteeismTimeInHours ~ .,
                        data = absente_train, coob = TRUE, nbag = 48)
arb_bagg_48
summary(arb_bagg_48)
pred_arb_bagg_48 <- predict(arb_bagg_48, newdata = absente_test, target = "class")
confusionMatrix(pred_arb_bagg_48, factor(absente_test$AbsenteeismTimeInHours))


###################################### NAIVE BAYSE ###########################################



absente %>%
  filter(AbsenteeismTimeInHours == "High") %>%    
  select_if(is.numeric) %>%       
  cor() %>%                     
  corrplot::corrplot()

absente %>%
  filter(AbsenteeismTimeInHours == "Low") %>%    
  select_if(is.numeric) %>%         
  cor() %>%                   
  corrplot::corrplot()


features <- setdiff(names(absente), "AbsenteeismTimeInHours")
x <- absente_train[,features]
y <- absente_train$AbsenteeismTimeInHours

train_control <- trainControl(
  method = "cv",
  number = 10 )


naive_b1<- train(
  x = x,
  y = y,
  method = "nb",
  trControl = train_control)

confusionMatrix(naive_b1)
naive_b1

pred_nb1 <- predict(naive_b1, absente_test)
confusionMatrix(pred_nb1, absente_test$AbsenteeismTimeInHours)


# combinatii posibile de atribute
search_grid <- expand.grid(
  usekernel = c(TRUE, FALSE),   
  fL = 0.5,                    
  adjust = seq(0, 5, by = 1))    


# model de antrenare
naive_b2 <- train(
  x = x,
  y = y,
  method = "nb",
  trControl = train_control,
  tuneGrid = search_grid)

confusionMatrix(naive_b2)
naive_b2

# TOP 10 
naive_b2$result %>%
  top_n(10, wt = Accuracy) %>%
  arrange(desc(Accuracy))
 

pred_nb2 <- predict(naive_b2, absente_test)
confusionMatrix(pred_nb2, absente_test$AbsenteeismTimeInHours)

############################################### COMPARARE REZULTATE ######################################################


pred_arb1_roc <- predict(arb1, absente_test, type = "prob")
pred_arb1_pruned_roc <- predict(arb1_pruned, absente_test, type = "prob")
pred_arb2_roc <- predict(arb2, absente_test, type = "prob")
pred_arb_gini_roc <- predict(arb_gini, absente_test)
pred_arb_bagg_48_roc <- predict(arb_bagg_48, newdata = absente_test, type = "prob")
pred_naive_b2_roc <- predict(naive_b2, absente_test, type = "prob")


# ARBORE 1
dataset_arb1 <- data.frame(
  actual.class = absente_test$AbsenteeismTimeInHours,
  probability = pred_arb1_roc[,1]
)
roc.val_arb1 <- roc(actual.class ~ probability, dataset_arb1)
adf_arb1 <- data.frame(  
  specificity = roc.val_arb1$specificities, 
  sensitivity = roc.val_arb1$sensitivities)

# ARBORE 1 PRUNED
dataset_arb1_pruned <- data.frame(
  actual.class = absente_test$AbsenteeismTimeInHours,
  probability = pred_arb1_pruned_roc[,1]
)
roc.val_arb1_pruned <- roc(actual.class ~ probability, dataset_arb1_pruned)
adf_arb1_pruned <- data.frame(  
  specificity = roc.val_arb1_pruned$specificities, 
  sensitivity = roc.val_arb1_pruned$sensitivities)

# ARBORE 2 CP=0
dataset_arb2 <- data.frame(
  actual.class = absente_test$AbsenteeismTimeInHours,
  probability = pred_arb2_roc[,1]
)
roc.val_arb2 <- roc(actual.class ~ probability, dataset_arb2)
adf_arb2 <- data.frame(  
  specificity = roc.val_arb2$specificities, 
  sensitivity = roc.val_arb2$sensitivities)

# ARBORE GINI
dataset_arb_gini <- data.frame(
  actual.class = absente_test$AbsenteeismTimeInHours,
  probability = pred_arb_gini_roc[,1]
)
roc.val_arb_gini <- roc(actual.class ~ probability, dataset_arb_gini)
adf_arb_gini <- data.frame(  
  specificity = roc.val_arb_gini$specificities, 
  sensitivity = roc.val_arb_gini$sensitivities)

# ARBORE BAGGING 48
dataset_arb_bagg_48 <- data.frame(
  actual.class = absente_test$AbsenteeismTimeInHours,
  probability = pred_arb_bagg_48_roc[,1]
)
roc.val_arb_bagg_48 <- roc(actual.class ~ probability, dataset_arb_bagg_48)
adf_arb_bagg_48 <- data.frame(  
  specificity = roc.val_arb_bagg_48$specificities, 
  sensitivity = roc.val_arb_bagg_48$sensitivities)

# NAIVE BAYES 2
dataset_naive_b2 <- data.frame(
  actual.class = absente_test$AbsenteeismTimeInHours,
  probability = pred_naive_b2_roc[,1]
)
roc.val_naive_b2 <- roc(actual.class ~ probability, dataset_naive_b2)
adf_naive_b2 <- data.frame(  
  specificity = roc.val_naive_b2$specificities, 
  sensitivity = roc.val_naive_b2$sensitivities)



ggplot() +
  geom_line(data=adf_arb1, aes(specificity, sensitivity, color = 'Arb1'), size = 1) +
  geom_line(data=adf_arb1_pruned, aes(specificity, sensitivity, color = 'Arb1 Pruned'), size = 1) +
  geom_line(data=adf_arb2, aes(specificity, sensitivity, color = 'Arb2'), size = 1) +
  geom_line(data=adf_arb_gini, aes(specificity, sensitivity, color = 'Arb Gini'), size = 1) +
  geom_line(data=adf_arb_bagg_48, aes(specificity, sensitivity, color = 'Arb Bagg 48'), size = 1) +
  geom_line(data=adf_naive_b2, aes(specificity, sensitivity, color = 'Naive B2'), size = 1) +
  scale_color_manual(values = c('Arb1' = 'deeppink', 'Arb1 Pruned' = 'purple4', 'Arb2' = 'green', 'Arb Gini' = 'blue', 'Arb Bagg 48' = 'cyan', 'Naive B2' = 'yellow')) +
  scale_x_reverse() +
  labs(x = "1 - Specificity (False Positive Rate)", y = "Sensitivity (True Positive Rate)",
       title = "ROC Curves", color = "Models") +
  theme_minimal() +
  theme(text = element_text(size = 17),
        plot.title = element_text(hjust = 0.5),
        legend.position = "bottom")

