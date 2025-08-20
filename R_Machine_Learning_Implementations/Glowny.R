## Ustawienia początkowe

# W poniższym projekcie instalowane są pakiety, które będą używane wyłącznie do porównania modeli zbudowanych samodzielnie 
# z modelami generowanymi za pomocą gotowych narzędzi. Pakiety te nie będą wykorzystywane w moich ręcznie budowanych modelach. 
# Oto krótkie informacje o używanych paczkach:
# rpart: Służy do budowy drzew decyzyjnych. 
# nnet: Implementuje proste sieci neuronowe.
# class: Służy do implementacji algorytmu k-Nearest Neighbors (KNN) dla problemu klasyfikacji binarnej 
# kknn: Służy do implementacji algorytmu k-Nearest Neighbors (KNN) dla problemu regresji
# Wszystkie te paczki będą używane do porównania jakości i skuteczności moich własnych modeli z modelami tworzonymi 
# za pomocą gotowych funkcji z R. Instalowane są również pakiety, służące do wizualizacji wyników - takie jak ggplot2 oraz reshape2.

if (!require(rpart)) install.packages("rpart")
if (!require(nnet)) install.packages("nnet")
if (!require(class)) install.packages("class")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(reshape2)) install.packages("reshape2")
if (!require(pROC)) install.packages("pROC")
if (!require(kknn)) install.packages("kknn")
library("kknn")
library("rpart")
library("nnet")
library("class")
library("ggplot2")
library("reshape2")
library("pROC")

# Określenie ścieżki dla folderu gdzie będą zapisywane generowane obiekty
generowane_obiekty <- file.path(getwd(), "generowane_obiekty")


## Wczytanie i wstępny pre-processing danych. Została dodana obsługa kilku popularnych błędów podczas wczytywania danych.
## Poprawne wczytanie danych zwraca w konsoli komunikat "Próba wczytania pliku zakończona pomyślnie.".

data <- tryCatch({
  file_path <- file.path(getwd(), "breast_cancer.csv")
  data <- read.csv(file_path)
  cat("Próba wczytania pliku zakończona pomyślnie.\n")
  data
}, warning = function(w) {
  cat("Ostrzeżenie: ", conditionMessage(w), "\n")
  cat("Próba wczytania pliku zakończona niepomyślnie. Dane nie zostały wczytane.\n")
  NULL  
}, error = function(e) {
  cat("Błąd: ", conditionMessage(e), "\n")
  cat("Próba wczytania pliku zakończona niepomyślnie. Dane nie zostały wczytane.\n")
  NULL  
})

# Przegląd informacji o wczytanych danych
head(data)
str(data)
dim(data)
summary(data)

data <- data[ , -1]  # Usunięcie kolumny ID, ponieważ jest ona niepotrzebna
data$X1 <- ifelse(data$X1 == "M", 1, 0)  # Zamiana M na 1, B na 0
data$X1 <- as.factor(data$X1)
head(data)
str(data)
dim(data)
summary(data)

# Dane do klasyfikacji wieloklasowej
data_wieloklasowa <- tryCatch({
  
  file_path <- file.path(getwd(), "glass.csv")
  data_wieloklasowa <- read.csv(file_path)
  cat("Próba wczytania pliku zakończona pomyślnie.\n")
  data_wieloklasowa 
}, warning = function(w) {
  cat("Ostrzeżenie: ", conditionMessage(w), "\n")
  cat("Próba wczytania pliku zakończona niepomyślnie. Dane nie zostały wczytane.\n")
  NULL  
}, error = function(e) {
  cat("Błąd: ", conditionMessage(e), "\n")
  cat("Próba wczytania pliku zakończona niepomyślnie. Dane nie zostały wczytane.\n")
  NULL
})

# Przegląd informacji o wczytanych danych
head(data_wieloklasowa)
str(data_wieloklasowa)
dim(data_wieloklasowa)
summary(data_wieloklasowa)

data_wieloklasowa <- data_wieloklasowa[, -1] # Usunięcie kolumny ID, ponieważ jest ona niepotrzebna
data_wieloklasowa$X10 <- as.factor(data_wieloklasowa$X10) # Ustawienie targetu

data_nn_multi <- data.frame(
  Target = data_wieloklasowa$X10, 
  data_wieloklasowa[, setdiff(names(data_wieloklasowa), "X10")] 
)


# Dane do regresji
data_regresja <- tryCatch({
  
  file_path <- file.path(getwd(), "abalone.csv")
  data_regresja <- read.csv(file_path)
  cat("Próba wczytania pliku zakończona pomyślnie.\n")
  data_regresja
}, warning = function(w) {
  cat("Ostrzeżenie: ", conditionMessage(w), "\n")
  cat("Próba wczytania pliku zakończona niepomyślnie. Dane nie zostały wczytane.\n")
  NULL  
}, error = function(e) {
  cat("Błąd: ", conditionMessage(e), "\n")
  cat("Próba wczytania pliku zakończona niepomyślnie. Dane nie zostały wczytane.\n")
  NULL 
})

# Przegląd informacji o wczytanych danych
head(data_regresja)
str(data_regresja)
dim(data_regresja)
summary(data_regresja)

# Kodowanie pierwszej zmiennej: 
data_regresja <- cbind(
  model.matrix(~ X0 - 1, data = data_regresja),
  data_regresja[, -1]
)

data_regresja_reordered <- data_regresja[, c("X8", "X1", "X2", "X3", "X4", "X5", "X6", "X7")]



#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#- Problem klasyfikacji binarnej
## Poniżej przedstawiono rozwiązanie problemu klasyfikacji binarnej z zastosowaniem trzech metod: k-najbliższych sąsiadów (KNN), 
## drzew decyzyjnych oraz sieci neuronowych. W pierwszej kolejności wykorzystano algorytmy zaimplementowane samodzielnie, a następnie 
## wyniki porównano z modelami stworzonymi przy użyciu wbudowanych funkcji z dostępnych pakietów.

#------------------------------------------------------------------------------- Metoda k-najbliższych sąsiadów (KNN)
set.seed(24061998)
parTune <- data.frame(k = c(3,5,7,9))

#-------------------------------------------------------------- Własny algorytm
execution_time <- system.time({
  wyniki_knn <- CrossValidTune(
    dane     = data,
    kFold    = 2,
    parTune  = parTune,
    seed     = 123,
    trainFn  = myTrainFnKNN,
    predictFn= myPredictFnKNN
  )
})

print(wyniki_knn)
print(execution_time)

typeof(wyniki_knn)

best_row_auc <- wyniki_knn[which.max(wyniki_knn$AUCW), ]
print("Najlepsza iteracja na podstawie AUCW:")
print(best_row_auc)

#-------------------------------------------------------------- Wbudowana paczka
execution_time <- system.time({
  results <- CrossValidTune(
    dane = data,
    kFold = 2,
    parTune = parTune,
    trainFn = myTrainFnKNNclass,
    predictFn = myPredictFnKNNclass
  )
})

print(results)
print(execution_time)

best_row_auc <- results[which.max(results$AUCW), ]
print("Najlepsza iteracja na podstawie AUCW:")
print(best_row_auc)

#------------------------------------------------------------------------------- Drzewa decyzyjne 
data_for_class <- data.frame(X1 = factor(data$X1), data[, -(1:2)])  
data_for_class$X1 <- factor(data$X1, levels = c(0,1))

parTune <- data.frame(
  depth   = c(1, 6),
  minobs  = c(1, 3),
  overfit = c("none","prune"),
  cf      = c(0.1, 0.4)
)

#-------------------------------------------------------------- Własny algorytm
execution_time <- system.time({
  wyniki_tree <- CrossValidTune(
    dane      = data_for_class,
    kFold     = 2,
    parTune   = parTune,
    seed      = 123,
    trainFn   = myTrainFnTree,
    predictFn = myPredictFnTree
  )
})

print(wyniki_tree)
print(execution_time)

best_row_auc_Tree <- wyniki_tree[which.max(wyniki_tree$AUCW), ]
print("Najlepsza iteracja na podstawie AUCW:")
print(best_row_auc_Tree)
#-------------------------------------------------------------- Wbudowana paczka
execution_time <- system.time({
  results <- CrossValidTune(
    dane      = data_for_class, 
    kFold     = 2,
    parTune   = parTune,
    seed      = 123,
    trainFn   = myTrainFnTreeRpart,
    predictFn = myPredictFnTreeRpart
  )
})

print(results)
print(execution_time)

best_row_auc_paczka <- results[which.max(results$AUCW), ]
print("Najlepsza iteracja na podstawie AUCW:")
print(best_row_auc_paczka)

#------------------------------------------------------------------------------- Sieci neuronowe 
#-------------------------------------------------------------- Własny algorytm
parTune <- data.frame(
  hsize = c(10, 15),
  lr    = c(0.05, 0.005),
  iter  = c(200, 2000),
  seed  = c(123, 123) 
)

execution_time <- system.time({
  wyniki_nn <- CrossValidTune(
    dane      = data,
    kFold     = 4,
    parTune   = parTune,  
    seed      = 321,
    trainFn   = myTrainFnNN,
    predictFn = myPredictFnNN
    
  )
})

print(wyniki_nn)
print(execution_time)

best_row_auc <- wyniki_nn[which.max(wyniki_nn$AUCW), ]
print("Najlepsza iteracja na podstawie AUCW:")
print(best_row_auc)
#-------------------------------------------------------------- Wbudowana paczka
parTune_nnet <- data.frame(
  size = c(5, 10),         
  decay = c(0.001, 0.01),  
  maxit = c(100, 200)
)

execution_time <- system.time({
  wyniki_nnet <- CrossValidTune(
    dane = data_for_class,
    kFold = 4,
    parTune = parTune_nnet,
    seed = 321,
    trainFn = myTrainFnNNet,
    predictFn = myPredictFnNNet
  )
})
print(wyniki_nnet)
print(execution_time)

best_row_auc_paczka <- wyniki_nnet[which.max(wyniki_nnet$AUCW), ]
print("Najlepsza iteracja na podstawie AUCW:")
print(best_row_auc_paczka)

#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#- Problem klasyfikacji wieloklasowej
## Poniżej przedstawiono rozwiązanie problemu klasyfikacji wieloklasowej z zastosowaniem trzech metod: k-najbliższych sąsiadów (KNN), 
## drzew decyzyjnych oraz sieci neuronowych. W pierwszej kolejności wykorzystano algorytmy zaimplementowane samodzielnie, a następnie 
## wyniki porównano z modelami stworzonymi przy użyciu wbudowanych funkcji z dostępnych pakietów.

#------------------------------------------------------------------------------- Metoda k-najbliższych sąsiadów (KNN)
#-------------------------------------------------------------- Własny algorytm
data_multi_cv <- data.frame(
  Target = data_wieloklasowa$X10, 
  data_wieloklasowa[, setdiff(names(data_wieloklasowa), "X10")]  
)

parTune <- data.frame(k = c(3, 5, 7, 9))


set.seed(24061998)
execution_time <- system.time({
  wyniki_knn_multi <- CrossValidTune(
    dane      = data_multi_cv, 
    kFold     = 5,                 
    parTune   = parTune,           
    seed      = 123,              
    trainFn   = myTrainFnKNN,    
    predictFn = myPredictFnKNN    
  )
})

print(wyniki_knn_multi)
print(execution_time)

print("Najlepsza iteracja na podstawie błędu F1 score:")
best_f1w_knn_wlasny <- wyniki_knn_multi[which.max(wyniki_knn_multi$MacroF1W), ]
print(best_f1w_knn_wlasny)

#-------------------------------------------------------------- Wbudowana paczka
parTune <- data.frame(k = c(3, 5, 7, 9))

set.seed(24061998)
execution_time <- system.time({
  wyniki_knn_multi <- CrossValidTune(
    dane      = data_multi_cv,    
    kFold     = 5,                 
    parTune   = parTune,          
    seed      = 123,              
    trainFn   = myTrainFnKNNmultiKnn,
    predictFn = myPredictFnKNNmultiKnn 
  )
})

print(wyniki_knn_multi)
print(execution_time)


print("Najlepsza iteracja na podstawie błędu F1 score:")
best_f1w_knn_wbudowany <- wyniki_knn_multi[which.max(wyniki_knn_multi$MacroF1W), ]
print(best_f1w_knn_wbudowany)

#------------------------------------------------------------------------------- Drzewa decyzyjne
#-------------------------------------------------------------- Własny algorytm
parTune <- data.frame(
  depth   = c(1, 3),       
  minobs  = c(10, 20),      
  overfit = c("none","prune"),
  cf      = c(0.3, 0.6)    
)

set.seed(24061998)

execution_time <- system.time({
  wyniki_tree_multi <- CrossValidTune(
    dane      = data_multi_cv,  
    kFold     = 5,              
    parTune   = parTune,          
    seed      = 123,          
    trainFn   = myTrainFnTree,    
    predictFn = myPredictFnTree
  )
})

print(wyniki_tree_multi)
print(execution_time)

print("Najlepsza iteracja na podstawie błędu F1 score:")
best_f1w_tree_wlasny <- wyniki_tree_multi[which.max(wyniki_tree_multi$MacroF1W), ]
print(best_f1w_tree_wlasny)
#-------------------------------------------------------------- Wbudowana paczka
parTune <- data.frame(
  depth   = c(3, 5),
  minobs  = c(5, 10), 
  overfit = c("none", "prune"), 
  cf      = c(0.1, 0.3)
)

set.seed(24061998)
execution_time <- system.time({
  wyniki_tree_multi <- CrossValidTune(
    dane      = data_multi_cv,
    kFold     = 5,                
    parTune   = parTune,
    seed      = 123,    
    trainFn   = myTrainFnTreeMultiRpart,
    predictFn = myPredictFnTreeMultiRpart  
  )
})

print(wyniki_tree_multi)
print(execution_time)


print("Najlepsza iteracja na podstawie błędu F1 score:")
best_f1w_tree_wbudowany <- wyniki_tree_multi[which.max(wyniki_tree_multi$MacroF1W), ]
print(best_f1w_tree_wbudowany)
#------------------------------------------------------------------------------- Sieci neuronowe
#-------------------------------------------------------------- Własny algorytm
parTune <- data.frame(
  size = c(500, 1000),       
  decay = c(0.1, 0.01),
  maxit = c(200, 50000)
)

set.seed(24061998)
execution_time <- system.time({
  wyniki_nn_multi <- CrossValidTune(
    dane      = data_nn_multi,     
    kFold     = 5,
    parTune   = parTune,           
    seed      = 123,
    trainFn   = myTrainFnNNMulti,
    predictFn = myPredictFnNNMulti
  )
})

print(wyniki_nn_multi)
print(execution_time)

print("Najlepsza iteracja na podstawie błędu F1 score:")
best_f1w_nn_wlasny <- wyniki_nn_multi[which.max(wyniki_nn_multi$MacroF1W), ]
print(best_f1w_nn_wlasny)
#-------------------------------------------------------------- Wbudowana paczka
data_nn_multi <- data.frame(
  Target = data_wieloklasowa$X10, 
  data_wieloklasowa[, setdiff(names(data_wieloklasowa), "X10")] 
)

parTune <- data.frame(
  size = c(5, 10),       
  decay = c(0.1, 0.01),
  maxit = c(200, 500) 
)

set.seed(24061998)
execution_time <- system.time({
  wyniki_nn_multi <- CrossValidTune(
    dane      = data_nn_multi,     
    kFold     = 5,             
    parTune   = parTune,
    seed      = 123,
    trainFn   = myTrainFnNNMultiNnet,
    predictFn = myPredictFnNNMultiNnet
  )
})

print(wyniki_nn_multi)
print(execution_time)

print("Najlepsza iteracja na podstawie błędu F1 score:")
best_f1w_nn_wbudowany <- wyniki_nn_multi[which.max(wyniki_nn_multi$MacroF1W), ]
print(best_f1w_nn_wbudowany)


#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#- Problem regresji
## Poniżej przedstawiono rozwiązanie problemu regresji liniowiej z zastosowaniem trzech metod: k-najbliższych sąsiadów (KNN), 
## drzew decyzyjnych oraz sieci neuronowych. W pierwszej kolejności wykorzystano algorytmy zaimplementowane samodzielnie, a następnie 
## wyniki porównano z modelami stworzonymi przy użyciu wbudowanych funkcji z dostępnych pakietów.

#------------------------------------------------------------------------------- Metoda k-najbliższych sąsiadów (KNN)
parTune <- data.frame(
  k = c(3, 5, 7, 9, 11))

#-------------------------------------------------------------- Własny algorytm
execution_time <- system.time({
  wyniki_knn_reg <- CrossValidTune(
    dane      = data_regresja_reordered,
    kFold     = 5,
    parTune   = parTune,
    seed      = 123,
    trainFn   = myTrainFnKNNreg,
    predictFn = myPredictFnKNNreg
  )
})

print(wyniki_knn_reg)
print(execution_time)


best_mape_knn_reg_wlasny <- wyniki_knn_reg[which.min(wyniki_knn_reg$MAPEw), ]
print("Najlepsza iteracja na podstawie błędu MAPE:")
print(best_mape_knn_reg_wlasny)

#-------------------------------------------------------------- Wbudowana paczka
execution_time <- system.time({
  wyniki_kknn <- CrossValidTune(
    dane = data_regresja_reordered,
    kFold = 5,
    parTune = parTune,
    seed = 123,
    trainFn = myTrainFnKKNN,
    predictFn = myPredictFnKKNN
  )
})

print(wyniki_kknn)
print(execution_time)

best_mape_knn_reg_wbudowany <- wyniki_kknn[which.min(wyniki_kknn$MAPEw), ]
print("Najlepsza iteracja na podstawie błędu MAPE:")
print(best_mape_knn_reg_wbudowany)

#------------------------------------------------------------------------------- Drzewa decyzyjne
#-------------------------------------------------------------- Własny algorytm
parTune <- data.frame(
  depth   = c(1,10),
  minobs  = c(100, 300),
  overfit = c("none","prune"),
  cf      = c(0.01, 0.4)
)

execution_time <- system.time({
  wyniki_tree_reg <- CrossValidTune(
    dane      = data_regresja_reordered,
    kFold     = 5,
    parTune   = parTune,
    seed      = 123,
    trainFn   = myTrainFnTreeReg,
    predictFn = myPredictFnTreeReg
  )
})

print(wyniki_tree_reg)
print(execution_time)

best_mape_tree_reg_wlasny <- wyniki_tree_reg[which.min(wyniki_tree_reg$MAPEw), ]
print("Najlepsza iteracja na podstawie błędu MAPE:")
print(best_mape_tree_reg_wlasny)

#-------------------------------------------------------------- Wbudowana paczka
parTune_rpart <- expand.grid(
  maxdepth = c(1, 10),    
  minsplit = c(100, 300), 
  overfit = c("none","prune"),
  cp       = c(0.01, 0.4)
)

execution_time <- system.time({
  wyniki_tree_reg_rpart <- CrossValidTune(
    dane      = data_regresja_reordered,
    kFold     = 5,  
    parTune   = parTune_rpart,
    seed      = 123,
    trainFn   = myTrainFnRpartReg,
    predictFn = myPredictFnRpartReg
  )
})

print(wyniki_tree_reg_rpart)
print(execution_time)


best_mape_tree_reg_wlasny <- wyniki_tree_reg_rpart[which.min(wyniki_tree_reg_rpart$MAPEw), ]
print("Najlepsza iteracja na podstawie błędu MAPE:")
print(best_mape_tree_reg_wlasny)

#------------------------------------------------------------------------------- Sieci neuronowe
#-------------------------------------------------------------- Własny algorytm
parTune <- data.frame(
  h1   = c(5, 10),
  h2   = c(5, 10),
  lr   = c(0.01, 0.001),
  iter = c(100, 1000),
  seed = c(123, 999)
)

execution_time <- system.time({
  wyniki_nn_reg <- CrossValidTune(
    dane      = data_regresja_reordered,
    kFold     = 5,
    parTune   = parTune,
    seed      = 321,              
    trainFn   = myTrainFnNNReg, 
    predictFn = myPredictFnNNReg
  )
})

head(wyniki_nn_reg)
print(execution_time)

best_mape_nn_reg_wlasny <- wyniki_nn_reg[which.min(wyniki_nn_reg$MAPEw), ]
print("Najlepsza iteracja na podstawie błędu MAPE:")
print(best_mape_nn_reg_wlasny)

#-------------------------------------------------------------- Wbudowana paczka
parTune_nnet_reg <- expand.grid(
  h1    = c(5, 10),        
  decay = c(0.1, 0.01),    
  iter  = c(500, 1000),
  seed  = c(123, 999)
)

execution_time <- system.time({
  wyniki_nn_reg <- CrossValidTune(
    dane      = data_regresja_reordered, 
    kFold     = 5,                       
    parTune   = parTune_nnet_reg,
    seed      = 321,
    trainFn   = myTrainFnNnetReg,
    predictFn = myPredictFnNnetReg
  )
})

print(wyniki_nn_reg)
print(execution_time)

best_mape_nn_reg_wbudowany <- wyniki_nn_reg[which.min(wyniki_nn_reg$MAPEw), ]
print("Najlepsza iteracja na podstawie błędu MAPE:")
print(best_mape_nn_reg_wbudowany)


#      ________  ________  ________  ___       ________  ________  ________       ___    ___ 
#     |\   ____\|\_____  \|\   __  \|\  \     |\   __  \|\   ____\|\   ___  \    |\  \  /  /|
#     \ \  \___|_\|___/  /\ \  \|\  \ \  \    \ \  \|\  \ \  \___|\ \  \\ \  \   \ \  \/  / /
#      \ \_____  \   /  / /\ \   __  \ \  \    \ \   __  \ \_____  \ \  \\ \  \   \ \    / / 
#       \|____|\  \ /  /_/__\ \  \ \  \ \  \____\ \  \ \  \|____|\  \ \  \\ \  \   \/  /  /  
#         ____\_\  \\________\ \__\ \__\ \_______\ \__\ \__\____\_\  \ \__\\ \__\__/  / /    
#        |\_________\|_______|\|__|\|__|\|_______|\|__|\|__|\_________\|__| \|__|\___/ /     
#        \|_________|                                      \|_________|         \|___|/      
#
# email: s207871@sggw.edu.pl
