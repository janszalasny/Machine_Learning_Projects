### --- Metryki regresji
MAE <- function(y_true, y_pred) {
  mean(abs(y_true - y_pred))
}
MSE <- function(y_true, y_pred) {
  mean((y_true - y_pred)^2)
}
MAPE <- function(y_true, y_pred) {
  mean(abs((y_true - y_pred) / y_true)) * 100
}

### --- Metryki klasyfikacji
AUC <- function(y_true, y_pred) {
  y_true <- as.numeric(y_true)
  y_pred <- as.numeric(y_pred)
  
  order_idx <- order(y_pred, decreasing = TRUE)
  y_true <- y_true[order_idx]
  y_pred <- y_pred[order_idx]
  
  P <- sum(y_true == 1)
  N <- sum(y_true == 0)
  
  TPR <- cumsum(y_true == 1) / P
  FPR <- cumsum(y_true == 0) / N
  
  TPR <- c(0, TPR)
  FPR <- c(0, FPR)
  
  auc <- sum((FPR[-1] - FPR[-length(FPR)]) * 
               (TPR[-1] + TPR[-length(TPR)]) / 2)
  return(auc)
}
MacierzKlasyfikacji <- function(y_true, y_pred, threshold) {
  y_hat <- ifelse(y_pred >= threshold, 1, 0)
  table(Actual = y_true, Predicted = y_hat)
}
Czulosc <- function(TP, FN) {
  TP / (TP + FN)
}
Specyficznosc <- function(TN, FP) {
  TN / (TN + FP)
}
Jakosc <- function(TP, TN, FP, FN) {
  (TP + TN) / (TP + TN + FP + FN)
}
IndexYoudena <- function(roc_obj) {
  
}

### --- ModelOcena
ModelOcena <- function(y_tar, y_hat, save_path = generowane_obiekty, iteration = NULL, dataset_type = "walidacyjny", params = parTune) {
  
  # ---------------------------
  # Funkcja pomocnicza dla tekstu parametrów
  # ---------------------------
  params_text <- if (!is.null(params)) {
    paste0("(", paste(names(params), params, sep = "=", collapse = ", "), ")")
  } else {
    ""
  }
  
  # ---------------------------
  # Sprawdzenie typu zadania
  # ---------------------------
  # 1) Regresja
  if (is.numeric(y_tar)) {
    mae  <- MAE(y_tar, y_hat)
    mse  <- MSE(y_tar, y_hat)
    mape <- MAPE(y_tar, y_hat)
    
    # Tworzymy data.frame do wizualizacji
    df_reg <- data.frame(Actual = y_tar, Predicted = y_hat)
    
    # Wykres: Rzeczywiste vs. Przewidywane
    p_reg <- ggplot(df_reg, aes(x = Actual, y = Predicted)) +
      geom_point(color = "blue") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
      labs(title = paste("Regresja:", params_text),
           x     = "Wartość rzeczywista (y_tar)",
           y     = "Wartość przewidywana (y_hat)") +
      theme_minimal()
    
    # Unikalny identyfikator do nazwy pliku
    unique_id <- if (!is.null(iteration)) {
      paste0("_iter_", iteration)
    } else {
      paste0("_", format(Sys.time(), "%H%M%S"))
    }
    
    # Zapis wykresu regresji
    file_name_reg <- file.path(
      save_path, 
      paste0("regresja_actual_vs_predicted_", dataset_type, unique_id, ".png")
    )
    ggsave(file_name_reg, plot = p_reg, width = 6, height = 6, dpi = 300)
    cat("Wykres regresji (actual vs. predicted) został zapisany w:", file_name_reg, "\n")
    
    return(c(MAE = mae, MSE = mse, MAPE = mape))
    
    # 2) Klasyfikacja (faktor)
  } else if (is.factor(y_tar)) {
    
    # --- 2a) Klasyfikacja binarna
    if (nlevels(y_tar) == 2) {
      # Uwaga: oryginalny kod do binarnej obsługiwał y_tar w postaci 0/1
      #        Zamieniamy factor -> numeric(0/1)
      y_tar_num <- as.numeric(as.character(y_tar))
      y_hat_num <- as.numeric(as.character(y_hat))
      
      auc <- AUC(y_tar_num, y_hat_num)
      
      # Szukanie progu Youdena (binary)
      thresholds <- unique(y_hat_num)
      youden_index <- sapply(thresholds, function(t) {
        y_pred <- ifelse(y_hat_num >= t, 1, 0)
        TP <- sum(y_pred == 1 & y_tar_num == 1)
        FP <- sum(y_pred == 1 & y_tar_num == 0)
        FN <- sum(y_pred == 0 & y_tar_num == 1)
        TN <- sum(y_pred == 0 & y_tar_num == 0)
        
        if ((TP + FN) == 0 || (TN + FP) == 0) {
          return(NA_real_)
        }
        
        sens <- TP / (TP + FN)
        spec <- TN / (TN + FP)
        return(sens + spec - 1)
      })
      
      youden_index <- as.numeric(youden_index)
      best_J <- max(youden_index, na.rm = TRUE)
      best_threshold <- thresholds[which.max(youden_index)]
      y_pred_class <- ifelse(y_hat_num >= best_threshold, 1, 0)
      
      # Tworzymy macierz konfuzji jako factor(0/1)
      mat <- table(
        Actual    = factor(y_tar_num, levels = c(0, 1)), 
        Predicted = factor(y_pred_class, levels = c(0, 1))
      )
      
      TP <- mat[2, 2]
      FP <- mat[1, 2]
      FN <- mat[2, 1]
      TN <- mat[1, 1]
      
      czu   <- TP / (TP + FN)  # czułość
      spec  <- TN / (TN + FP)  # specyficzność
      jakosc <- (TP + TN) / sum(mat)  # Accuracy
      
      # Przygotowanie danych do wizualizacji macierzy
      mat_melt <- melt(mat)
      colnames(mat_melt) <- c("Actual", "Predicted", "Count")
      
      # Tworzenie wykresu macierzy pomyłek
      p <- ggplot(data = mat_melt, aes(x = Predicted, y = Actual, fill = Count)) +
        geom_tile(color = "black") +
        geom_text(aes(label = Count), size = 6) +
        scale_fill_gradient(low = "white", high = "blue") +
        labs(title = paste("Macierz pomyłek", params_text), 
             x = "Przewidywana klasa", 
             y = "Rzeczywista klasa") +
        theme_minimal()
      
      # Unikalny identyfikator do nazw plików
      unique_id <- if (!is.null(iteration)) {
        paste0("_iter_", iteration)
      } else {
        paste0("_", format(Sys.time(), "%H%M%S"))
      }
      
      # Zapis macierzy pomyłek
      file_name_mat <- file.path(save_path, 
                                 paste0("macierz_pomylek_", dataset_type, unique_id, ".png"))
      ggsave(file_name_mat, plot = p, width = 6, height = 6, dpi = 300)
      cat("Macierz pomyłek została zapisana jako obrazek w:", file_name_mat, "\n")
      
      # Rysowanie i zapisywanie krzywej ROC
      roc_obj <- roc(y_tar, y_hat)
      p_roc <- ggplot() +
        geom_line(
          data = data.frame(TPR = roc_obj$sensitivities, 
                            FPR = 1 - roc_obj$specificities), 
          aes(x = FPR, y = TPR), color = "red"
        ) +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
        labs(title = paste("Krzywa ROC", params_text), 
             x = "Fałszywie pozytywny wskaźnik (FPR)", 
             y = "Prawdziwie pozytywny wskaźnik (TPR)") +
        theme_minimal()
      
      # Zapis krzywej ROC
      file_name_roc <- file.path(save_path, 
                                 paste0("krzywa_ROC_", dataset_type, unique_id, ".png"))
      ggsave(file_name_roc, plot = p_roc, width = 6, height = 6, dpi = 300)
      cat("Krzywa ROC została zapisana jako obrazek w:", file_name_roc, "\n")
      
      return(list(
        Mat   = mat,
        J     = best_J,
        Miary = c(AUC = auc, Czułość = czu, Specyficzność = spec, Jakość = jakosc)
      ))
      
      # --- 2b) Klasyfikacja wieloklasowa
    } else {
      # Tutaj y_tar i y_hat to wektory factor (z >= 2 levels).
      # Upewnijmy się, że y_hat jest faktycznie factor z tymi samymi levels:
      if (!is.factor(y_hat)) {
        y_hat <- as.factor(y_hat)
      }
      
      # Bywa, że levels w y_hat mogą być w innej kolejności niż w y_tar:
      wspolne_poziomy <- union(levels(y_tar), levels(y_hat))
      y_tar <- factor(y_tar, levels = wspolne_poziomy)
      y_hat <- factor(y_hat, levels = wspolne_poziomy)
      
      # Macierz konfuzji
      mat <- table(Actual = y_tar, Predicted = y_hat)
      
      # Accuracy
      accuracy <- sum(diag(mat)) / sum(mat)
      
      # Makro-F1
      macro_f1 <- function(conf_mat) {
        num_classes <- nrow(conf_mat)
        f1_vals <- numeric(num_classes)
        for (i in 1:num_classes) {
          TP <- conf_mat[i, i]
          FP <- sum(conf_mat[, i]) - TP
          FN <- sum(conf_mat[i, ]) - TP
          precision <- if ((TP + FP) == 0) 0 else TP / (TP + FP)
          recall    <- if ((TP + FN) == 0) 0 else TP / (TP + FN)
          if ((precision + recall) == 0) {
            f1_vals[i] <- 0
          } else {
            f1_vals[i] <- 2 * precision * recall / (precision + recall)
          }
        }
        return(mean(f1_vals, na.rm = TRUE))
      }
      
      mf1 <- macro_f1(mat)
      
      # ---------------------------
      # Rysowanie macierzy pomyłek (wieloklasowej)
      # ---------------------------
      mat_melt <- melt(mat)
      colnames(mat_melt) <- c("Actual", "Predicted", "Count")
      
      p_multi <- ggplot(data = mat_melt, aes(x = Predicted, y = Actual, fill = Count)) +
        geom_tile(color = "black") +
        geom_text(aes(label = Count), size = 4) +
        scale_fill_gradient(low = "white", high = "blue") +
        labs(title = paste("Macierz pomyłek (wieloklasowa)", params_text),
             x = "Przewidywana klasa", 
             y = "Rzeczywista klasa") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
      
      # Unikalny identyfikator do nazw plików
      unique_id <- if (!is.null(iteration)) {
        paste0("_iter_", iteration)
      } else {
        paste0("_", format(Sys.time(), "%H%M%S"))
      }
      
      # Zapis macierzy pomyłek wieloklasowej
      file_name_multi <- file.path(save_path, 
                                   paste0("macierz_pomylek_wieloklasowa_", dataset_type, unique_id, ".png"))
      ggsave(file_name_multi, plot = p_multi, width = 7, height = 7, dpi = 300)
      cat("Macierz pomyłek (wieloklasowa) została zapisana jako obrazek w:", file_name_multi, "\n")
      
      return(list(
        Mat      = mat,
        Miary    = c(Accuracy = accuracy),
        MacroF1  = mf1
      ))
    }
    
  } else {
    stop("Nieznany typ danych dla y_tar. Oczekiwano: numeric (regresja) lub factor (klasyfikacja).")
  }
}


### --- CrossValidTune
CrossValidTune <- function(dane, kFold = 5, parTune, seed = 123, trainFn, predictFn){
  set.seed(seed)
  n <- nrow(dane)
  
  # 1) Wyodrębnienie targetu i cech
  y_tar <- dane[[1]]
  X <- dane[, -1, drop = FALSE]
  
  # Rozpoznanie typu zadania
  is_regression <- is.numeric(y_tar)
  if (!is_regression && is.factor(y_tar)) {
    n_levels <- nlevels(y_tar)
    is_binary <- (n_levels == 2)
    is_multiclass <- (n_levels > 2)
  } else {
    is_binary <- FALSE
    is_multiclass <- FALSE
  }
  
  # 2) Podziały
  shuffled_idx <- sample(1:n)
  folds <- cut(seq_along(shuffled_idx), breaks = kFold, labels = FALSE)
  
  folds_list <- vector("list", kFold)
  for (i in 1:kFold) {
    fold_vec <- rep(1, n) 
    val_idx <- shuffled_idx[folds == i]
    fold_vec[val_idx] <- 2
    folds_list[[i]] <- fold_vec
  }
  
  # 3) Tworzenie tabeli wynikowej
  nComb <- nrow(parTune)
  tune_expanded <- do.call("rbind", replicate(kFold, parTune, simplify = FALSE))
  tune_expanded$kFold <- rep(1:kFold, each = nComb)
  
  # Dodanie kolumn w zależności od typu
  if (is_regression) {
    metrics_cols <- c("MAEt","MSEt","MAPEt","MAEw","MSEw","MAPEw")
  } else if (is_binary) {
    metrics_cols <- c("AUCT","CzułośćT","SpecyficznośćT","JakośćT",
                      "AUCW","CzułośćW","SpecyficznośćW","JakośćW")
  } else if (is_multiclass) {
    metrics_cols <- c("AccT","AccW")  
  } else {
    stop("Nieobsługiwany typ y_tar w CrossValidTune.")
  }
  
  for (colName in metrics_cols) {
    tune_expanded[[colName]] <- NA
  }
  
  # 4) Pętla po wierszach tune_expanded
  for (row_i in 1:nrow(tune_expanded)) {
    curr_fold <- tune_expanded$kFold[row_i]
    fold_vec <- folds_list[[curr_fold]]
    
    train_idx <- which(fold_vec == 1)
    val_idx   <- which(fold_vec == 2)
    
    X_train <- X[train_idx, , drop=FALSE]
    y_train <- y_tar[train_idx]
    
    X_val   <- X[val_idx, , drop=FALSE]
    y_val   <- y_tar[val_idx]
    
    param_vector <- tune_expanded[row_i, setdiff(names(parTune), "kFold"), drop=FALSE]
    params_list <- as.list(param_vector)
    
    # 4a) Trenowanie modelu
    model <- trainFn(X_train, y_train, params_list)
    
    # 4b) Predykcja
    pred_train <- predictFn(model, X_train)  
    pred_val   <- predictFn(model, X_val)
    
    # 4c) Metryki
    met_train <- ModelOcena(y_train, pred_train, iteration = row_i, dataset_type = "treningowy")
    met_val   <- ModelOcena(y_val,   pred_val,   iteration = row_i, dataset_type = "walidacyjny")
    
    # Zapisywanie do tune_expanded
    if (is_regression) {
      tune_expanded[row_i, "MAEt"]  <- met_train["MAE"]
      tune_expanded[row_i, "MSEt"]  <- met_train["MSE"]
      tune_expanded[row_i, "MAPEt"] <- met_train["MAPE"]
      
      tune_expanded[row_i, "MAEw"]  <- met_val["MAE"]
      tune_expanded[row_i, "MSEw"]  <- met_val["MSE"]
      tune_expanded[row_i, "MAPEw"] <- met_val["MAPE"]
      
    } else if (is_binary) {
      tune_expanded[row_i, "AUCT"]           <- met_train$Miary["AUC"]
      tune_expanded[row_i, "CzułośćT"]       <- met_train$Miary["Czułość"]
      tune_expanded[row_i, "SpecyficznośćT"] <- met_train$Miary["Specyficzność"]
      tune_expanded[row_i, "JakośćT"]        <- met_train$Miary["Jakość"]
      
      tune_expanded[row_i, "AUCW"]           <- met_val$Miary["AUC"]
      tune_expanded[row_i, "CzułośćW"]       <- met_val$Miary["Czułość"]
      tune_expanded[row_i, "SpecyficznośćW"] <- met_val$Miary["Specyficzność"]
      tune_expanded[row_i, "JakośćW"]        <- met_val$Miary["Jakość"]
      
    } else if (is_multiclass) {
      tune_expanded[row_i, "AccT"] <- met_train$Miary["Accuracy"]
      tune_expanded[row_i, "AccW"] <- met_val$Miary["Accuracy"]
      
      tune_expanded[row_i, "MacroF1T"] <- met_train$MacroF1
      tune_expanded[row_i, "MacroF1W"] <- met_val$MacroF1
    }
  }
  
  return(tune_expanded)
}

### --- KNN
KNNtrain <- function(X, y_tar, k, XminNew = 0, XmaxNew = 1) {
  # Walidacja danych wejściowych
  if (any(is.na(X)) || any(is.na(y_tar))) {
    stop("Dane zawierają braki danych.")
  }
  if (!is.data.frame(X) && !is.matrix(X)) {
    stop("X musi być ramką danych lub macierzą.")
  }
  if (k <= 0) {
    stop("Liczba k musi być większa od 0.")
  }
  
  # Sprawdzenie, czy wszystkie zmienne w X są na skali ilorazowej
  if (!all(sapply(X, is.numeric))) {
    stop("Wszystkie zmienne w X muszą być numeryczne (na skali ilorazowej).")
  }
  
  # Normalizacja danych
  X_min <- apply(X, 2, min)
  X_max <- apply(X, 2, max)
  
  X_norm <- as.data.frame(scale(X, center = X_min, scale = X_max - X_min) * 
                            (XmaxNew - XminNew) + XminNew)
  
  # Dodawanie atrybutów
  attr(X_norm, "minOrg") <- X_min
  attr(X_norm, "maxOrg") <- X_max
  attr(X_norm, "minmaxNew") <- c(XminNew, XmaxNew)
  
  # Tworzenie listy modelu
  model <- list(X = X_norm, y = y_tar, k = k)
  
  # Zwrócenie modelu jako lista
  return(model)
}
KNNpred <- function(KNNmodel, X) {
  # Walidacja danych wejściowych
  if (any(is.na(X))) {
    stop("Dane zawierają braki danych.")
  }
  if (!all(colnames(X) %in% colnames(KNNmodel$X))) {
    stop("Niezgodność kolumn między modelem a danymi X.")
  }
  
  # Normalizacja danych na podstawie KNNmodel
  X_min <- attr(KNNmodel$X, "minOrg")
  X_max <- attr(KNNmodel$X, "maxOrg")
  XminNew <- attr(KNNmodel$X, "minmaxNew")[1]
  XmaxNew <- attr(KNNmodel$X, "minmaxNew")[2]
  
  X_norm <- as.data.frame(scale(X, center = X_min, scale = X_max - X_min) * 
                            (XmaxNew - XminNew) + XminNew)
  
  # Rozpoznanie problemu: klasyfikacja czy regresja
  y_tar <- KNNmodel$y
  is_classification <- is.factor(y_tar)
  
  # Obliczanie odległości
  if (is_classification) {
    dist_matrix <- as.matrix(dist(rbind(X_norm, KNNmodel$X), method = "euclidean"))
  } else {
    library(cluster)
    dist_matrix <- daisy(rbind(X_norm, KNNmodel$X), metric = "gower")
    dist_matrix <- as.matrix(dist_matrix)
  }
  
  dist_matrix <- dist_matrix[1:nrow(X_norm), (nrow(X_norm) + 1):nrow(dist_matrix)]
  
  # Predykcja
  if (is_classification) {
    # Klasyfikacja - głosowanie większościowe
    predictions <- apply(dist_matrix, 1, function(row) {
      neighbors <- order(row)[1:KNNmodel$k]
      classes <- KNNmodel$y[neighbors]
      prob <- table(factor(classes, levels = levels(y_tar))) / KNNmodel$k
      pred_class <- names(sort(prob, decreasing = TRUE))[1]
      return(c(prob, Klasa = pred_class))
    })
    
    predictions <- t(predictions)
    predictions <- as.data.frame(predictions)
    colnames(predictions) <- c(levels(y_tar), "Klasa")
    predictions$Klasa <- factor(predictions$Klasa, levels = levels(y_tar))
    
  } else {
    # Regresja - średnia z sąsiadów
    predictions <- apply(dist_matrix, 1, function(row) {
      neighbors <- order(row)[1:KNNmodel$k]
      mean(KNNmodel$y[neighbors])
    })
  }
  
  return(predictions)
}

### --- Drzewa decyzyjne
StopIfNot <- function(Y, X, data, type, depth, minobs, overfit, cf) {
  # Sprawdzamy, czy data to ramka danych
  if (!is.data.frame(data)) {
    cat("Błąd: Dane muszą być ramką danych.\n")
    return(FALSE)
  }
  
  # Sprawdzamy, czy Y i X istnieją w data
  if (!(Y %in% colnames(data))) {
    cat("Błąd: Zmienna Y nie istnieje w danych.\n")
    return(FALSE)
  }
  if (any(!X %in% colnames(data))) {
    cat("Błąd: Jedna lub więcej zmiennych X nie istnieją w danych.\n")
    return(FALSE)
  }
  
  # Sprawdzamy braki danych w Y i X
  if (any(is.na(data[[Y]]))) {
    cat("Błąd: Zmienna Y zawiera braki danych.\n")
    return(FALSE)
  }
  if (any(sapply(X, function(col) any(is.na(data[[col]]))))) {
    cat("Błąd: Jedna lub więcej zmiennych X zawiera braki danych.\n")
    return(FALSE)
  }
  
  # Sprawdzamy wartości depth i minobs
  if (depth <= 0) {
    cat("Błąd: Parametr depth musi być większy od 0.\n")
    return(FALSE)
  }
  if (minobs <= 0) {
    cat("Błąd: Parametr minobs musi być większy od 0.\n")
    return(FALSE)
  }
  
  # Sprawdzamy typ miary
  if (!(type %in% c("Gini", "Entropy", "SS"))) {
    cat("Błąd: Parametr type musi mieć wartość 'Gini', 'Entropy' lub 'SS'.\n")
    return(FALSE)
  }
  
  # Sprawdzamy parametr overfit
  if (!(overfit %in% c("none", "prune"))) {
    cat("Błąd: Parametr overfit musi mieć wartość 'none' lub 'prune'.\n")
    return(FALSE)
  }
  
  # Sprawdzamy parametr cf
  if (!(cf > 0 && cf <= 0.5)) {
    cat("Błąd: Parametr cf musi być w przedziale (0, 0.5].\n")
    return(FALSE)
  }
  
  # Dodatkowa weryfikacja kombinacji parametrów
  if (type == "SS" && is.factor(data[[Y]])) {
    cat("Błąd: Parametr 'type = SS' nie może być użyty, gdy zmienna Y jest faktorem.\n")
    return(FALSE)
  }
  
  # Jeżeli wszystkie warunki są spełnione
  return(TRUE)
}
Tree <- function(Y, X, data, type="Gini", depth=3, minobs=5, overfit="none", cf=0.25) {
  
  # Tworzymy korzeń
  tree <- list()
  tree$Y <- Y
  tree$X <- X
  tree$data <- data
  tree$type <- type
  tree$depth <- 0        # korzeń na poziomie 0
  tree$max_depth <- depth
  tree$minobs <- minobs
  tree$overfit <- overfit
  tree$cf <- cf
  
  # Miara w korzeniu
  if (type == "Gini") {
    p <- table(data[[Y]]) / nrow(data)
    tree$measure <- 1 - sum(p^2)
  } else {

    tree$measure <- 0
  }
  
  # Budowa drzewa
  tree <- BuildTree(tree, data)
  
  # Ewentualne przycięcie
  if (tree$overfit == "prune") {
    tree <- PruneTree(tree, cf=tree$cf)
  }
  
  return(tree)
}
BuildTree <- function(tree, data) {
  # Sprawdzamy głębokość
  if (tree$depth >= tree$max_depth) {
    tree$is_terminal <- TRUE
    tree$prob_1 <- mean(data[[tree$Y]] == "1")
    return(tree)
  }
  
  best_split <- FindBestSplit(
    Y=tree$Y, X=tree$X, data=data,
    parentVal=tree$measure, type=tree$type,
    minobs=tree$minobs
  )
  
  if (is.na(best_split$point) || best_split$infGain <= 0) {
    tree$is_terminal <- TRUE
    tree$prob_1 <- mean(data[[tree$Y]] == "1")
    return(tree)
  }
  
  # Ustawiamy podział
  tree$is_terminal <- FALSE
  tree$split_var   <- best_split$var
  tree$split_point <- best_split$point
  tree$infGain     <- best_split$infGain
  
  # Podzbiory
  left_data  <- data[data[[best_split$var]] <= best_split$point, ]
  right_data <- data[data[[best_split$var]] >  best_split$point, ]
  
  # Lewy węzeł
  left_node <- list()
  left_node$Y <- tree$Y
  left_node$X <- tree$X
  left_node$type <- tree$type
  left_node$depth <- tree$depth + 1
  left_node$max_depth <- tree$max_depth
  left_node$minobs <- tree$minobs
  left_node$overfit <- tree$overfit
  left_node$cf <- tree$cf
  left_node$measure <- best_split$lVal
  
  tree$left <- BuildTree(left_node, left_data)
  
  # Prawy węzeł
  right_node <- list()
  right_node$Y <- tree$Y
  right_node$X <- tree$X
  right_node$type <- tree$type
  right_node$depth <- tree$depth + 1
  right_node$max_depth <- tree$max_depth
  right_node$minobs <- tree$minobs
  right_node$overfit <- tree$overfit
  right_node$cf <- tree$cf
  right_node$measure <- best_split$rVal
  
  tree$right <- BuildTree(right_node, right_data)
  
  return(tree)
}
FindBestSplit <- function(Y, X, data, parentVal, type, minobs) {
  best_split <- data.frame(
    infGain = -Inf, lVal=NA, rVal=NA, point=NA, var=NA, Ln=0, Rn=0
  )
  
  for (var in X) {
    unique_vals <- sort(unique(data[[var]]))
    if (length(unique_vals) < 2) next
    
    for (point in unique_vals[-length(unique_vals)]) {
      left  <- data[data[[var]] <= point, ]
      right <- data[data[[var]] >  point, ]
      
      if (nrow(left) < minobs || nrow(right) < minobs) next
    
      p_left <- table(left[[Y]]) / nrow(left)
      lVal <- 1 - sum(p_left^2)
      
      p_right <- table(right[[Y]]) / nrow(right)
      rVal <- 1 - sum(p_right^2)
      
      infGain <- parentVal - (nrow(left)/nrow(data)*lVal + nrow(right)/nrow(data)*rVal)
      
      if (infGain > best_split$infGain) {
        best_split$infGain <- infGain
        best_split$lVal    <- lVal
        best_split$rVal    <- rVal
        best_split$point   <- point
        best_split$var     <- var
        best_split$Ln      <- nrow(left)
        best_split$Rn      <- nrow(right)
      }
    }
  }
  
  if (best_split$infGain == -Inf) {
    cat("Brak sensownych podziałów dla zmiennych X:", paste(X, collapse = ", "), "\n")
  }
  
  return(best_split)
}
PruneTree <- function(tree, cf) {
  if (tree$is_terminal) return(tree)
  
  if (!is.null(tree$left))  tree$left  <- PruneTree(tree$left,  cf)
  if (!is.null(tree$right)) tree$right <- PruneTree(tree$right, cf)
  
  if (!tree$left$is_terminal || !tree$right$is_terminal) {
    return(tree)
  }

  
  return(tree)
}
PredictTree <- function(tree, data) {
  if (tree$is_terminal) {
    return(rep(tree$prob_1, nrow(data)))
  }

  left_idx  <- data[[tree$split_var]] <= tree$split_point
  right_idx <- !left_idx
  
  pred_left  <- PredictTree(tree$left,  data[left_idx,  , drop=FALSE])
  pred_right <- PredictTree(tree$right, data[right_idx, , drop=FALSE])
  
  predictions <- numeric(nrow(data))
  predictions[left_idx]  <- pred_left
  predictions[right_idx] <- pred_right
  
  return(predictions)
}

### --- Sieci neuronowe
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}
sigmoid_derivative <- function(x) {
  x * (1 - x)
}
softmax <- function(x) {
  exp_x <- exp(x - max(x))
  exp_x / rowSums(exp_x)
}
mse_cost <- function(y, y_pred) {
  mean((y - y_pred)^2)
}
binary_cross_entropy <- function(y, y_pred, eps = 1e-15) {
  y_pred <- pmin(pmax(y_pred, eps), 1 - eps)  
  -mean(y * log(y_pred) + (1 - y) * log(1 - y_pred))
}
categorical_cross_entropy <- function(y, y_pred, eps = 1e-15) {
  y_pred <- pmin(pmax(y_pred, eps), 1 - eps) 
  -mean(rowSums(y * log(y_pred)))
}
initialize_weights <- function(input_size, hidden_sizes, output_size) {
  weights <- list()
  biases <- list()
  
  layer_sizes <- c(input_size, hidden_sizes, output_size)
  
  for (i in 1:(length(layer_sizes) - 1)) {
    # Tworzymy wagi jako macierz typu numeric
    w <- matrix(runif(layer_sizes[i] * layer_sizes[i + 1], -1, 1), 
                nrow = layer_sizes[i], ncol = layer_sizes[i + 1])
    weights[[i]] <- as.matrix(w)
    
    # Biasy też na numeric (wektor):
    b <- runif(layer_sizes[i + 1], -1, 1)
    biases[[i]] <- as.numeric(b)
  }
  
  list(weights = weights, biases = biases)
}
forward_propagation <- function(X, weights, biases, task_type = c("regression","binary","multiclass")) {
  task_type <- match.arg(task_type)
  
  A <- list(as.matrix(X)) 
  
  for (j in 1:(length(weights) - 1)) {
    # --- DIAGNOSTYKA (tutaj bylo duzo problemow)
    # message(sprintf("DEBUG: j = %d", j))
    # message(sprintf("  dim(A[[j]])      = %s", paste(dim(A[[j]]), collapse = " x ")))
    # message(sprintf("  dim(weights[[j]]) = %s", paste(dim(weights[[j]]), collapse = " x ")))
    # message(sprintf("  class(A[[j]])    = %s", class(A[[j]])))
    # message(sprintf("  class(weights[[j]]) = %s", class(weights[[j]])))
    # message(sprintf("  typeof(A[[j]])    = %s", typeof(A[[j]])))
    # message(sprintf("  typeof(weights[[j]]) = %s", typeof(weights[[j]])))
    
    # Rzut na matrix (na wszelki wypadek):
    A[[j]] <- as.matrix(A[[j]], drop = FALSE)
    weights[[j]] <- as.matrix(weights[[j]])  # jeśli nie masz pewności, że to numeric
    
    # --- TERAZ MNOŻENIE ---
    Z <- A[[j]] %*% weights[[j]] + 
      matrix(biases[[j]], nrow = nrow(A[[j]]), ncol = length(biases[[j]]), byrow = TRUE)
    
    A[[j + 1]] <- sigmoid(Z)
  }
  
  # Ostatnia warstwa (bez pętli)
  Z <- A[[length(weights)]] %*% weights[[length(weights)]] +
    matrix(biases[[length(weights)]], 
           nrow = nrow(A[[length(weights)]]), 
           ncol = length(biases[[length(weights)]]), 
           byrow = TRUE)
  
  if (task_type == "regression") {
    A[[length(weights) + 1]] <- Z
  } else if (task_type == "binary") {
    A[[length(weights) + 1]] <- sigmoid(Z)
  } else {
    A[[length(weights) + 1]] <- softmax(Z)
  }
  
  return(A)
}
trainNN <- function(Yname, Xnames, data, hidden_sizes, lr = 0.01, iter = 1000, seed = 123) {
  set.seed(seed)
  
  # Dane wejściowe (macierz numeric)
  X <- as.matrix(data[, Xnames])
  y_raw <- data[[Yname]]
  
  # Rozpoznanie typu problemu
  if (is.numeric(y_raw) && !is.factor(y_raw)) {
    task_type <- "regression"
    y <- as.matrix(y_raw)  # kolumna
    output_size <- 1
  } else if (is.factor(y_raw)) {
    n_levels <- length(levels(y_raw))
    if (n_levels == 2) {
      task_type <- "binary"
      # Zmiana factor na 0/1
      y <- as.numeric(y_raw) - 1
      y <- matrix(y) 
      output_size <- 1
    } else {
      task_type <- "multiclass"
      # One-hot encoding
      y <- model.matrix(~ y_raw - 1)  
      output_size <- ncol(y)
    }
  } else {
    stop("Nieobsługiwany typ zmiennej docelowej.")
  }
  
  # Inicjalizacja wag i biasów
  net <- initialize_weights(input_size = ncol(X), 
                            hidden_sizes = hidden_sizes, 
                            output_size = output_size)
  weights <- net$weights
  biases <- net$biases
  
  cost_history <- numeric(iter)
  
  for (i in seq_len(iter)) {
    # 1. Forward
    A <- forward_propagation(X, weights, biases, task_type)
    A_out <- A[[length(A)]]
    
    # 2. Koszt
    if (task_type == "regression") {
      cost <- mse_cost(y, A_out)
    } else if (task_type == "binary") {
      cost <- binary_cross_entropy(y, A_out)
    } else {  # multiclass
      cost <- categorical_cross_entropy(y, A_out)
    }
    cost_history[i] <- cost
    
    # 3. Backprop
   
    dZ <- A_out - y  # sprawdza się zarówno dla binary, jak i multiclass
    
    # Od końca do początku
    for (j in seq(length(weights), 1, by = -1)) {
      A[[j]] <- as.matrix(A[[j]], drop = FALSE)

      A_j <- as.matrix(A[[j]])        # aktywacje warstwy j
      W_j <- as.matrix(weights[[j]])  # wagi warstwy j
      
      dW <- crossprod(A_j, dZ) / nrow(X)
      db <- colSums(dZ) / nrow(X)
      
      # Aktualizacja
      W_j <- W_j - lr * dW
      b_j <- biases[[j]] - lr * db
      
      # Zapisujemy zmodyfikowane wagi/biasy
      weights[[j]] <- W_j
      biases[[j]] <- b_j
      
      # propagacja błędu do poprzedniej warstwy
      if (j > 1) {
        dZ <- (dZ %*% t(W_j)) * sigmoid_derivative(A_j)
      }
    }
    
    # Podgląd co 100 iteracji
    if (i %% 100 == 0) {
      cat("Iteracja:", i, "Koszt:", round(cost, 6), "\n")
    }
  }
  
  # Wykres kosztu
  plot(seq_len(iter), cost_history, type = "l", col = "blue", lwd = 2,
       xlab = "Iteracje", ylab = "Koszt",
       main = paste("Funkcja kosztu (", task_type, ")", sep=""))
  
  # Zwracamy wytrenowane parametry i info o typie zadania
  list(weights = weights, biases = biases, 
       task_type = task_type,
       cost_history = cost_history)
}
predictNN <- function(model, newdata) {
  weights <- model$weights
  biases <- model$biases
  task_type <- model$task_type
  
  X <- as.matrix(newdata)
  
  # Forward
  A <- forward_propagation(X, weights, biases, task_type)
  A_out <- A[[length(A)]]
  
  if (task_type == "regression") {
    # wartości ciągłe
    return(A_out)
    
  } else if (task_type == "binary") {
    # próg 0.5
    return(ifelse(A_out > 0.5, 1, 0))
    
  } else { # multiclass
    # klasa o najwyższym prawdopodobieństwie
    preds <- apply(A_out, 1, which.max)
    return(preds)
  }
}


### --- Wrappery dla własnych modeli
### --- Klasyfikacja binarna
myTrainFnKNN <- function(X, y, paramsList) {
  k_value <- paramsList$k
  model <- KNNtrain(X, y, k = k_value, XminNew = 0, XmaxNew = 1)
  return(model)
}
myPredictFnKNN <- function(model, Xnew) {
  pred <- KNNpred(model, Xnew)

  if ("Klasa" %in% names(pred)) {
    pred_vector <- as.numeric(as.character(pred$Klasa))
  } else {

    pred_vector <- as.numeric(pred)
  }
  return(pred_vector)
}
myTrainFnTree <- function(X, y, paramsList) {
  depth_value  <- paramsList$depth
  minobs_value <- paramsList$minobs
  overfit_value<- as.character(paramsList$overfit) 
  cf_value     <- paramsList$cf
  
 
  data_train <- data.frame(
    Y = factor(y, levels = c(0,1)),
    X,
    check.names=FALSE
  )
  
  model <- Tree(
    Y      = "Y",
    X      = colnames(data_train)[-1],
    data   = data_train,
    type   = "Gini",
    depth  = depth_value,
    minobs = minobs_value,
    overfit= overfit_value,
    cf     = cf_value
  )
  return(model)
}
myPredictFnTree <- function(model, Xnew) {

  data_test <- data.frame(Y=rep(NA, nrow(Xnew)), Xnew, check.names=FALSE)
  
  pred_raw <- PredictTree(model, data_test)  # numeric
  
  pred_num <- as.numeric(pred_raw)
  
  return(pred_num)
}
myTrainFnNN <- function(X, y, paramsList) {
  
  
  hsize <- paramsList$hsize
  lr    <- paramsList$lr
  iter  <- paramsList$iter
  seed  <- paramsList$seed
  

  dataNN <- data.frame(
    Y = ifelse(y == "1", 1, 0),
    X,
    check.names = FALSE
  )
  
  net <- trainNN(
    Yname  = "Y",
    Xnames = colnames(X),
    data   = dataNN,
    h      = c(hsize),  
    lr     = lr,
    iter   = iter,
    seed   = seed
  )
  
  return(net)
}
myPredictFnNN <- function(model, Xnew) {

  
  pred_raw <- predictNN(
    model,
    as.matrix(Xnew)  
  )
  

  pred_vector <- as.numeric(pred_raw)  
  
  return(pred_vector)
}

### --- Regresja
myTrainFnKNNreg <- function(X, y, paramsList) {

  k_value <- paramsList$k
  
  KNNmodel <- KNNtrain(
    X = X,
    y_tar = y,
    k = k_value,
    XminNew = 0,
    XmaxNew = 1
  )
  
  return(KNNmodel)
}
myPredictFnKNNreg <- function(model, Xnew) {
  predictions <- KNNpred(model, Xnew)
  if (is.data.frame(predictions)) {
    pred_vector <- as.numeric(predictions[[1]])
  } else {
    pred_vector <- as.numeric(predictions)
  }
  
  return(pred_vector)
}
myTrainFnTreeReg <- function(X, y, paramsList) {
  depth_value   <- paramsList$depth
  minobs_value  <- paramsList$minobs
  overfit_value <- as.character(paramsList$overfit)  
  cf_value      <- paramsList$cf
  
  data_train <- data.frame(
    Y = y, 
    X,
    check.names = FALSE
  )
  
  tree_model <- Tree(
    Y       = "Y",                       
    X       = colnames(data_train)[-1], 
    data    = data_train,
    type    = "SS",           
    depth   = depth_value,
    minobs  = minobs_value,
    overfit = overfit_value,
    cf      = cf_value
  )

  return(tree_model)
}
myPredictFnTreeReg <- function(model, Xnew) {
  data_test <- data.frame(
    Y = rep(NA, nrow(Xnew)),  
    Xnew,
    check.names = FALSE
  )
  
  pred_raw <- PredictTree(model, data_test)
  
  pred_vector <- as.numeric(pred_raw)
  
  return(pred_vector)
}
myTrainFnNNReg <- function(X, y, paramsList) {
  h1_value   <- paramsList$h1
  h2_value   <- paramsList$h2
  lr_value   <- paramsList$lr
  iter_value <- paramsList$iter
  seed_value <- paramsList$seed
  
  data_train <- data.frame(
    Y = y, 
    X,
    check.names = FALSE
  )
  
  net <- trainNN(
    Yname  = "Y",
    Xnames = colnames(X),
    data   = data_train,
    h      = c(h1_value, h2_value),  # przykładowo 2 warstwy
    lr     = lr_value,
    iter   = iter_value,
    seed   = seed_value
  )
  
  return(net)
}
myPredictFnNNReg <- function(model, Xnew) {
  pred_raw <- predictNN(model, as.matrix(Xnew))
  
  pred_vector <- as.numeric(pred_raw)
  
  return(pred_vector)
}

### --- Klasyfikacja wieloklasowa
myTrainFnTreeMulti <- function(X, y, paramsList) {
  depth_value  <- paramsList$depth
  minobs_value <- paramsList$minobs
  overfit_value <- as.character(paramsList$overfit)
  cf_value     <- paramsList$cf

  data_train <- data.frame(
    Y = factor(y),  
    X,
    check.names = FALSE
  )
  
    model <- Tree(
    Y      = "Y",
    X      = colnames(data_train)[-1],
    data   = data_train,
    type   = "Gini", 
    depth  = depth_value,
    minobs = minobs_value,
    overfit= overfit_value,
    cf     = cf_value
  )
  
  return(model)
}
myPredictFnTreeMulti <- function(model, Xnew) {
  data_test <- data.frame(Y = rep(NA, nrow(Xnew)), Xnew, check.names = FALSE)
  
  pred_probs <- PredictTree(model, data_test)
  
  print("Output PredictTree:")
  print(pred_probs)
  
  if (is.null(pred_probs)) {
    stop("PredictTree zwrócił NULL.")
  }
  
  if (any(is.na(pred_probs))) {
    stop("PredictTree zwrócił wartości NA.")
  }
  
  if (!is.matrix(pred_probs)) {
    stop("PredictTree nie zwrócił macierzy.")
  }
  
  pred_class <- apply(pred_probs, 1, function(row) {
    colnames(pred_probs)[which.max(row)]
  })
  
  pred_factor <- factor(pred_class, levels = levels(model$data$Y))
  
  return(pred_factor)
}
myTrainFnNNMulti <- function(X, y, paramsList) {
  if (!is.factor(y)) {
    stop("Target 'y' musi być typu factor dla klasyfikacji wieloklasowej.")
  }

  training_data <- as.data.frame(X, check.names = FALSE)
  training_data$Target <- y

  
  model <- trainNN(
    Yname        = "Target",
    Xnames       = colnames(X),
    data         = training_data,
    hidden_sizes = if (is.null(paramsList$size)) { 
      10  
    } else {
      if (length(paramsList$size) == 1) {
        c(paramsList$size)
      } else {
        paramsList$size
      }
    },
    lr           = if (!is.null(paramsList$lr)) paramsList$lr else 0.01,
    iter         = if (!is.null(paramsList$maxit)) paramsList$maxit else 1000,
    seed         = if (!is.null(paramsList$seed)) paramsList$seed else 123
  )
  
  # Zachowujemy levels, żeby potem poprawnie przemapować numery klas na etykiety
  model$levels <- levels(y)
  
  return(model)
}
myPredictFnNNMulti <- function(model, Xnew) {
  # Wywołujemy Twoją funkcję predictNN
  preds_num <- predictNN(model, Xnew)

  levels_vec <- model$levels
  predictions <- factor(levels_vec[preds_num], levels = levels_vec)
  
  return(predictions)
}


### --- Wbudowane paczki
### --- Klasyfikacja binarna 
myTrainFnKNNclass <- function(X, y, paramsList) {
  model <- list(
    X_train = as.matrix(X),
    y_train = y,
    k = paramsList$k
  )
  
  return(model)
}
myPredictFnKNNclass <- function(model, Xnew) {
  pred <- knn(
    train = model$X_train,
    test = as.matrix(Xnew),
    cl = model$y_train,
    k = model$k,
    prob = TRUE
  )
  
  probs <- attr(pred, "prob")
  
  probs[pred == "0"] <- 1 - probs[pred == "0"]
  
  return(probs)
}
myTrainFnTreeRpart <- function(X, y, paramsList) {
  training_data <- data.frame(
    target = factor(y, levels = c("0", "1")),
    X,
    check.names = FALSE
  )

  cp_value <- if(paramsList$overfit == "none") 0.00001 else paramsList$cf

  model <- rpart(
    target ~ .,
    data = training_data,
    method = "class",
    control = rpart.control(
      maxdepth = paramsList$depth,
      minsplit = paramsList$minobs * 2, 
      minbucket = paramsList$minobs,    
      cp = cp_value               
    )
  )
  
  if(paramsList$overfit == "prune" && paramsList$cf > 0) {
    model <- prune(model, cp = paramsList$cf)
  }
  
  return(model)
}
myPredictFnTreeRpart <- function(model, Xnew) {
  pred_probs <- predict(model, Xnew, type = "prob")[,2]
  return(pred_probs)
}
myTrainFnNNet <- function(X, y, paramsList) {
  if (is.factor(y)) {
    y <- as.numeric(as.character(y))
  }

  train_data <- cbind(y = y, X)
  
  size <- paramsList$size       
  decay <- paramsList$decay     
  maxit <- paramsList$maxit 

  model <- nnet(y ~ ., 
                data = train_data,
                size = size,
                decay = decay,
                maxit = maxit,
                entropy = TRUE,  
                linout = FALSE,  
                trace = FALSE)
  
  return(model)
}
myPredictFnNNet <- function(model, Xnew) {
  pred_raw <- predict(model, newdata = Xnew, type = "raw")
  
  if (model$entropy) {
    return(as.numeric(pred_raw))
  } else {
    return(pred_raw)
  }
}

### --- Regresja
myTrainFnKKNN <- function(X, y, paramsList) {
  model <- list(
    X = X,
    y = y,
    k = paramsList$k,
    scale = TRUE
  )
  
  return(model)
}
myPredictFnKKNN <- function(model, Xnew) {
  train_data <- data.frame(y = model$y, model$X)
  
  test_data <- data.frame(y = rep(NA, nrow(Xnew)), Xnew)
  
  pred <- kknn(
    formula = y ~ ., 
    train = train_data,
    test = test_data,
    k = model$k,
    scale = model$scale
  )
  
  return(pred$fitted.values)
}
myTrainFnRpartReg <- function(X, y, paramsList) {
  data_train <- data.frame(Y = y, X, check.names = FALSE)
  formula <- as.formula("Y ~ .")
  model <- rpart(
    formula = formula,
    data    = data_train,
    method  = "anova", 
    control = rpart.control(
      maxdepth = paramsList$maxdepth,
      minsplit = paramsList$minsplit,
      cp       = paramsList$cp
    )
  )
  
  return(model)
}
myPredictFnRpartReg <- function(model, Xnew) {
  data_test <- data.frame(Xnew, check.names = FALSE)

  predictions <- predict(model, newdata = data_test)
  
  return(predictions)
} 
myTrainFnNnetReg <- function(X, y, paramsList) {
  data_train <- data.frame(Y = y, X, check.names = FALSE)
  
  formula <- as.formula("Y ~ .")

  set.seed(paramsList$seed)

  nn_model <- nnet(
    formula     = formula,
    data        = data_train,
    size        = paramsList$h1,  
    decay       = paramsList$decay, 
    linout      = TRUE,      
    maxit       = paramsList$iter,   
    trace       = FALSE 
  )
  
  return(nn_model)
}
myPredictFnNnetReg <- function(model, Xnew) {
  data_test <- data.frame(Xnew, check.names = FALSE)
  
  predictions <- predict(model, newdata = data_test, type = "raw")
  
  return(predictions)
}

### --- Klasyfikacja wieloklasowa
myTrainFnKNNmultiKnn <- function(X, y, paramsList) {
  model <- list(
    X_train = as.matrix(X),
    y_train = y,
    k = paramsList$k
  )
  return(model)
}
myPredictFnKNNmultiKnn <- function(model, Xnew) {
  pred <- knn(
    train = model$X_train,
    test = as.matrix(Xnew),
    cl = model$y_train,
    k = model$k
  )
  return(pred)  # Zwraca factor z poziomami klas
}
myTrainFnTreeMultiRpart <- function(X, y, paramsList) {
  training_data <- data.frame(
    Target = y, 
    X,
    check.names = FALSE
  )
  
  model <- rpart(
    Target ~ ., 
    data = training_data, 
    method = "class",  
    control = rpart.control(
      maxdepth = paramsList$depth,  
      minsplit = paramsList$minobs * 2, 
      minbucket = paramsList$minobs,
      cp = if (paramsList$overfit == "prune") paramsList$cf else 0.00001 
    )
  )
  
  return(model)
}
myPredictFnTreeMultiRpart <- function(model, Xnew) {
  predictions <- predict(model, newdata = Xnew, type = "class")
  return(predictions)
}
myTrainFnNNMultiNnet <- function(X, y, paramsList) {

  training_data <- data.frame(
    Target = y,  
    X,
    check.names = FALSE
  )
  
  
  model <- nnet(
    Target ~ ., 
    data = training_data,
    size = paramsList$size, 
    rang = 0.1,
    decay = paramsList$decay,  
    maxit = paramsList$maxit, 
    trace = FALSE 
  )
  
  return(model)
}
myPredictFnNNMultiNnet <- function(model, Xnew) {
  predictions <- predict(model, newdata = Xnew, type = "class")
  return(predictions)
}

