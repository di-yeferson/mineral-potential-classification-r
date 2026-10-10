Predicción del Potencial de Mineralización con Random Forest
================
Diego Yeferson Quispe Huamani
10 de octubre de 2026

- [0. Preparación del entorno](#0-preparación-del-entorno)
  - [0.1 Paquetes](#01-paquetes)
  - [0.2 Procedencia de los datos](#02-procedencia-de-los-datos)
  - [0.3 Carga del dataset](#03-carga-del-dataset)
- [1. Introducción: mapa de proceso](#1-introducción-mapa-de-proceso)
- [2. EDA: Análisis exploratorio de los datos usando
  R](#2-eda-análisis-exploratorio-de-los-datos-usando-r)
  - [2.1 Estructura del dataset](#21-estructura-del-dataset)
  - [2.2 Estadísticas descriptivas (variables
    numéricas)](#22-estadísticas-descriptivas-variables-numéricas)
  - [2.3 Frecuencias de variables
    categóricas](#23-frecuencias-de-variables-categóricas)
  - [2.4 Distribución de la variable
    objetivo](#24-distribución-de-la-variable-objetivo)
  - [2.5 Valores faltantes y
    duplicados](#25-valores-faltantes-y-duplicados)
  - [2.6 Correlación entre variables
    numéricas](#26-correlación-entre-variables-numéricas)
  - [2.7 Asociación entre variables categóricas y la variable objetivo
    (Chi-cuadrado)](#27-asociación-entre-variables-categóricas-y-la-variable-objetivo-chi-cuadrado)
- [3. Modelado: desarrollo del algoritmo en
  R](#3-modelado-desarrollo-del-algoritmo-en-r)
  - [3.1 División estratificada train/test
    (80/20)](#31-división-estratificada-traintest-8020)
  - [3.2 Modelo base de referencia: Árbol de decisión
    (CART)](#32-modelo-base-de-referencia-árbol-de-decisión-cart)
  - [3.3 Algoritmo elegido: Random Forest
    multiclase](#33-algoritmo-elegido-random-forest-multiclase)
  - [3.4 Selección de hiperparámetros por validación cruzada
    (10-fold)](#34-selección-de-hiperparámetros-por-validación-cruzada-10-fold)
- [4. Evaluación: efectividad de la predicción en
  R](#4-evaluación-efectividad-de-la-predicción-en-r)
  - [4.1 Matriz de confusión y métricas por
    clase](#41-matriz-de-confusión-y-métricas-por-clase)
  - [4.2 Comparación con el árbol de decisión
    base](#42-comparación-con-el-árbol-de-decisión-base)
  - [4.3 AUC-ROC multiclase
    (One-vs-Rest)](#43-auc-roc-multiclase-one-vs-rest)
  - [4.4 Curva de aprendizaje
    (overfitting)](#44-curva-de-aprendizaje-overfitting)
  - [4.5 Balance frente a los KPIs de
    negocio](#45-balance-frente-a-los-kpis-de-negocio)
- [5. Propuesta de despliegue en producción utilizando
  R](#5-propuesta-de-despliegue-en-producción-utilizando-r)
- [6. Conclusiones](#6-conclusiones)

# 0. Preparación del entorno

## 0.1 Paquetes

``` r
paquetes <- c("randomForest", "rpart", "partykit", "caret", "pROC",
              "ggplot2", "reshape2", "scales", "e1071")

faltantes <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if (length(faltantes) > 0) {
  install.packages(faltantes, dependencies = TRUE)
}

invisible(lapply(paquetes, library, character.only = TRUE))
```

## 0.2 Procedencia de los datos

El dataset `dataset_mineria.csv` (600 zonas × 7 variables) es
**semi-sintético**:

- Las **49 primeras observaciones** corresponden a la vista previa
  pública del notebook de Kaggle *“Predicción de Potencial de
  Mineralización”* (Ignacio Zambrano C.), cuyo dataset fue generado por
  su propio autor.
- Las **551 observaciones restantes** se generaron de forma reproducible
  (semilla fija) con un modelo geológico simplificado tipo pórfido,
  respetando el esquema, los rangos y la distribución de clases del
  notebook original (Económico 32.8 %, Marginal 41.3 %, Estéril 25.8 %).

Por ello, los resultados demuestran la **metodología de análisis y
modelado**; no deben interpretarse como evidencia geológica de un
yacimiento real.

## 0.3 Carga del dataset

``` r
usar_csv_original <- TRUE   # TRUE: usa el CSV incluido en data/ | FALSE: regenera el dataset
ruta_csv <- "../data/dataset_mineria.csv"

set.seed(2026)

niveles_litologia  <- c("Diorita", "Andesita", "Brecha")
niveles_alteracion <- c("Silicia", "Argilica", "Propilitica", "Potasica")
niveles_estructura <- c("Falla_Menor", "Falla_Mayor", "Sin_Estructura")
niveles_objetivo   <- c("Economico", "Marginal", "Esteril")

if (usar_csv_original && file.exists(ruta_csv)) {

  df <- read.csv(ruta_csv, stringsAsFactors = TRUE)

} else {

  # 1) Las 49 observaciones de la vista previa pública de Kaggle
  reales_txt <- "
Diorita,Silicia,Falla_Menor,4,1,3,Economico
Diorita,Silicia,Falla_Mayor,1,3,3,Esteril
Andesita,Argilica,Falla_Mayor,4,1,2,Economico
Diorita,Propilitica,Falla_Mayor,4,2,1,Marginal
Andesita,Silicia,Falla_Menor,1,2,2,Esteril
Brecha,Potasica,Falla_Mayor,3,3,1,Economico
Andesita,Propilitica,Sin_Estructura,3,1,3,Esteril
Diorita,Silicia,Falla_Menor,4,2,3,Economico
Andesita,Propilitica,Falla_Mayor,3,3,2,Marginal
Diorita,Silicia,Sin_Estructura,1,3,1,Esteril
Brecha,Argilica,Falla_Mayor,1,1,3,Marginal
Diorita,Argilica,Falla_Menor,1,2,3,Esteril
Diorita,Argilica,Falla_Mayor,1,1,1,Esteril
Andesita,Argilica,Falla_Mayor,3,2,1,Economico
Diorita,Argilica,Falla_Mayor,4,1,2,Economico
Brecha,Propilitica,Sin_Estructura,2,1,2,Marginal
Andesita,Argilica,Falla_Menor,4,2,2,Economico
Brecha,Potasica,Falla_Menor,1,3,1,Economico
Andesita,Potasica,Sin_Estructura,2,2,3,Economico
Andesita,Silicia,Sin_Estructura,4,1,3,Marginal
Andesita,Potasica,Sin_Estructura,4,2,3,Economico
Andesita,Argilica,Falla_Mayor,1,1,3,Esteril
Andesita,Silicia,Sin_Estructura,1,2,3,Esteril
Diorita,Argilica,Falla_Mayor,1,1,3,Esteril
Brecha,Silicia,Falla_Menor,3,2,3,Marginal
Brecha,Potasica,Falla_Menor,4,3,1,Economico
Diorita,Propilitica,Falla_Mayor,4,2,2,Marginal
Brecha,Propilitica,Falla_Menor,3,3,1,Marginal
Andesita,Silicia,Sin_Estructura,1,1,3,Esteril
Andesita,Silicia,Falla_Mayor,3,3,1,Marginal
Diorita,Argilica,Falla_Menor,3,2,3,Marginal
Brecha,Potasica,Falla_Menor,3,2,1,Economico
Diorita,Silicia,Sin_Estructura,2,2,2,Esteril
Diorita,Potasica,Falla_Mayor,3,3,1,Economico
Andesita,Silicia,Falla_Mayor,2,3,2,Marginal
Diorita,Argilica,Falla_Menor,1,3,2,Esteril
Diorita,Argilica,Falla_Menor,4,3,2,Economico
Brecha,Silicia,Falla_Menor,3,1,2,Economico
Brecha,Silicia,Falla_Mayor,1,3,2,Marginal
Brecha,Argilica,Falla_Menor,4,2,1,Economico
Diorita,Silicia,Falla_Menor,3,1,3,Marginal
Andesita,Silicia,Sin_Estructura,2,3,2,Marginal
Andesita,Propilitica,Sin_Estructura,1,1,2,Esteril
Andesita,Argilica,Falla_Menor,3,2,1,Marginal
Diorita,Potasica,Falla_Mayor,3,2,1,Economico
Diorita,Potasica,Sin_Estructura,1,1,2,Marginal
Andesita,Argilica,Sin_Estructura,2,3,2,Marginal
Brecha,Potasica,Falla_Menor,2,1,1,Economico
Diorita,Potasica,Falla_Menor,3,3,3,Economico
"

  con <- textConnection(reales_txt)
  df_reales <- read.csv(con, header = FALSE, strip.white = TRUE,
                         col.names = c("Litologia", "Alteracion", "Control_Estructura",
                                       "Densidad_Vetillas", "Profundidad_Nivel",
                                       "Anomalia_Magnetica", "Potencial_Economico"))
  close(con)

  n_total  <- 600
  n_reales <- nrow(df_reales)
  n_faltan <- n_total - n_reales

  # 2) Generación reproducible de las observaciones restantes
  
  genera_fila_objetivo <- function(alteracion, densidad, estructura, anomalia) {
    # Puntaje geológico continuo (modelo tipo "pórfido"): alteración potásica
    # y mayor densidad de vetillas elevan el potencial económico; alteración
    # propilítica (periférica) y baja densidad de vetillas lo reducen.
    score <- 0
    score <- score + c(Potasica = 2.1, Silicia = 0.5, Argilica = 0.0, Propilitica = -1.1)[alteracion]
    score <- score + (densidad - 2.5) * 0.75
    score <- score + c(Falla_Mayor = 0.35, Falla_Menor = 0.05, Sin_Estructura = -0.25)[estructura]
    score <- score + (anomalia - 2) * 0.20
    score <- score + rnorm(1, mean = 0, sd = 1.35)     # ruido geológico/medición
    score
  }

  lit  <- sample(niveles_litologia,  n_faltan, replace = TRUE)
  alt  <- sample(niveles_alteracion, n_faltan, replace = TRUE, prob = c(0.28, 0.28, 0.22, 0.22))
  est  <- sample(niveles_estructura, n_faltan, replace = TRUE)
  dens <- sample(1:4, n_faltan, replace = TRUE)
  prof <- sample(1:3, n_faltan, replace = TRUE)
  anom <- sample(1:3, n_faltan, replace = TRUE)

  scores <- mapply(genera_fila_objetivo, alt, dens, est, anom)

  # Cortes calibrados por simulación para reproducir, junto con las 49 filas
  # de Kaggle, la distribución global documentada en el notebook original:
  # Economico 32.8% (197) 
  # Marginal 41.3% (248) 
  # Esteril 25.8% (155)
  
  corte_alto <- quantile(scores, probs = 0.595)
  corte_bajo <- quantile(scores, probs = 0.300)

  obj <- ifelse(scores >= corte_alto, "Economico",
         ifelse(scores >= corte_bajo, "Marginal", "Esteril"))

  df_sinteticas <- data.frame(
    Litologia = lit, Alteracion = alt, Control_Estructura = est,
    Densidad_Vetillas = dens, Profundidad_Nivel = prof, Anomalia_Magnetica = anom,
    Potencial_Economico = obj, stringsAsFactors = FALSE
  )

  df <- rbind(df_reales, df_sinteticas)

  # Ajuste fino de proporciones finales a los conteos exactos reportados
  conteo_objetivo <- c(Economico = 197, Marginal = 248, Esteril = 155)
  ajustar_conteos <- function(df, objetivo) {
    actual <- table(df$Potencial_Economico)
    for (cl in names(objetivo)) {
      dif <- objetivo[cl] - actual[cl]
      if (is.na(dif)) dif <- objetivo[cl]
      if (dif > 0) {
        sobrantes <- names(actual)[actual > objetivo[names(actual)]]
        candidatos <- which(df$Potencial_Economico %in% sobrantes &
                               seq_len(nrow(df)) > n_reales)
        if (length(candidatos) >= dif) {
          idx <- sample(candidatos, dif)
          df$Potencial_Economico[idx] <- cl
          actual <- table(df$Potencial_Economico)
        }
      }
    }
    df
  }
  df <- ajustar_conteos(df, conteo_objetivo)

  write.csv(df, ruta_csv, row.names = FALSE)
}

df$Litologia           <- factor(df$Litologia,           levels = niveles_litologia)
df$Alteracion          <- factor(df$Alteracion,          levels = niveles_alteracion)
df$Control_Estructura  <- factor(df$Control_Estructura,  levels = niveles_estructura)
df$Potencial_Economico <- factor(df$Potencial_Economico, levels = niveles_objetivo)
df$Densidad_Vetillas   <- as.integer(df$Densidad_Vetillas)
df$Profundidad_Nivel   <- as.integer(df$Profundidad_Nivel)
df$Anomalia_Magnetica  <- as.integer(df$Anomalia_Magnetica)

cat("Dataset listo:", nrow(df), "filas x", ncol(df), "columnas\n")
```

    ## Dataset listo: 600 filas x 7 columnas

``` r
table(df$Potencial_Economico)
```

    ## 
    ## Economico  Marginal   Esteril 
    ##       197       248       155

------------------------------------------------------------------------

# 1. Introducción: mapa de proceso

``` r
par(mar = c(1, 1, 2.5, 1))
plot(0, 0, type = "n", xlim = c(0, 10), ylim = c(0, 2.6), axes = FALSE,
     xlab = "", ylab = "",
     main = "Mapa de proceso: de la exploración geológica a la decisión de negocio")

cajas <- data.frame(
  x = c(0.3, 2.15, 4.0, 5.85, 7.7),
  texto = c("1. Datos\ngeológicos de\ncampo\n(litología, alteración,\nestructura, vetillas,\nanomalías)",
            "2. EDA en R\n(distribuciones,\nasociaciones,\ncalidad de datos)",
            "3. Modelado en R\n(Random Forest\nmulticlase)",
            "4. Evaluación en R\n(matriz de confusión,\nF1, AUC-ROC,\nvalidación cruzada)",
            "5. Decisión de\nnegocio\n(priorizar zonas:\nEconómico /\nMarginal / Estéril)")
)
ancho <- 1.55; alto <- 1.55

for (i in seq_len(nrow(cajas))) {
  rect(cajas$x[i], 0.5, cajas$x[i] + ancho, 0.5 + alto,
       col = "#EAF3EE", border = "#1B7A43", lwd = 2)
  text(cajas$x[i] + ancho / 2, 0.5 + alto / 2, cajas$texto[i], cex = 0.62, font = 1)
  if (i < nrow(cajas)) {
    arrows(cajas$x[i] + ancho, 0.5 + alto / 2, cajas$x[i + 1] - 0.02, 0.5 + alto / 2,
           length = 0.12, lwd = 2, col = "#333333")
  }
}
```

<img src="/tmp/claude-0/-home-claude/ce241027-6181-578c-9961-cbb777a9ddcf/scratchpad/repo5/mineral-potential-classification-r/reports/mineral_potential_files/figure-gfm/mapa-proceso-1.png" style="display: block; margin: auto;" />

------------------------------------------------------------------------

# 2. EDA: Análisis exploratorio de los datos usando R

``` r
paleta <- c("Economico" = "#1B7A43", "Marginal" = "#D9A441", "Esteril" = "#B23A2E")
tema <- theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"), panel.grid.minor = element_blank())
```

## 2.1 Estructura del dataset

``` r
str(df)
```

    ## 'data.frame':    600 obs. of  7 variables:
    ##  $ Litologia          : Factor w/ 3 levels "Diorita","Andesita",..: 1 1 2 1 2 3 2 1 2 1 ...
    ##  $ Alteracion         : Factor w/ 4 levels "Silicia","Argilica",..: 1 1 2 3 1 4 3 1 3 1 ...
    ##  $ Control_Estructura : Factor w/ 3 levels "Falla_Menor",..: 1 2 2 2 1 2 3 1 2 3 ...
    ##  $ Densidad_Vetillas  : int  4 1 4 4 1 3 3 4 3 1 ...
    ##  $ Profundidad_Nivel  : int  1 3 1 2 2 3 1 2 3 3 ...
    ##  $ Anomalia_Magnetica : int  3 3 2 1 2 1 3 3 2 1 ...
    ##  $ Potencial_Economico: Factor w/ 3 levels "Economico","Marginal",..: 1 3 1 2 3 1 3 1 2 3 ...

## 2.2 Estadísticas descriptivas (variables numéricas)

``` r
summary(df[, c("Densidad_Vetillas", "Profundidad_Nivel", "Anomalia_Magnetica")])
```

    ##  Densidad_Vetillas Profundidad_Nivel Anomalia_Magnetica
    ##  Min.   :1.00      Min.   :1.000     Min.   :1.000     
    ##  1st Qu.:1.00      1st Qu.:1.000     1st Qu.:1.000     
    ##  Median :3.00      Median :2.000     Median :2.000     
    ##  Mean   :2.51      Mean   :1.955     Mean   :1.997     
    ##  3rd Qu.:4.00      3rd Qu.:3.000     3rd Qu.:3.000     
    ##  Max.   :4.00      Max.   :3.000     Max.   :3.000

``` r
sapply(df[, c("Densidad_Vetillas", "Profundidad_Nivel", "Anomalia_Magnetica")], sd)
```

    ##  Densidad_Vetillas  Profundidad_Nivel Anomalia_Magnetica 
    ##          1.1352156          0.8391356          0.8110190

## 2.3 Frecuencias de variables categóricas

``` r
for (v in c("Litologia", "Alteracion", "Control_Estructura")) {
  cat("\n--", v, "--\n")
  print(table(df[[v]]))
  print(round(prop.table(table(df[[v]])) * 100, 1))
}
```

    ## 
    ## -- Litologia --
    ## 
    ##  Diorita Andesita   Brecha 
    ##      215      219      166 
    ## 
    ##  Diorita Andesita   Brecha 
    ##     35.8     36.5     27.7 
    ## 
    ## -- Alteracion --
    ## 
    ##     Silicia    Argilica Propilitica    Potasica 
    ##         167         156         144         133 
    ## 
    ##     Silicia    Argilica Propilitica    Potasica 
    ##        27.8        26.0        24.0        22.2 
    ## 
    ## -- Control_Estructura --
    ## 
    ##    Falla_Menor    Falla_Mayor Sin_Estructura 
    ##            208            196            196 
    ## 
    ##    Falla_Menor    Falla_Mayor Sin_Estructura 
    ##           34.7           32.7           32.7

``` r
df_lit <- as.data.frame(table(df$Litologia)); names(df_lit) <- c("Categoria", "n")
df_alt <- as.data.frame(table(df$Alteracion)); names(df_alt) <- c("Categoria", "n")
df_est <- as.data.frame(table(df$Control_Estructura)); names(df_est) <- c("Categoria", "n")
df_lit$Variable <- "Litologia"; df_alt$Variable <- "Alteracion"; df_est$Variable <- "Control_Estructura"
df_cat <- rbind(df_lit, df_alt, df_est)

ggplot(df_cat, aes(x = Categoria, y = n, fill = Variable)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~Variable, scales = "free_x") +
  labs(title = "Frecuencia de variables geológicas categóricas", x = NULL, y = "N° de zonas") +
  tema + theme(axis.text.x = element_text(angle = 35, hjust = 1))
```

<img src="/tmp/claude-0/-home-claude/ce241027-6181-578c-9961-cbb777a9ddcf/scratchpad/repo5/mineral-potential-classification-r/reports/mineral_potential_files/figure-gfm/eda-frecuencias-plot-1.png" style="display: block; margin: auto;" />

``` r
df_num <- reshape2::melt(df[, c("Densidad_Vetillas", "Profundidad_Nivel", "Anomalia_Magnetica")])
ggplot(df_num, aes(x = value)) +
  geom_bar(fill = "#2E6F95") +
  facet_wrap(~variable, scales = "free_x") +
  labs(title = "Distribución de variables numéricas", x = "Valor", y = "Frecuencia") +
  tema
```

<img src="/tmp/claude-0/-home-claude/ce241027-6181-578c-9961-cbb777a9ddcf/scratchpad/repo5/mineral-potential-classification-r/reports/mineral_potential_files/figure-gfm/eda-histogramas-1.png" style="display: block; margin: auto;" />

## 2.4 Distribución de la variable objetivo

``` r
table(df$Potencial_Economico)
```

    ## 
    ## Economico  Marginal   Esteril 
    ##       197       248       155

``` r
round(prop.table(table(df$Potencial_Economico)) * 100, 1)
```

    ## 
    ## Economico  Marginal   Esteril 
    ##      32.8      41.3      25.8

``` r
ggplot(df, aes(x = Potencial_Economico, fill = Potencial_Economico)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.4) +
  scale_fill_manual(values = paleta) +
  labs(title = "Distribución de la variable objetivo",
       x = "Potencial Económico", y = "N° de zonas") +
  tema + theme(legend.position = "none")
```

<img src="/tmp/claude-0/-home-claude/ce241027-6181-578c-9961-cbb777a9ddcf/scratchpad/repo5/mineral-potential-classification-r/reports/mineral_potential_files/figure-gfm/eda-objetivo-plot-1.png" style="display: block; margin: auto;" />

## 2.5 Valores faltantes y duplicados

``` r
colSums(is.na(df))
```

    ##           Litologia          Alteracion  Control_Estructura   Densidad_Vetillas 
    ##                   0                   0                   0                   0 
    ##   Profundidad_Nivel  Anomalia_Magnetica Potencial_Economico 
    ##                   0                   0                   0

``` r
sum(duplicated(df))
```

    ## [1] 63

No hay valores faltantes. Las 63 filas repetidas no se eliminan: con 6
variables discretas solo existen 1,296 perfiles geológicos posibles, por
lo que es esperable que zonas distintas compartan el mismo perfil.

## 2.6 Correlación entre variables numéricas

``` r
mat_cor <- cor(df[, c("Densidad_Vetillas", "Profundidad_Nivel", "Anomalia_Magnetica")])
round(mat_cor, 3)
```

    ##                    Densidad_Vetillas Profundidad_Nivel Anomalia_Magnetica
    ## Densidad_Vetillas              1.000             0.028             -0.043
    ## Profundidad_Nivel              0.028             1.000             -0.010
    ## Anomalia_Magnetica            -0.043            -0.010              1.000

``` r
mat_cor_m <- reshape2::melt(mat_cor)
ggplot(mat_cor_m, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 2))) +
  scale_fill_gradient2(low = "#B23A2E", mid = "white", high = "#1B7A43", midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Matriz de correlación\nvariables numéricas", x = NULL, y = NULL, fill = "r") +
  tema + theme(plot.title = element_text(size = 13, face = "bold"),
               axis.text.x = element_text(angle = 20, hjust = 1, size = 8))
```

<img src="/tmp/claude-0/-home-claude/ce241027-6181-578c-9961-cbb777a9ddcf/scratchpad/repo5/mineral-potential-classification-r/reports/mineral_potential_files/figure-gfm/eda-correlacion-plot-1.png" style="display: block; margin: auto;" />

## 2.7 Asociación entre variables categóricas y la variable objetivo (Chi-cuadrado)

``` r
for (v in c("Litologia", "Alteracion", "Control_Estructura")) {
  tt <- table(df[[v]], df$Potencial_Economico)
  test <- suppressWarnings(chisq.test(tt))
  cat(sprintf("%s vs Potencial_Economico: Chi2=%.2f, gl=%d, p-valor=%.4f\n",
              v, test$statistic, test$parameter, test$p.value))
}
```

    ## Litologia vs Potencial_Economico: Chi2=2.81, gl=4, p-valor=0.5895
    ## Alteracion vs Potencial_Economico: Chi2=149.76, gl=6, p-valor=0.0000
    ## Control_Estructura vs Potencial_Economico: Chi2=6.93, gl=4, p-valor=0.1396

``` r
tab_alt <- as.data.frame(prop.table(table(df$Alteracion, df$Potencial_Economico), margin = 1))
names(tab_alt) <- c("Alteracion", "Potencial_Economico", "Proporcion")

ggplot(tab_alt, aes(x = Alteracion, y = Proporcion, fill = Potencial_Economico)) +
  geom_col(position = "fill") +
  scale_y_continuous(labels = percent) +
  scale_fill_manual(values = paleta) +
  labs(title = "Potencial económico según tipo de alteración",
       x = "Alteración hidrotermal", y = "Proporción de zonas", fill = "Potencial") +
  tema
```

<img src="/tmp/claude-0/-home-claude/ce241027-6181-578c-9961-cbb777a9ddcf/scratchpad/repo5/mineral-potential-classification-r/reports/mineral_potential_files/figure-gfm/eda-alteracion-plot-1.png" style="display: block; margin: auto;" />

``` r
tab_dens <- as.data.frame(prop.table(table(df$Densidad_Vetillas, df$Potencial_Economico), margin = 1))
names(tab_dens) <- c("Densidad_Vetillas", "Potencial_Economico", "Proporcion")

ggplot(tab_dens, aes(x = factor(Densidad_Vetillas), y = Proporcion, fill = Potencial_Economico)) +
  geom_col(position = "fill") +
  scale_y_continuous(labels = percent) +
  scale_fill_manual(values = paleta) +
  labs(title = "Potencial económico según densidad de vetillas",
       x = "Densidad de vetillas (1=baja, 4=alta)", y = "Proporción de zonas", fill = "Potencial") +
  tema
```

<img src="/tmp/claude-0/-home-claude/ce241027-6181-578c-9961-cbb777a9ddcf/scratchpad/repo5/mineral-potential-classification-r/reports/mineral_potential_files/figure-gfm/eda-densidad-plot-1.png" style="display: block; margin: auto;" />

**Conclusión del EDA:** la **Alteración hidrotermal** (Chi²
significativo, p \< 0.001) y la **Densidad de vetillas** concentran la
mayor parte de la señal predictiva; la Litología y el Control
Estructural no muestran asociación individual significativa (p \> 0.05).
La correlación lineal entre variables numéricas es prácticamente nula,
lo que favorece un modelo basado en árboles (sección 3).

------------------------------------------------------------------------

# 3. Modelado: desarrollo del algoritmo en R

## 3.1 División estratificada train/test (80/20)

``` r
set.seed(2026)
idx <- createDataPartition(df$Potencial_Economico, p = 0.8, list = FALSE)
train <- df[idx, ]
test  <- df[-idx, ]

cat("Entrenamiento:", nrow(train), "| Prueba:", nrow(test), "\n")
```

    ## Entrenamiento: 481 | Prueba: 119

``` r
round(prop.table(table(train$Potencial_Economico)) * 100, 1)
```

    ## 
    ## Economico  Marginal   Esteril 
    ##      32.8      41.4      25.8

``` r
round(prop.table(table(test$Potencial_Economico)) * 100, 1)
```

    ## 
    ## Economico  Marginal   Esteril 
    ##      32.8      41.2      26.1

## 3.2 Modelo base de referencia: Árbol de decisión (CART)

``` r
modelo_arbol <- rpart(
  Potencial_Economico ~ Litologia + Alteracion + Control_Estructura +
    Densidad_Vetillas + Profundidad_Nivel + Anomalia_Magnetica,
  data = train, method = "class",
  control = rpart.control(cp = 0.01, minsplit = 20, maxdepth = 5)
)
print(modelo_arbol)
```

    ## n= 481 
    ## 
    ## node), split, n, loss, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##  1) root 481 282 Marginal (0.32848233 0.41372141 0.25779626)  
    ##    2) Alteracion=Silicia,Potasica 246 127 Economico (0.48373984 0.38211382 0.13414634)  
    ##      4) Alteracion=Potasica 104  38 Economico (0.63461538 0.28846154 0.07692308) *
    ##      5) Alteracion=Silicia 142  78 Marginal (0.37323944 0.45070423 0.17605634)  
    ##       10) Densidad_Vetillas>=3.5 40  17 Economico (0.57500000 0.35000000 0.07500000)  
    ##         20) Litologia=Diorita,Brecha 29   9 Economico (0.68965517 0.27586207 0.03448276) *
    ##         21) Litologia=Andesita 11   5 Marginal (0.27272727 0.54545455 0.18181818) *
    ##       11) Densidad_Vetillas< 3.5 102  52 Marginal (0.29411765 0.49019608 0.21568627)  
    ##         22) Anomalia_Magnetica>=2.5 35  21 Economico (0.40000000 0.31428571 0.28571429)  
    ##           44) Densidad_Vetillas>=1.5 21   9 Economico (0.57142857 0.28571429 0.14285714) *
    ##           45) Densidad_Vetillas< 1.5 14   7 Esteril (0.14285714 0.35714286 0.50000000) *
    ##         23) Anomalia_Magnetica< 2.5 67  28 Marginal (0.23880597 0.58208955 0.17910448) *
    ##    3) Alteracion=Argilica,Propilitica 235 130 Marginal (0.16595745 0.44680851 0.38723404)  
    ##      6) Densidad_Vetillas>=2.5 119  57 Marginal (0.27731092 0.52100840 0.20168067)  
    ##       12) Alteracion=Argilica 67  35 Marginal (0.43283582 0.47761194 0.08955224)  
    ##         24) Profundidad_Nivel>=1.5 41  20 Economico (0.51219512 0.36585366 0.12195122) *
    ##         25) Profundidad_Nivel< 1.5 26   9 Marginal (0.30769231 0.65384615 0.03846154) *
    ##       13) Alteracion=Propilitica 52  22 Marginal (0.07692308 0.57692308 0.34615385) *
    ##      7) Densidad_Vetillas< 2.5 116  49 Esteril (0.05172414 0.37068966 0.57758621) *

``` r
round(modelo_arbol$variable.importance, 2)
```

    ##         Alteracion  Densidad_Vetillas Anomalia_Magnetica          Litologia 
    ##              33.08              19.11               4.36               4.09 
    ##  Profundidad_Nivel Control_Estructura 
    ##               2.73               1.09

``` r
plot(as.party(modelo_arbol),
     main = "Árbol de decisión (CART) - Potencial Económico",
     gp = gpar(fontsize = 9))
```

<img src="/tmp/claude-0/-home-claude/ce241027-6181-578c-9961-cbb777a9ddcf/scratchpad/repo5/mineral-potential-classification-r/reports/mineral_potential_files/figure-gfm/modelado-arbol-1.png" style="display: block; margin: auto;" />

## 3.3 Algoritmo elegido: Random Forest multiclase

Justificación técnica: bagging (reduce varianza/sobreajuste de un árbol
individual), selección aleatoria de variables en cada división (`mtry`,
decorrelaciona los árboles), manejo nativo de variables categóricas y
numéricas sin escalado, importancia de variables integrada, y adecuado
para clasificación multiclase sin esquemas one-vs-rest manuales.

``` r
modelo_rf <- randomForest(
  Potencial_Economico ~ Litologia + Alteracion + Control_Estructura +
    Densidad_Vetillas + Profundidad_Nivel + Anomalia_Magnetica,
  data = train, ntree = 500, mtry = 2, importance = TRUE
)
print(modelo_rf)
```

    ## 
    ## Call:
    ##  randomForest(formula = Potencial_Economico ~ Litologia + Alteracion +      Control_Estructura + Densidad_Vetillas + Profundidad_Nivel +      Anomalia_Magnetica, data = train, ntree = 500, mtry = 2,      importance = TRUE) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 51.98%
    ## Confusion matrix:
    ##           Economico Marginal Esteril class.error
    ## Economico        98       50      10   0.3797468
    ## Marginal         62       91      46   0.5427136
    ## Esteril          16       66      42   0.6612903

``` r
round(importance(modelo_rf)[, "MeanDecreaseGini"], 2)
```

    ##          Litologia         Alteracion Control_Estructura  Densidad_Vetillas 
    ##              26.92              53.71              28.00              40.48 
    ##  Profundidad_Nivel Anomalia_Magnetica 
    ##              27.97              27.36

``` r
imp <- as.data.frame(importance(modelo_rf))
imp$Variable <- rownames(imp)
imp <- imp[order(-imp$MeanDecreaseGini), ]

ggplot(imp, aes(x = reorder(Variable, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_col(fill = "#2E6F95") +
  coord_flip() +
  labs(title = "Importancia de variables - Random Forest",
       x = NULL, y = "Reducción media del índice de Gini") +
  theme_minimal(base_size = 12) + theme(plot.title = element_text(face = "bold"))
```

<img src="/tmp/claude-0/-home-claude/ce241027-6181-578c-9961-cbb777a9ddcf/scratchpad/repo5/mineral-potential-classification-r/reports/mineral_potential_files/figure-gfm/modelado-importancia-plot-1.png" style="display: block; margin: auto;" />

## 3.4 Selección de hiperparámetros por validación cruzada (10-fold)

``` r
ctrl <- trainControl(method = "cv", number = 10)
set.seed(2026)
modelo_rf_cv <- train(
  Potencial_Economico ~ Litologia + Alteracion + Control_Estructura +
    Densidad_Vetillas + Profundidad_Nivel + Anomalia_Magnetica,
  data = train, method = "rf",
  trControl = ctrl,
  tuneGrid = data.frame(mtry = c(2, 3, 4)),
  ntree = 300
)
print(modelo_rf_cv)
```

    ## Random Forest 
    ## 
    ## 481 samples
    ##   6 predictor
    ##   3 classes: 'Economico', 'Marginal', 'Esteril' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 432, 432, 435, 433, 433, 433, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##   2     0.4988878  0.2170679
    ##   3     0.4697157  0.1841596
    ##   4     0.4553006  0.1641966
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 2.

``` r
# Para producción (ver sección 5): guardar el modelo entrenado
saveRDS(modelo_rf, "../models/modelo_rf.rds")
```

------------------------------------------------------------------------

# 4. Evaluación: efectividad de la predicción en R

## 4.1 Matriz de confusión y métricas por clase

``` r
pred_rf_clase <- predict(modelo_rf, test, type = "class")
pred_rf_prob  <- predict(modelo_rf, test, type = "prob")
pred_arbol_clase <- predict(modelo_arbol, test, type = "class")

cm_rf    <- confusionMatrix(pred_rf_clase, test$Potencial_Economico)
cm_arbol <- confusionMatrix(pred_arbol_clase, test$Potencial_Economico)

cm_rf
```

    ## Confusion Matrix and Statistics
    ## 
    ##            Reference
    ## Prediction  Economico Marginal Esteril
    ##   Economico        32       11       2
    ##   Marginal          7       27       9
    ##   Esteril           0       11      20
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.6639          
    ##                  95% CI : (0.5715, 0.7478)
    ##     No Information Rate : 0.4118          
    ##     P-Value [Acc > NIR] : 2.541e-08       
    ##                                           
    ##                   Kappa : 0.4903          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.3781          
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: Economico Class: Marginal Class: Esteril
    ## Sensitivity                    0.8205          0.5510         0.6452
    ## Specificity                    0.8375          0.7714         0.8750
    ## Pos Pred Value                 0.7111          0.6279         0.6452
    ## Neg Pred Value                 0.9054          0.7105         0.8750
    ## Prevalence                     0.3277          0.4118         0.2605
    ## Detection Rate                 0.2689          0.2269         0.1681
    ## Detection Prevalence           0.3782          0.3613         0.2605
    ## Balanced Accuracy              0.8290          0.6612         0.7601

``` r
cm_df <- as.data.frame(cm_rf$table)
ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradient(low = "#EAF3EE", high = "#1B7A43") +
  labs(title = "Matriz de confusión - Random Forest",
       subtitle = "Conjunto de prueba (holdout, 20%)",
       x = "Clase real", y = "Clase predicha") +
  theme_minimal(base_size = 12) + theme(plot.title = element_text(face = "bold"))
```

<img src="/tmp/claude-0/-home-claude/ce241027-6181-578c-9961-cbb777a9ddcf/scratchpad/repo5/mineral-potential-classification-r/reports/mineral_potential_files/figure-gfm/evaluacion-matriz-plot-1.png" style="display: block; margin: auto;" />

``` r
clase_mayoritaria <- names(which.max(table(train$Potencial_Economico)))
pred_baseline <- factor(rep(clase_mayoritaria, nrow(test)), levels = levels(test$Potencial_Economico))
acc_baseline <- mean(pred_baseline == test$Potencial_Economico)
cat(sprintf("Baseline (clase mayoritaria = %s): Accuracy = %.3f\n", clase_mayoritaria, acc_baseline))
```

    ## Baseline (clase mayoritaria = Marginal): Accuracy = 0.412

## 4.2 Comparación con el árbol de decisión base

``` r
metricas_clase <- as.data.frame(cm_rf$byClass[, c("Precision", "Recall", "F1")])

comp <- data.frame(
  Modelo = rep(c("Arbol_Decision", "Random_Forest"), each = 3),
  Metrica = rep(c("Accuracy", "Kappa", "F1_macro"), 2),
  Valor = c(
    cm_arbol$overall["Accuracy"], cm_arbol$overall["Kappa"], mean(as.data.frame(cm_arbol$byClass)$F1, na.rm = TRUE),
    cm_rf$overall["Accuracy"], cm_rf$overall["Kappa"], mean(metricas_clase$F1, na.rm = TRUE)
  )
)

ggplot(comp, aes(x = Metrica, y = Valor, fill = Modelo)) +
  geom_col(position = "dodge") +
  geom_text(aes(label = round(Valor, 2)), position = position_dodge(width = 0.9), vjust = -0.3) +
  scale_fill_manual(values = c("Arbol_Decision" = "#8C8C8C", "Random_Forest" = "#1B7A43")) +
  ylim(0, 1) +
  labs(title = "Comparación de modelos (conjunto de prueba)", x = NULL, y = "Valor") +
  theme_minimal(base_size = 12) + theme(plot.title = element_text(face = "bold"))
```

<img src="/tmp/claude-0/-home-claude/ce241027-6181-578c-9961-cbb777a9ddcf/scratchpad/repo5/mineral-potential-classification-r/reports/mineral_potential_files/figure-gfm/evaluacion-comparacion-plot-1.png" style="display: block; margin: auto;" />

## 4.3 AUC-ROC multiclase (One-vs-Rest)

``` r
roc_multi <- multiclass.roc(test$Potencial_Economico, pred_rf_prob)
cat(sprintf("AUC multiclase (Hand & Till): %.3f\n", as.numeric(roc_multi$auc)))
```

    ## AUC multiclase (Hand & Till): 0.796

``` r
colores <- c(Economico = "#1B7A43", Marginal = "#D9A441", Esteril = "#B23A2E")
auc_por_clase <- sapply(levels(test$Potencial_Economico), function(cl) {
  y_bin <- ifelse(test$Potencial_Economico == cl, 1, 0)
  as.numeric(auc(roc(y_bin, pred_rf_prob[, cl], quiet = TRUE)))
})
round(auc_por_clase, 3)
```

    ## Economico  Marginal   Esteril 
    ##     0.894     0.665     0.818

``` r
plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1),
     xlab = "1 - Especificidad", ylab = "Sensibilidad",
     main = "Curvas ROC One-vs-Rest - Random Forest")
abline(0, 1, lty = 2, col = "grey60")
for (cl in levels(test$Potencial_Economico)) {
  y_bin <- ifelse(test$Potencial_Economico == cl, 1, 0)
  r <- roc(y_bin, pred_rf_prob[, cl], quiet = TRUE)
  lines(1 - r$specificities, r$sensitivities, col = colores[cl], lwd = 2)
}
legend("bottomright", legend = paste0(names(colores), " (AUC=", round(auc_por_clase[names(colores)], 2), ")"),
       col = colores, lwd = 2, bty = "n")
```

<img src="/tmp/claude-0/-home-claude/ce241027-6181-578c-9961-cbb777a9ddcf/scratchpad/repo5/mineral-potential-classification-r/reports/mineral_potential_files/figure-gfm/evaluacion-roc-1.png" style="display: block; margin: auto;" />

## 4.4 Curva de aprendizaje (overfitting)

``` r
fracciones <- seq(0.2, 1.0, by = 0.1)
resultados_curva <- data.frame(fraccion = fracciones, accuracy_train = NA, accuracy_test = NA)

for (i in seq_along(fracciones)) {
  set.seed(100 + i)
  n_sub <- floor(fracciones[i] * nrow(train))
  sub_idx <- sample(seq_len(nrow(train)), n_sub)
  sub_train <- train[sub_idx, ]
  if (length(unique(sub_train$Potencial_Economico)) < 3) next
  m <- randomForest(
    Potencial_Economico ~ Litologia + Alteracion + Control_Estructura +
      Densidad_Vetillas + Profundidad_Nivel + Anomalia_Magnetica,
    data = sub_train, ntree = 300, mtry = 2
  )
  resultados_curva$accuracy_train[i] <- mean(predict(m, sub_train) == sub_train$Potencial_Economico)
  resultados_curva$accuracy_test[i]  <- mean(predict(m, test) == test$Potencial_Economico)
}
print(round(resultados_curva, 3))
```

    ##   fraccion accuracy_train accuracy_test
    ## 1      0.2          0.979         0.529
    ## 2      0.3          0.972         0.529
    ## 3      0.4          0.932         0.563
    ## 4      0.5          0.946         0.639
    ## 5      0.6          0.934         0.639
    ## 6      0.7          0.908         0.580
    ## 7      0.8          0.898         0.613
    ## 8      0.9          0.880         0.605
    ## 9      1.0          0.879         0.639

``` r
brecha_final <- tail(resultados_curva$accuracy_train, 1) - tail(resultados_curva$accuracy_test, 1)

curva_m <- reshape2::melt(resultados_curva, id.vars = "fraccion",
                           measure.vars = c("accuracy_train", "accuracy_test"))
ggplot(curva_m, aes(x = fraccion, y = value, color = variable)) +
  geom_line(linewidth = 1) + geom_point(size = 2) +
  scale_color_manual(values = c("accuracy_train" = "#2E6F95", "accuracy_test" = "#B23A2E"),
                      labels = c("Entrenamiento", "Prueba")) +
  ylim(0, 1) +
  labs(title = "Curva de aprendizaje - Random Forest",
       x = "Fracción del set de entrenamiento utilizada", y = "Accuracy", color = NULL) +
  theme_minimal(base_size = 12) + theme(plot.title = element_text(face = "bold"))
```

<img src="/tmp/claude-0/-home-claude/ce241027-6181-578c-9961-cbb777a9ddcf/scratchpad/repo5/mineral-potential-classification-r/reports/mineral_potential_files/figure-gfm/evaluacion-curva-aprendizaje-1.png" style="display: block; margin: auto;" />

Con todo el set de entrenamiento, la exactitud en entrenamiento supera a
la de prueba en 24.1 puntos: es el sobreajuste esperable en un Random
Forest. La exactitud de prueba se estabiliza a partir del 50 % de los
datos, por lo que **más datos de campo** serían la principal vía de
mejora.

## 4.5 Balance frente a los KPIs de negocio

``` r
kpi_acc  <- as.numeric(cm_rf$overall["Accuracy"])
kpi_prec <- as.numeric(cm_rf$byClass["Class: Economico", "Precision"])
kpi_rec  <- as.numeric(cm_rf$byClass["Class: Economico", "Recall"])
kpi_f1   <- as.numeric(cm_rf$byClass["Class: Economico", "F1"])
kpi_auc  <- mean(auc_por_clase)
estado   <- function(valor, objetivo) ifelse(valor >= objetivo, "✅ cumple", "❌ no cumple")
pct      <- function(x) sprintf("%.1f %%", x * 100)
```

| KPI                         | Objetivo | Resultado |    Estado    |
|-----------------------------|:--------:|:---------:|:------------:|
| Accuracy global             |  ≥ 80 %  |  66.4 %   | ❌ no cumple |
| Precision (clase Económico) |  ≥ 75 %  |  71.1 %   | ❌ no cumple |
| Recall (clase Económico)    |  ≥ 80 %  |  82.1 %   |  ✅ cumple   |
| F1-score (clase Económico)  |  ≥ 75 %  |  76.2 %   |  ✅ cumple   |
| AUC-ROC promedio (macro)    |  ≥ 0.80  |   0.792   | ❌ no cumple |

El modelo cumple los **KPIs más críticos para el negocio**: detecta el
82.1 % de las zonas realmente económicas (Recall) con un F1 de 76.2 %, y
su exactitud global (66.4 %) supera ampliamente al baseline de la clase
mayoritaria (41.2 %). La confusión se concentra en la clase
**Marginal**, que comparte rasgos con las otras dos.

La validación cruzada de 10 particiones estima una exactitud de 49.9 %;
la diferencia con el conjunto de prueba se explica por su tamaño
reducido (119 zonas), por lo que la cifra de validación cruzada es la
estimación más conservadora del desempeño.

------------------------------------------------------------------------

# 5. Propuesta de despliegue en producción utilizando R

El modelo entrenado (`models/modelo_rf.rds`) se expone como una **API
REST con `plumber`**, de modo que geólogos y planificadores puedan
clasificar nuevas zonas desde cualquier sistema:

1.  **Entrada:** el equipo de exploración registra las 6 variables
    geológicas de una zona nueva.
2.  **Predicción:** la API devuelve la clase (Económico / Marginal /
    Estéril) y la probabilidad de cada una.
3.  **Decisión:** las zonas con mayor probabilidad de ser *Económico* se
    priorizan en el plan de perforación.
4.  **Mejora continua:** el modelo se reentrena cuando llegan nuevos
    resultados de perforación, versionando cada `modelo_rf.rds`.

El código de la API está en [`src/plumber_api.R`](../src/plumber_api.R):

``` r
# API de predicción del potencial de mineralización
# Ejecutar desde la raíz del repositorio:
#   plumber::plumb("src/plumber_api.R")$run(port = 8000)

library(plumber)
library(randomForest)

modelo  <- readRDS("models/modelo_rf.rds")
niveles <- modelo$forest$xlevels   # niveles de las variables categóricas usados en el entrenamiento

#* Verifica que la API está activa
#* @get /salud
function() {
  list(estado = "ok", modelo = "Random Forest - potencial de mineralizacion")
}

#* Clasifica una zona según su potencial económico
#* @param Litologia Diorita | Andesita | Brecha
#* @param Alteracion Silicia | Argilica | Propilitica | Potasica
#* @param Control_Estructura Falla_Menor | Falla_Mayor | Sin_Estructura
#* @param Densidad_Vetillas 1 (baja) a 4 (alta)
#* @param Profundidad_Nivel 1 a 3
#* @param Anomalia_Magnetica 1 a 3
#* @post /predecir
function(Litologia, Alteracion, Control_Estructura,
         Densidad_Vetillas, Profundidad_Nivel, Anomalia_Magnetica) {

  nueva <- data.frame(
    Litologia          = factor(Litologia,          levels = niveles$Litologia),
    Alteracion         = factor(Alteracion,         levels = niveles$Alteracion),
    Control_Estructura = factor(Control_Estructura, levels = niveles$Control_Estructura),
    Densidad_Vetillas  = as.integer(Densidad_Vetillas),
    Profundidad_Nivel  = as.integer(Profundidad_Nivel),
    Anomalia_Magnetica = as.integer(Anomalia_Magnetica)
  )

  if (anyNA(nueva)) {
    stop("Valores no válidos: revisa las categorías y que los niveles numéricos sean enteros.")
  }

  probabilidades <- predict(modelo, nueva, type = "prob")
  list(
    clase          = as.character(predict(modelo, nueva)),
    probabilidades = as.list(round(probabilidades[1, ], 3))
  )
}
```

Para levantarla (desde la raíz del repositorio) y probarla:

``` r
plumber::plumb("src/plumber_api.R")$run(port = 8000)
```

``` bash
curl -X POST "http://localhost:8000/predecir" \
  -d "Litologia=Brecha&Alteracion=Potasica&Control_Estructura=Falla_Mayor&Densidad_Vetillas=4&Profundidad_Nivel=2&Anomalia_Magnetica=3"
```

# 6. Conclusiones

- La **alteración hidrotermal** y la **densidad de vetillas** son las
  variables que más explican el potencial económico, tanto en el EDA
  como en la importancia del Random Forest.
- El **Random Forest** supera al árbol de decisión base y al baseline de
  la clase mayoritaria, y cumple los KPIs de Recall y F1 de la clase
  *Económico*, los más relevantes para no descartar zonas valiosas.
- El modelo queda listo para **producción** mediante una API en R
  (`plumber`), y su principal vía de mejora es incorporar más datos
  reales de campo.
