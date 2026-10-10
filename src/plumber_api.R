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
