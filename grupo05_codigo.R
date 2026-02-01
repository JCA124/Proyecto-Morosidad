# ============================================================
# Librerías
# ============================================================
library(fpp3)
library(dplyr)
library(ggplot2)
library(purrr)


# ============================================================
# Carga y preparación de datos 
# ============================================================
base <- readRDS("base_morosidad_exogenas_2020_2025.rds")

azuay <- base %>%
  filter(Provincia == "AZUAY") %>%index_by(Fecha) %>%
  summarise(
    Morosidad = mean(Morosidad, na.rm = TRUE),
    TasaConsumo = first(TasaConsumo),
    Inflacion_mensual = first(Inflacion_mensual),
    Remesas = first(Remesas)
  ) %>% fill_gaps() %>%
  mutate(CovidEscalon = as.numeric(Fecha >= yearmonth("2020 Mar")))
min(azuay$Morosidad)

# ============================================================
# Análisis exploratorio
# ============================================================
media_mora <- mean(azuay$Morosidad, na.rm = TRUE)

azuay %>%
  autoplot(Morosidad) +
  geom_hline(yintercept = media_mora,
             linetype = "dashed", linewidth = 1,
             colour = "red") +
  annotate("text",
           x = min(azuay$Fecha, na.rm = TRUE),
           y = media_mora,
           label = paste0("Media = ", round(media_mora, 4)),
           vjust = -0.8, hjust = 0,
           colour = "red", size = 3.5) +
  labs(title = "Serie de Morosidad - Azuay",
       y = "Morosidad", x = "Fecha")


azuay %>% gg_season(Morosidad)
azuay %>% ACF(Morosidad) %>% autoplot()
azuay %>% PACF(Morosidad) %>% autoplot()

# Diferenciación 
azuay %>% mutate(Dif1 = difference(Morosidad)) %>% ACF(Dif1) %>% autoplot()
azuay %>% mutate(Dif1 = difference(Morosidad)) %>% PACF(Dif1) %>% autoplot()

# ============================================================
# Entrenamiento / Prueba
# ============================================================

corte <- yearmonth("2024 Aug")
entrenamiento <- azuay %>% filter(Fecha <= corte)
prueba        <- azuay %>% filter(Fecha >  corte)

# ============================================================
# Modelos 
# ============================================================

#Visualizar correlacion de las variables exogenas
corr_exogenas <- azuay %>%as_tibble() %>% select(TasaConsumo, Inflacion_mensual, Remesas) %>% as.matrix()
cor(corr_exogenas, use = "complete.obs")


modelos <- entrenamiento %>%
  model(
    NAIVE = NAIVE(Morosidad),
    ARIMA = ARIMA(Morosidad ),
    ARIMAX = ARIMA(Morosidad ~ TasaConsumo + Inflacion_mensual + Remesas),
    ARIMAX_Covid = ARIMA(Morosidad ~ TasaConsumo + Inflacion_mensual + Remesas + CovidEscalon)
  )

# Ver orden de arimas
# modelos %>% select(ARIMA) %>% report()
# modelos %>% select(ARIMAX) %>% report()
# modelos %>% select(ARIMAX_Covid) %>% report()

# ============================================================
# Pronóstico en prueba + métricas
# ============================================================
pronostico_prueba <- modelos %>% forecast(new_data = prueba)

metricas_prueba <- pronostico_prueba %>%accuracy(prueba) %>%
                   select(.model, RMSE, MAE, MAPE) %>% arrange(RMSE)
metricas_prueba

autoplot(pronostico_prueba, prueba, level = NULL, linewidth = 1) +
  labs(
    title = "Ajuste de Modelos",
    y = "Morosidad",
    x = "Fecha"
  ) +
  scale_colour_manual(values = c(
    "ARIMAX"       = "#ADA82D",
    "ARIMAX_Covid" = "#31B0B0",
    "ARIMA"        = "#2B9E56",
    "NAIVE"        = "#B03198"
  )) +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal"
  )



# ============================================================
# Métricas de ajuste (AICc / BIC)
# ============================================================
criterios_info <- modelos %>%glance() %>%select(.model, AICc, BIC) %>%arrange(AICc)
criterios_info

# Tabla combinada (AICc/BIC + RMSE/MAPE)
tabla_comparacion <- metricas_prueba %>%left_join(criterios_info, by = ".model") %>%
                     select(.model, AICc, BIC, RMSE, MAE, MAPE) %>%arrange(RMSE)

print(tabla_comparacion)

# ============================================================
# Validación por residuos (Ljung-Box + gráficos)
# ============================================================
tabla_ljung <- augment(modelos) %>%
  group_by(.model) %>%
  features(.innov, feasts::ljung_box, lag = 24) %>%
  select(.model, lb_stat, lb_pvalue) %>%
  arrange(desc(lb_pvalue))

tabla_ljung


# Gráficos de residuos 
modelos %>% select(ARIMA) %>% gg_tsresiduals()
modelos %>% select(ARIMAX_Covid) %>% gg_tsresiduals()

# ============================================================
# Pronóstico futuro (h meses) 
# ============================================================
h <- 9

# Pronóstico de exógenas
modelo_tasa <- azuay %>% model(ARIMA(TasaConsumo))
modelo_inflacion <- azuay %>% model(ARIMA(Inflacion_mensual))
modelo_remesas <- azuay %>% model(ARIMA(Remesas))

pron_tasa <- modelo_tasa %>% forecast(h = h)
pron_inflacion <- modelo_inflacion %>% forecast(h = h)
pron_remesas <- modelo_remesas %>% forecast(h = h)

exogenas_futuras <- pron_tasa %>% as_tibble() %>% transmute(Fecha, TasaConsumo = .mean) %>%
  left_join(pron_inflacion %>% as_tibble() %>% transmute(Fecha, Inflacion_mensual = .mean), by = "Fecha") %>%
  left_join(pron_remesas %>% as_tibble() %>% transmute(Fecha, Remesas = .mean), by = "Fecha") %>%
  mutate(CovidEscalon = 1) %>% as_tsibble(index = Fecha)

modelo_final <- azuay %>%
  model(ARIMAX_Covid = ARIMA(Morosidad ~ TasaConsumo + Inflacion_mensual + Remesas + CovidEscalon))

pronostico_futuro <- modelo_final %>% forecast(new_data = exogenas_futuras)

serie_grafico <- azuay %>% filter(Fecha >= yearmonth("2024 Apr"))

# Datos y grafico de Predicción
tabla_ic <- pronostico_futuro %>%hilo(level = 95) %>% as_tibble()

valores_futuros <- tabla_ic %>%transmute(
    `Mes (fin)` = Fecha,
    `Morosidad esperada` = round(.mean, 5),
    `IC 95% inferior` = round(sapply(`95%`, function(z) z$lower), 5),
    `IC 95% superior` = round(sapply(`95%`, function(z) z$upper), 5)
  )
valores_futuros



pi95 <- pronostico_futuro %>% hilo(level = 95)

ggplot() + geom_line(data = serie_grafico, aes(x = Fecha, y = Morosidad), colour = "black") +
  geom_ribbon(data = pi95,aes(x = Fecha, ymin = `95%`$lower, ymax = `95%`$upper),
  fill = "#EFCEC8", alpha = 0.55) +geom_line(data = pi95,aes(x = Fecha, y = .mean),
  colour = "orange", linewidth = 1.1) +labs(title = 
  "Predicción futura Morosidad - Azuay",y = "Morosidad",x = "Fecha")



#=========================== Modelacion Cortando la serie en Enero 2023 + Modelo de Red Neuronal ===============================
candidatos <- c("2023 Jan")
corte_train <- "2025 Jan"
inicio_test <- "2025 Feb"
set.seed(123)

evaluar_y_graficar <- function(inicio_estable_chr){
  
  azuay_cut <- azuay %>% filter_index(inicio_estable_chr ~ .)
  entrenamiento <- azuay_cut %>% filter_index(~ corte_train)
  prueba        <- azuay_cut %>% filter_index(inicio_test ~ .)
  
  entrenamiento_ok <- entrenamiento %>%filter(!is.na(TasaConsumo), !is.na(Inflacion_mensual), !is.na(Remesas), !is.na(Morosidad))
  
  prueba_ok <- prueba %>%filter(!is.na(TasaConsumo), !is.na(Inflacion_mensual), !is.na(Remesas), !is.na(Morosidad))

  modelos <- entrenamiento_ok %>%
    model(
      ARIMAX = ARIMA(Morosidad ~ TasaConsumo + Inflacion_mensual + Remesas),
      NNARX = NNETAR(Morosidad ~ TasaConsumo + Inflacion_mensual + Remesas)
    )
  
  fc <- modelos %>% forecast(new_data = prueba_ok)
  
  #  Tabla de métricas
  mape_tbl <- fc %>% accuracy(prueba_ok) %>%select(.model, MAPE, RMSE) %>%
    mutate(inicio_estable = inicio_estable_chr) %>%arrange(MAPE)
  
  print(mape_tbl)
  
  # Gráfico de ajuste 
  p <- autoplot(fc, azuay_cut, level = NULL, linewidth = 0.7) +
    autolayer(prueba_ok, Morosidad, colour = "black", linewidth = 0.8) +
    labs(
      title = paste0("Ajuste en prueba (recorte desde ", inicio_estable_chr, ")"),
      y = "Morosidad",
      x = "Fecha"
    ) +
    guides(colour = guide_legend(title = "Modelo"))
  
  print(p)
  
  return(mape_tbl)
}

mape_todos <- purrr::map(candidatos, evaluar_y_graficar) %>%purrr::compact() %>%bind_rows() %>%arrange(MAPE)
mape_todos

# -----------------------------
# Prediccion (8 meses)
# -----------------------------
inicio_estable <- "2023 Jan"
ultimo_obs     <- "2025 Aug"  
inicio_fc      <- "2025 Sep"
fin_fc         <- "2026 Feb"

azuay_cut <- azuay %>% filter_index(inicio_estable ~ .)
train <- azuay_cut %>% filter_index(~ ultimo_obs)
fit_tasa <- train %>% model(ARIMA(TasaConsumo))
fit_inf  <- train %>% model(ARIMA(Inflacion_mensual))
fit_rem  <- train %>% model(ARIMA(Remesas))

fc_tasa <- fit_tasa %>% forecast(h = 8)
fc_inf  <- fit_inf  %>% forecast(h = 8)
fc_rem  <- fit_rem  %>% forecast(h = 8)

exogenas_futuras <- fc_tasa %>% as_tibble() %>%
  transmute(Fecha, TasaConsumo = .mean) %>%
  left_join(fc_inf %>% as_tibble() %>% transmute(Fecha, Inflacion_mensual = .mean), by = "Fecha") %>%
  left_join(fc_rem %>% as_tibble() %>% transmute(Fecha, Remesas = .mean), by = "Fecha") %>%
  filter(Fecha >= yearmonth(inicio_fc), Fecha <= yearmonth(fin_fc)) %>%
  as_tsibble(index = Fecha)

fit_arimax <- train %>%model(ARIMAX = ARIMA(Morosidad ~ TasaConsumo + Inflacion_mensual + Remesas))

fc_sep25_feb26 <- fit_arimax %>% forecast(new_data = exogenas_futuras)
historia_plot <- azuay_cut %>% filter_index("2024 Jan" ~ .)

autoplot(fc_sep25_feb26, historia_plot, level = 95, linewidth = 1) +
  labs(
    title = "Prediccion ARIMAX (recorte 2023 Jan)",
    y = "Morosidad",
    x = "Fecha"
  )
  
# =============================
# TABLA PRONÓSTICO 6 MESES 
# =============================

# fc_sep25_feb26 es un fable con distribuciones en .distribution
# Para sacar límites, usa hilo() y luego unpack_hilo()

tabla_pron_6m <- fc_sep25_feb26 %>%
  fabletools::hilo(level = 95) %>%
  fabletools::unpack_hilo(`95%`) %>%     
  tibble::as_tibble() %>%
  dplyr::transmute(
    Mes = Fecha,
    Morosidad_esperada = round(.mean, 5),
    IC95_min = round(`95%_lower`, 5),
    IC95_max = round(`95%_upper`, 5)
  )

tabla_pron_6m


