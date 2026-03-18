# 📃 Documentación AdaBoostClassifier Alejandro Simón Aparicio CE IA y Big Data

## Introducción:

**AdaBoost (Adaptive Boosting)** es un algoritmo de *ensemble* propuesto en 1995 por Yoav Freund y Robert Schapire. Su idea principal consiste en entrenar **clasificadores débiles de forma secuencial**, donde cada nuevo modelo trata de corregir los errores cometidos por el anterior.

En lugar de construir un único modelo complejo, AdaBoost combina múltiples modelos simples (normalmente árboles de decisión poco profundos) hasta formar **un clasificador fuerte** con mejor capacidad predictiva. <br> <br>

![Ejemplo de AdaBoostClassifier](https://fhernanb.github.io/libro_mod_pred/images/adaboost_illustration.png) <br> <br>

## Funcionamiento: La idea de los "Investigadores"

Se puede entender AdaBoost como un conjunto de clasificadores que analizan el mismo problema de manera secuencial, donde cada uno trata de corregir los errores cometidos por el anterior.

Cada clasificador representa un **modelo débil**. Individualmente no son especialmente precisos, pero al combinarse logran un rendimiento elevado.

1. **Primer Clasificador**: Se entrena con todas las muestras con el mismo peso. Tras evaluar su rendimiento, AdaBoost calcula su error y le asigna un peso en función de ese error.
    - Si el error es bajo, su peso en la decisión final será mayor.
    - Las muestras mal clasificadas aumentan su peso para la siguiente iteración.

2. **Segundo Clasificador**: Se entrena sobre el mismo conjunto de datos, pero utilizando los pesos actualizados tras la primera iteración. Las muestras que fueron clasificadas incorrectamente tienen ahora mayor peso, por lo que influyen más en el entrenamiento. De esta forma, el modelo comienza en los casos más difíciles.

3. **Clasificadores sucesivos**: El proceso continúa de manera secuencial, donde cada nuevo clasificador intenta corregir los errores acumulados hasta ese momento. En cada iteración se recalculan tanto los pesos de las muestras como el peso del clasificador en función de su error.

4. **Proceso iterativo completo**:
   Este procedimiento se repite durante un número determinado de iteraciones definido por el hiperparámetro `n_estimators` (por ejemplo, 100).

**Predicción final:**
La clasificación se obtiene mediante una votación ponderada entre todos los clasificadores entrenados. Cada uno aporta su predicción con un peso proporcional a su rendimiento, y la clase con mayor suma de pesos se selecciona como resultado final.

## Implementación del modelo:

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# Definición del modelo
model = AdaBoostClassifier(n_estimators = 100, random_state = SEED)

# Entrenamiento
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)

# Evaluación
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
```

## Parámetros y Atributos útiles: 

- **`n_estimators`**: Número total de clasificadores que se entrenan en secuencia. Aumentar este valor puede mejorar la precisión, pero también incrementar el riesgo de sobreajuste y el tiempo de entrenamiento.

- **`learning_rate`**: Peso que se asigna a cada clasificador en la votación final. Un valor más bajo puede mejorar la generalización, pero requiere más estimadores.

- **`estimator_errors_`**: Lista con el error de cada clasificador durante el entrenamiento. Permite analizar cuáles contribuyen más o menos al conjunto final.

- **`estimator_weights_`**: Peso que AdaBoost asigna a cada clasificador en la predicción final. Los clasificadores con menor error tienen mayor influencia.

- **`feature_importances_`**: Importancia de cada variable del dataset, calculada a partir de la contribución de los clasificadores base. Solo disponible si el clasificador base lo soporta, útil para interpretar el modelo y seleccionar características relevantes.

## Ventajas y Desventajas frente a otros modelos ensembles (árboles):

| Modelo | Ventajas | Desventajas | 
|--------|----------|-------------|
| **Árbol de decisión** | Modelo simple y base para muchos ensembles. <br> Fácil de interpretar y visualizar. <br> Entrenamiento rápido incluso con pocos datos. <br> No requiere gran ajuste de hiperparámetros. | Alta tendencia al sobreajuste. <br> Sensible a pequeñas variaciones en los datos. <br> Menor precisión frente a métodos ensemble. |
| **Random Forest** | Combina múltiples árboles entrenados en paralelo (**bagging**), aumentando la precisión y estabilidad. <br> Robusto frente a ruido y valores atípicos. <br> Reduce la varianza del modelo. | Menor interpretabilidad que un árbol simple. <br> Mayor consumo de memoria RAM. <br> Puede aumentar el tiempo de predicción. |
| **AdaBoost** | Corrige los errores de los clasificadores anteriores (**boosting**), mejorando el rendimiento de modelos débiles. <br> Muy eficaz en problemas complejos y cuando hay pocas muestras de alguna clase. | Sensible a valores atípicos. <br> Entrenamiento secuencial (no paralelizable). <br> Riesgo de sobreajuste si no se regula adecuadamente. |
| **VotingClassifier** | Combina modelos diferentes para aumentar la robustez y estabilidad de las predicciones. <br> Permite dos métodos de votación: **Hard** (mayoría simple) y **Soft** (promedio de probabilidades). | Mayor tiempo de entrenamiento al requerir varios modelos completos. <br> Cada modelo necesita ajuste individual de hiperparámetros. <br> Un modelo con bajo rendimiento puede afectar negativamente al resultado final. |

## Consideraciones de implementación:

Al trabajar con AdaBoost es importante tener en cuenta varios aspectos que pueden afectar a su rendimiento y capacidad de generalización.

- **Sensibilidad a valores atípicos:**
  Las muestras muy diferentes al resto pueden recibir un peso elevado tras ser clasificadas incorrectamente, lo que puede distorsionar el entrenamiento de los clasificadores posteriores.

- **Riesgo de sobreajuste:**
  Un número elevado de estimadores (`n_estimators`) o una tasa de aprendizaje (`learning_rate`) inadecuada puede provocar que el modelo se ajuste en exceso al conjunto de entrenamiento.

- **Entrenamiento secuencial:**
  Al depender cada clasificador del anterior, el proceso no puede paralelizarse fácilmente, lo que puede aumentar el tiempo de entrenamiento en comparación con métodos como Random Forest.

- **Datos desbalanceados:** 
  Aunque el algoritmo puede prestar mayor atención a la clase minoritaria al aumentar su peso, también puede amplificar el efecto del ruido si no se controla adecuadamente.

- **Clasificador base:**
  AdaBoost suele utilizar clasificadores simples como árboles de decisión muy poco profundos. Si se emplea un modelo base demasiado complejo, puede aumentar el riesgo de sobreajuste y reducir la capacidad de generalización.
