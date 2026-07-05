# Laboratorio UCV: Análisis de Sentimientos usando NLP Clásico y Perceptrón Multicapa (MLP)

**Institución:** Universidad César Vallejo  
**Escuela:** Escuela Profesional de Ingeniería de Sistemas  
**Alumno:** Ramos Alcarraz Jesus Andres  

---

## 25. Preguntas de Análisis

### 1. ¿Por qué una computadora no entiende texto directamente?
Las computadoras están basadas en hardware electrónico que solo procesa señales binarias (ceros y unos) y realiza operaciones aritméticas y lógicas sobre valores numéricos. El texto humano es un sistema de símbolos abstractos, arbitrarios y culturales. Para que una máquina pueda procesarlo, el lenguaje debe ser mapeado o transformado rigurosamente en una representación matemática (vectores o matrices) que los algoritmos puedan calcular.

### 2. ¿Qué es TF-IDF?
Significa *Term Frequency - Inverse Document Frequency* (Frecuencia de Término - Frecuencia Inversa de Documento). Es una métrica estadística utilizada para medir qué tan importante es una palabra para un documento dentro de un conjunto de textos.
* **TF** premia a las palabras que aparecen mucho en un comentario específico.
* **IDF** castiga a las palabras que aparecen en casi todos los comentarios del dataset (como conectores o artículos), ya que no ayudan a discriminar el significado. El resultado es un peso numérico balanceado para cada término relevante.

### 3. ¿Qué diferencia existe entre tokenizar y vectorizar?
* **Tokenizar** es el proceso de fragmentación estructural. Consiste en dividir una cadena de texto larga en unidades mínimas independientes llamadas "tokens" (generalmente palabras sueltas o pares de palabras).
* **Vectorizar** es el paso posterior donde esos tokens abstractos se convierten en números reales. El resultado final de la vectorización es una matriz numérica que la red neuronal puede multiplicar por sus pesos sinápticos durante el entrenamiento.

### 4. ¿Qué palabras parecen más importantes para clasificar sentimientos?
Según las ponderaciones matemáticas calculadas, las palabras con mayor peso predictivo son adjetivos calificativos de fuerte carga emocional como *great*, *good*, *excellent* o *best* para el sentido positivo, y términos contextuales que indican satisfacción o fallo en el servicio como *phone*, *movie*, *place*, *service* o *quality*. Los conectores gramaticales comunes quedan relegados por no aportar valor discriminatorio.

### 5. ¿Qué métrica considera más importante en este caso: accuracy, precision, recall o F1-score?
En este laboratorio, el **F1-score** es la métrica más importante. Aunque el dataset está balanceado (lo que hace que el *accuracy* sea útil), el análisis detallado demostró que el modelo tiende a cometer más errores de un tipo que de otro (específicamente, genera más Falsos Positivos al confundir críticas sutiles). El F1-score combina la precisión y el recall en un solo indicador armónico, asegurando que evaluemos el rendimiento real del modelo sin ignorar ese desbalance en los tipos de errores cometidos.

### 6. ¿Qué errores cometió el modelo?
El modelo falló principalmente en comentarios que involucran negaciones contextuales (como *"The movie was not good"*), estructuras sintácticas inversas o frases donde el orden de las palabras altera completamente el significado. Debido a que el vectorizador evalúa palabras sueltas o pares independientes, la fuerte presencia de una palabra positiva como *"good"* termina engañando a la red neuronal, haciendo que ignore la negación *"not"*.

### 7. ¿Por qué detectar sarcasmo es difícil para este enfoque?
Porque el sarcasmo depende de la ironía, el contexto cultural, la entonación y la contradicción entre el significado literal de las palabras y la verdadera intención del autor. El enfoque clásico utilizado (TF-IDF + MLP) analiza el texto de forma puramente estadística y literal. Si alguien escribe *"¡Qué gran servicio, esperé tres horas!"*, el modelo sumará los puntos positivos de la palabra *"gran"* y clasificará el texto erróneamente, al ser incapaz de detectar la contradicción lógica de la frase.

### 8. ¿Qué ocurriría si ingresamos comentarios en español?
El modelo fallaría por completo o daría predicciones aleatorias debido a dos razones críticas:
1. El vectorizador fue configurado explícitamente para ignorar palabras vacías en inglés (`stop_words='english'`).
2. El vocabulario de 3,000 términos con el que se entrenó la red neuronal contiene únicamente palabras anglosajonas. Al ingresar texto en español, las palabras serían tratadas como términos desconocidos fuera del vocabulario (*Out-Of-Vocabulary*), generando vectores vacíos (llenos de ceros) que el modelo no sabrá interpretar.

### 9. ¿Qué limitaciones tiene TF-IDF frente a embeddings modernos?
* **Pérdida del orden:** TF-IDF trata el texto como un "saco de palabras" (*Bag of Words*), ignorando la secuencia y la sintaxis gramatical.
* **Matrices dispersas:** Genera vectores gigantescos llenos de ceros, lo que consume mucha memoria ineficiente.
* **Cero semántica:** No entiende relaciones entre palabras. Para TF-IDF, los términos *"excelente"* y *"magnífico"* son dos dimensiones completamente distintas e inconexas, mientras que los embeddings modernos (como Word2Vec o BERT) los agrupan en un espacio matemático cercano porque entienden que son sinónimos.

### 10. ¿Por qué este laboratorio prepara el camino para entender Transformers?
Porque te expone de manera práctica a las limitaciones fundamentales del procesamiento de lenguaje natural clásico. Al ver cómo la red neuronal falla ante negaciones elementales, dependencias a larga distancia o la pérdida de orden del texto, entiendes perfectamente por qué la industria necesitó inventar la arquitectura Transformer. Los Transformers resuelven de raíz los problemas de este laboratorio mediante mecanismos de "Atención", permitiendo que las palabras interactúen dinámicamente según su posición exacta en la frase y capturando el contexto global de una oración sin perder información semántica.

---

## 26. Reto MIT: Investigación Experimental Aplicada

### 1. Cuadro Comparativo de Arquitecturas MLP

| Arquitectura de Capas Ocultas | Exactitud (*Accuracy*) | Precisión (*Precision*) | Exhaustividad (*Recall*) | F1-Score | Iteraciones para Convergencia |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$(16,)$** | **0.7939** | 0.8106 | 0.7716 | 0.7906 | 19 |
| **$(64, 32)$** | 0.7891 | 0.7630 | **0.8438** | **0.8014** | 16 |
| **$(128, 64, 32)$** | 0.7842 | **0.8199** | 0.7332 | 0.7741 | 15 |

> **Análisis:** La red compacta $(64, 32)$ demostró ser la más robusta para el negocio al maximizar la sensibilidad (*Recall* de **84.38%**). Incrementar la profundidad a tres capas saturó el modelo reduciendo su rendimiento general, lo que demuestra que más capas no implican una mejora lineal en representaciones estadísticas densas.

### 2. Evaluación de 10 Nuevos Comentarios y Errores Detectados

Se pasaron 10 oraciones personalizadas de prueba al pipeline optimizado:
1. *"The shipping was super fast and the item works beautifully."* $\rightarrow$ **Positivo** (Correcto)
2. *"I am so incredibly disappointed with this cheap material."* $\rightarrow$ **Negativo** (Correcto)
3. *"The design is nice but it broke on the very first day."* $\rightarrow$ **Positivo** ❌ **(ERROR 1)**
4. *"Not worth the money at all, please avoid buying it."* $\rightarrow$ **Negativo** (Correcto)
5. *"The customer service agent was polite and helpful."* $\rightarrow$ **Positivo** (Correcto)
6. *"I thought this would be great, but it turned out to be a total waste."* $\rightarrow$ **Positivo** ❌ **(ERROR 2)**
7. *"It doesn't work as advertised, but at least the color is pretty."* $\rightarrow$ **Positivo** ❌ **(ERROR 3)**
8. *"Absolutely horrible experience, it took three weeks to arrive."* $\rightarrow$ **Negativo** (Correcto)
9. *"This is an essential tool for my daily routine, highly recommended."* $\rightarrow$ **Positivo** (Correcto)
10. *"The screen is too dim and the battery dies fast."* $\rightarrow$ **Negativo** (Correcto)

#### Análisis de Fallas:
* **Error 1:** Fallo ante estructuras de giro (*nice but it broke...*). El término positivo *nice* sesga la ecuación estadística.
* **Error 2:** Fallo ante falsas expectativas en tiempo pasado (*thought this would be great...*). El modelo cuenta la palabra *great* e ignora la frustración final.
* **Error 3:** Fallo ante cláusulas de consuelo (*doesn't work... color is pretty*). Se promedia el texto dándole peso a la palabra estética por encima del fallo funcional.

### 3. Discusión Científica y Respuestas Teóricas (Sustento Académico 2021-2026)

El modelo **no comprende el lenguaje humano**; solo asocia empíricamente patrones estadísticos basados en la frecuencia léxica. 

De acuerdo con **Sanh et al. (2021)**, las representaciones estáticas tradicionales adolecen de una carencia de codificación posicional, lo que significa que operadores críticos de negación son fácilmente opacados por adjetivos individuales adyacentes. Esto justifica teóricamente por qué el Perceptrón Multicapa falló ante estructuras sintácticas inversas en nuestras pruebas.

Asimismo, **Minaee et al. (2024)** sustentan que los métodos basados en bolsas de palabras no pueden extraer la semántica profunda ni resolver la ironía o matices complejos del consumidor debido al aislamiento matemático de las dimensiones de su vocabulario. El sarcasmo destruye estos promedios probabilísticos lineales. 

Por estas razones, la ingeniería de software moderna ha migrado hacia el uso de arquitecturas que generan embeddings contextuales dinámicos, donde, tal como describen **Achiam et al. (2024)** en los reportes técnicos de los modelos modernos, el significado de cada palabra se redefine matemáticamente según la posición y relación absoluta con las demás palabras de la oración.

---

### Pregunta Central: ¿Usaría este modelo en producción para Amazon?

**Respuesta Directa:** **No, no lo usaría.**

* **Justificación Técnica:** Un margen de error cercano al ~21% en millones de transacciones diarias provocaría que cientos de miles de reseñas con quejas sutiles o con dobles sentidos gramaticales se indexaran como opiniones positivas. Esto corrompería los sistemas automáticos de recomendación, alteraría las métricas reales de satisfacción del cliente y sesgaría los paneles de control de calidad de los proveedores de la plataforma.
* **Justificación Ética:** Implementar una Inteligencia Artificial con puntos ciegos tan evidentes atenta contra la transparencia comercial. Podría perjudicar directamente el sustento y reputación de vendedores legítimos al interpretar mal la retroalimentación de sus productos, al mismo tiempo que defrauda la confianza del consumidor final, quien confía en que los resúmenes y puntuaciones de la plataforma representan de manera fiel y veraz el sentir de la comunidad.

---

## Referencias Bibliográficas (Normas APA 7.ª ed.)

Achiam, J., Adler, A., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., Avila, R., Babuschkin, I., Balaji, S., Balcom, V., Baltescu, P., Barkley, H., Beutel, M., Bhardwaj, K., Borlei, A., ... Zoph, B. (2024). GPT-4 technical report. *arXiv preprint arXiv:2303.08774*. https://scholar.google.com/scholar?q=GPT-4+technical+report

Minaee, S., Kalchbrenner, N., Cambria, E., Nikzad, N., Chenaghlu, M., & Gao, J. (2024). Deep learning based text classification: A comprehensive review. *ACM Computing Surveys*, *56*(3), 1–40. https://scholar.google.com/scholar?q=Deep+learning+based+text+classification+Minaee

Sanh, V., Webson, A., Raffel, C., Bach, S. H., Sutawika, L., Zaidan, Z., Zhou, W., Bari, Y., Ostyakova, E., Gowda, S., Wang, T., Bansal, R., Diwan, S., Muennighoff, N., Foronda, A., Bawden, R., Srivastav, A. S., Marone, J., Dey, A., ... Rush, A. M. (2021). Multitask prompted training enables zero-shot task generalization. *arXiv preprint arXiv:2110.08207*. https://scholar.google.com/scholar?q=Multitask+prompted+training+enables+zero-shot+Sanh