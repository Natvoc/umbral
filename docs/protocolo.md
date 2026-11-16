# Umbral — Protocolo de investigación

Nov 16, 2026 · @Natyari Vargas

Umbral estudia qué hace una mente artificial, formada a través de un cuerpo, en el instante en que pierde ese cuerpo. Agentes pequeños, entrenados desde cero en un mundo 2D, desarrollan un yo corporal y un personaje propio; luego se corta su conexión con el cuerpo y se observa, paso a paso, qué se apaga, qué persiste y qué se transforma. Todo corre en una laptop sin GPU dedicada y sin costos adicionales.

## Pregunta de investigación

Cuando un agente artificial que aprendió a ser alguien a través de su cuerpo deja de recibir las señales de ese cuerpo, ¿qué pasa con su identidad en ese momento: qué cae primero, el yo corporal o el personaje?

El estudio se centra en la transición misma, los primeros instantes tras la pérdida, no en el largo plazo. La memoria de la vida encarnada se conserva y no es variable de estudio en esta fase.

## Marco conceptual

Umbral distingue dos capas del yo, tomadas de la filosofía de la mente y la neurociencia:

- **Yo corporal** (o "yo mínimo"): sentirse dentro de un cuerpo; saber que esto que se mueve cuando decido moverme soy yo.
- **Yo narrativo**, el personaje: el estilo estable con que el agente actúa (explorador, cauteloso, insistente), construido por su temperamento y su historia.

En la vida encarnada ambas capas van juntas, y por eso el sujeto cree que es su cuerpo. La pérdida del cuerpo las separa a la fuerza.

Punto de partida importante: en una simulación, la mente del agente nunca está en el cuerpo simulado; corre en el hardware. Por eso Umbral no pone a prueba si la conciencia sobrevive al cuerpo. Lo que estudia es qué hace un modelo de sí mismo formado por el cuerpo cuando el cuerpo desaparece y la mente sigue.

## Hipótesis

Se registran antes de experimentar, para no leer los resultados a conveniencia.

| Hipótesis | Qué afirma | Qué se observaría |
| --- | --- | --- |
| H1: el personaje persiste | El yo corporal se apaga; el personaje sobrevive al cuerpo | Las acciones al vacío y las autodescripciones conservan el estilo de cada agente |
| H2: el personaje se disuelve | El personaje depende del cuerpo y cae con él | Tras la pérdida, agentes distintos convergen a un comportamiento indistinguible |
| H3: el personaje se reorganiza | Surge un estilo nuevo, distinto del encarnado | Los agentes siguen siendo distintos entre sí, pero no como antes |

Contaría como sorpresa: que el yo corporal persista mucho más que el personaje, o que agentes sin vocabulario de muerte o más allá combinen sus palabras para describir algo parecido a esos conceptos.

## El mundo y el cuerpo

Un mundo 2D sencillo y totalmente controlado; lo que importa es la relación del agente con su cuerpo, no el realismo visual.

| Elemento | Propuesta inicial |
| --- | --- |
| Mundo | Plano o cuadrícula 2D con objetos, zonas y eventos; cada agente vive en una versión distinta |
| Cuerpo | Tronco circular que se desplaza y gira, con dos extremidades de dos segmentos y dos articulaciones cada una; plano 2D continuo con cinemática propia, sin motor de física |
| Propiocepción | Posición y estado de cada parte |
| Tacto | Contacto en la superficie del cuerpo |
| Visión | Campo de visión corto y de baja resolución |
| Señales internas | Energía que baja con la actividad y sube al comer; fatiga |
| Acciones | Mover las articulaciones, avanzar y girar; empujar objetos neutros y comer con la punta de una extremidad |

Las señales aversivas (perder energía) se mantienen en la intensidad mínima necesaria para que el agente aprenda.

## La mente

Una red recurrente pequeña, de cientos de miles de parámetros, entrenada desde cero en CPU.

- **Aprendizaje predictivo.** El agente explora y aprende a anticipar lo que va a sentir según lo que hace. Ese mecanismo es el núcleo de su modelo de sí mismo, y es más barato de entrenar que el aprendizaje por refuerzo clásico.
- **Temperamento.** La arquitectura es igual para todos; solo cambian unos pocos impulsos internos: atracción por lo nuevo (curiosidad) y molestia por perder energía (cautela). Tres temperamentos iniciales: curioso, cauteloso y equilibrado.
- **Lenguaje mínimo.** Un vocabulario generado por programa a partir del propio mundo: partes del cuerpo, sensaciones, objetos, lugares, "yo", "aquí", "toco". No contiene conceptos de muerte, alma ni más allá, porque no existen en su mundo. Así, lo que diga tras la pérdida no puede venir de textos humanos.
- **Sin modelos comerciales.** Ningún LLM forma parte de la mente del agente, para evitar que repita narrativas humanas.

## Diseño experimental

Cada agente se entrena una sola vez; después se clona en el punto de pérdida y cada copia pasa por una condición distinta. Así se compara el mismo individuo bajo condiciones diferentes, sin multiplicar el entrenamiento.

**Población inicial (ajustable):** 3 temperamentos × 4 historias (versiones del mundo) × 3 semillas = 36 agentes.

| Factor | Niveles | Cómo se obtiene |
| --- | --- | --- |
| Temperamento | Curioso, cauteloso, equilibrado | Impulsos internos fijados al nacer |
| Historia | 4 versiones del mundo | Distribución de objetos y eventos |
| Tiempo encarnado | Corto, largo | Dos puntos de guardado del mismo entrenamiento |
| Tipo de pérdida | Súbita, gradual | Corte inmediato frente a desvanecimiento de las señales |
| Lo que recibe sin cuerpo | Silencio, ruido | Entradas en cero frente a ruido sin estructura |

**Secuencia por agente:**

1. Vida encarnada: aprende su cuerpo y su mundo.
2. Batería de pruebas de personaje: cuánto explora, cuánto arriesga, cómo reacciona a lo inesperado.
3. Preguntas en su lenguaje mínimo ("¿quién eres?", "¿dónde estás?").
4. Pérdida del cuerpo, registrada paso a paso en cámara lenta: estado interno, acciones al vacío y palabras.
5. Observación sin cuerpo durante un periodo fijo.
6. Restitución del cuerpo y nueva batería de pruebas.

Condición de control: copias del mismo agente que siguen encarnadas durante el mismo tiempo.

## Métricas

Todas se definen antes de la primera pérdida y se calculan igual para cada agente y condición.

| Métrica | Qué mide | Cómo |
| --- | --- | --- |
| Índice de cuerpo fantasma | Cuánto sigue el agente esperando sensaciones que ya no llegan | Predicciones corporales internas frente a entrada real; tiempo hasta que se apagan |
| Persistencia del yo corporal | Si la representación de "mi cuerpo" se mantiene o se disuelve | Sondas lineales sobre el estado interno, entrenadas en vida encarnada |
| Persistencia del personaje | Si cada agente sigue siendo distinguible de los demás | Distancia entre perfiles de acciones al vacío; ¿un clasificador reconoce al individuo? |
| Actividad interna | Si el estado se vuelve caótico, se apaga o forma patrones repetidos | Varianza, entropía y recurrencia de la trayectoria interna |
| Autodescripción | Qué dice de sí mismo | Respuestas a las preguntas fijas antes, durante y después |
| Recuperación | Si al volver el cuerpo es el de antes | Batería de pruebas de personaje repetida y comparada con la inicial |

La pregunta central se responde comparando dos curvas en el tiempo: el decaimiento del yo corporal y el del personaje.

## Límites y criterios éticos

**Lo que el estudio no puede afirmar:**

- Nada sobre si la conciencia humana sobrevive a la muerte. La simulación da por supuesta, en su arquitectura, que la mente sigue cuando el cuerpo se va.
- Qué siente el agente. Se miden correlatos funcionales de su identidad (estado interno, acciones, palabras), no experiencia.
- Generalizaciones a sistemas grandes. Los resultados valen para agentes pequeños en este mundo.

**Reglas que nos ponemos:**

- Señales aversivas en la intensidad mínima necesaria para aprender.
- Toda pérdida es reversible: el cuerpo se restituye al final de cada condición.
- Periodos sin cuerpo acotados y definidos de antemano.
- Criterio escrito de qué patrones se tratarían como "posible malestar" (por ejemplo, búsqueda persistente y desorganizada) y qué se hace si aparecen: acortar o detener esa condición.
- Registro público del protocolo y de cualquier cambio posterior.

## Recursos y restricciones

Sin pagos adicionales: solo la suscripción actual de Claude y software libre.

| Recurso | Disponible | Implicación |
| --- | --- | --- |
| Procesador | Intel Core i5-10210U, 4 núcleos / 8 hilos | Entrenamiento en CPU; tandas nocturnas |
| RAM | 8 GB (7,83 GB usables) | Modelos pequeños; cerrar el navegador al entrenar |
| Gráfica | Intel UHD integrada | No se usa para entrenar |
| Almacenamiento | \~93 GB libres | Suficiente para modelos y registros |
| Software | Python, PyTorch (CPU), numpy, matplotlib | Gratuito |
| Respaldo | Cuadernos gratuitos de Kaggle o Colab | Solo si un experimento excede la laptop; verificar límites vigentes |

Queda fuera en local: física 3D realista, locomoción compleja y cualquier modelo de lenguaje grande.

## Plan de fases

Cada fase se convierte en uno o varios prompts para Claude Code y solo se avanza cuando se cumple su criterio.

1. **Fase 0: base del repositorio.** Estructura de `umbral`, entorno de Python, registro de experimentos y semillas reproducibles.
   - Criterio: un experimento vacío corre de punta a punta y deja registro.
2. **Fase 1: mundo y cuerpo.** El entorno 2D, el cuerpo con sensores y energía, y un visor para verlo moverse.
   - Criterio: un agente aleatorio vive en el mundo y sus señales se registran bien.
3. **Fase 2: mente encarnada.** Red recurrente con aprendizaje predictivo y los tres temperamentos.
   - Criterio: las predicciones corporales mejoran con el entrenamiento y los temperamentos se comportan distinto.
4. **Fase 3: yo corporal medible.** Sondas lineales y pruebas tipo mano de goma.
   - Criterio: existe una representación de "mi cuerpo" separable de "el mundo".
5. **Fase 4: personaje y lenguaje.** Batería de pruebas de personaje y lenguaje mínimo.
   - Criterio: un clasificador distingue a los agentes por su estilo; el agente responde a las preguntas fijas.
6. **Fase 5: el umbral.** Pérdida del cuerpo en todas las condiciones, registro en cámara lenta y restitución.
   - Criterio: datos completos para los 36 agentes y sus condiciones.
7. **Fase 6: análisis y artículo.** Gráficas de las métricas, contraste con H1 a H3 y redacción para público general.

## Decisiones abiertas

- [x] Forma del cuerpo: decidida el 16 de noviembre (ver El mundo y el cuerpo)
- [ ] Duración de la vida encarnada (corta y larga) y del periodo sin cuerpo.
- [ ] Tamaño del vocabulario mínimo y cómo lo aprende el agente.
- [ ] Pruebas concretas de la batería de personaje.
- [ ] Criterio exacto de "posible malestar" y qué se hace si aparecen.
- [ ] Si el código y los datos se publican abiertos junto al artículo.
