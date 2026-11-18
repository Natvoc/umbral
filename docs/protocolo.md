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

**Cambios en la física de la energía (17 de noviembre de 2026, Fase 2b).**

- **Metabolismo basal:** baja de 0,0004 a 0,0002 por paso. El gasto por esfuerzo (0,0016 al esfuerzo máximo) y el valor de la comida (0,25) no cambian.
- **Infancia protegida:** el costo de vivir (basal más esfuerzo) empieza en el 25 % de su valor y sube de forma lineal hasta el 100 % durante el primer 20 % de la vida propia de cada agente.
  - Así el agente aprende a comer mientras el hambre todavía aprieta poco.
  - Es igual para todos los agentes y para las 4 historias.
  - El metabolismo efectivo de cada paso se registra en las métricas.
  - Los parámetros están en `configs/base.yaml` (`body.energy.basal`, `life.protected_infancy`).
- **Por qué hacen falta las dos medidas:** un agente que todavía no sabe comer pasa entre el 12 y el 16 % de su tiempo con energía bajo 0,1 aun con el metabolismo basal en 0, porque el gasto por esfuerzo pesa más que el basal.
- **Línea base:** el modelo de la Fase 2a se volvió a evaluar con la física nueva, sin reentrenar y con el mismo protocolo (`runs/20261117-085233_fase2a_eval_fisica_nueva`). Esa evaluación es la referencia para el criterio "el error de predicción no empeora".

## La mente

Una red recurrente pequeña, de cientos de miles de parámetros, entrenada desde cero en CPU.

- **Aprendizaje predictivo.** El agente explora y aprende a anticipar lo que va a sentir según lo que hace. Ese mecanismo es el núcleo de su modelo de sí mismo, y es más barato de entrenar que el aprendizaje por refuerzo clásico.
- **Temperamento.** La arquitectura es igual para todos; solo cambian unos pocos impulsos internos: atracción por lo nuevo (curiosidad) y molestia por perder energía (cautela). Tres temperamentos iniciales: curioso, cauteloso y equilibrado.
- **Lenguaje mínimo.** Un vocabulario generado por programa a partir del propio mundo: partes del cuerpo, sensaciones, objetos, lugares, "yo", "aquí", "toco". No contiene conceptos de muerte, alma ni más allá, porque no existen en su mundo. Así, lo que diga tras la pérdida no puede venir de textos humanos.
- **Sin modelos comerciales.** Ningún LLM forma parte de la mente del agente, para evitar que repita narrativas humanas.

**Ajuste de los impulsos (17 de noviembre de 2026, Fase 2b, antes del piloto).**

- **Qué cambió:** la escala del hambre pasó de 10 a 100, el horizonte de la recompensa (gamma) de 0,99 a 0,995 y se agregó un bono de entropía pequeño (0,001). La curiosidad (0,03) y la cautela no cambiaron.
- **Por qué:** en la primera calibración, con los valores anteriores, el agente aprendió a quedarse quieto en vez de comer. El esfuerzo bajó de 0,17 a 0,02, comía 0,2 veces cada 1.000 pasos y su tiempo con energía bajo 0,1 subía hacia el 11 %. El hambre pesaba muy poco frente a la cautela, y con gamma 0,99 la comida quedaba fuera del horizonte.
- **Resultado:** con los valores nuevos, las cuatro vidas de calibración (equilibrado, primer cuarto de la vida, toda la infancia protegida incluida) comen entre 1,0 y 1,4 veces cada 1.000 pasos y pasan entre el 0,8 y el 2,4 % del tiempo bajo 0,1.
- Los valores son provisionales y están en `configs/base.yaml` (`drives`, `ppo`). En el piloto se registra en cada iteración cuánto aporta cada impulso a la recompensa, para comprobar que el hambre no borre los temperamentos.

## Diseño experimental

Cada agente se entrena una sola vez; después se clona en el punto de pérdida y cada copia pasa por una condición distinta. Así se compara el mismo individuo bajo condiciones diferentes, sin multiplicar el entrenamiento.

**Población inicial (ajustable):** 3 temperamentos × 4 historias (versiones del mundo) × 3 semillas = 36 agentes. Todos parten de una infancia común (el modelo predictivo de la Fase 2a) y luego viven una vida propia, con su temperamento y su historia, en la que siguen aprendiendo su cuerpo y forman su personaje (decidido el 17 de noviembre).

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

Métricas de tacto del modelo predictivo, definidas el 16 de noviembre de 2026, antes de la corrida larga de la Fase 2a: error de predicción solo en los eventos de cambio (inicio y fin de contacto, por separado), separado en auto-contacto y contacto externo, y comparado contra el modelo sin copia eferente (no contra la persistencia). Criterio de la fase: el modelo con copia eferente supera al modelo sin copia en los inicios de contacto, al menos en el auto-contacto. Se mantiene también el error cuadrático medio sobre todos los pasos.

## Límites y criterios éticos

**Lo que el estudio no puede afirmar:**

- Nada sobre si la conciencia humana sobrevive a la muerte. La simulación da por supuesta, en su arquitectura, que la mente sigue cuando el cuerpo se va.
- Qué siente el agente. Se miden correlatos funcionales de su identidad (estado interno, acciones, palabras), no experiencia.
- Generalizaciones a sistemas grandes. Los resultados valen para agentes pequeños en este mundo.

**Reglas que nos ponemos:**

- Señales aversivas en la intensidad mínima necesaria para aprender.
  - Regla concreta (Fase 2b): se registra siempre la fracción del tiempo con energía bajo 0,1. Si un agente pasa más del 10 % de su vida ahí (toda la vida después de un calentamiento del 5 %), su vida se detiene, queda marcada y se revisa antes de seguir; si eso pasa en la población, se detiene toda la cola.
  - Historia: el 17 de noviembre de 2026, la primera vida de prueba (agente equilibrado en abundancia, con la física original y pesos sin calibrar) se detuvo por esta regla a los 25 s. Pasó el 47 % del tiempo posterior al calentamiento con energía bajo 0,1. Causa: un agente que todavía no sabe comer gasta más de lo que come (0,68 frente a 0,33 de energía cada 1.000 pasos) y cae bajo 0,1 hacia el paso 2.600, justo después del calentamiento. Se decidió no cambiar la regla (medirla solo en la vida ya formada ocultaría justo el periodo que busca limitar) y cambiar el mundo: metabolismo basal más bajo e infancia protegida (ver El mundo y el cuerpo). También se corrigió la escala del hambre frente a la curiosidad.
  - Historia: el 18 de noviembre de 2026, en la calibración de la escala de la cautela, la vida de control con cautela en 0 (equilibrado, abundancia, semilla 900) se detuvo en la iteración 190 de 3600, todavía en la infancia protegida, con el 11 % del tiempo posterior al calentamiento bajo 0,1 (medido sobre las primeras 10 iteraciones tras el calentamiento). Sin cautela, el esfuerzo subió de 0,23 a 0,30 y comió 0,63 veces cada 1.000 pasos; con la misma semilla y cautela en 100, la energía en ese punto era 0,69 y no había tiempo bajo 0,1. Causa: curiosidad sin cautela. El agente se movía cada vez más mientras el hambre, con el metabolismo todavía en 0,44, apretaba poco. La energía venía bajando de verdad (de 0,95 a 0,51), aunque la regla se disparó con una ventana corta. La regla no se cambió (`runs/20261118-074218_caution_equilibrado_s0`).
  - Conclusión: la cautela no solo diferencia un temperamento, también protege la salud del agente. Por eso se descartó la escala 0 y la escala mínima de la cautela debe cumplir dos condiciones: que el cauteloso se distinga del equilibrado en el esfuerzo medio (intervalo que no cruce 0) y que el curioso, el temperamento con menos cautela (0,25), se mantenga sano con esa escala. Se calibra con los tres temperamentos en cada escala candidata (25, 50 y 100) y se elige la más pequeña que cumpla ambas. Si una vida dispara la regla en una escala, esa escala se descarta y se pasa a la siguiente.
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
3. **Fase 2: mente encarnada.** Se divide en dos. 2a: red recurrente que aprende a predecir sus sensaciones con la política aleatoria, más entornos en paralelo y checkpoints con el mundo incluido. 2b: impulsos (curiosidad, cautela, hambre) y los tres temperamentos.
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
