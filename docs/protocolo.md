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

**El hambre modula la curiosidad (19 de noviembre de 2026, Fase 2b).**

- **Qué es:** con la energía que la mente observa en la interocepción, `g(E) = clip((E − E_low) / (E* − E_low), 0, 1)`, con `E_low = 0,2` (provisional, `drives.gate_low`) y `E* = 0,8`. La recompensa de curiosidad se multiplica por `g(E)`. La de hambre y la de cautela no cambian.
- **Efecto:** con energía alta, cada temperamento se expresa entero. A medida que baja la energía, la curiosidad se apaga para todos y manda el hambre, como en los animales, donde la necesidad urgente le gana a la exploración.
- **Es arquitectura compartida:** igual para los tres temperamentos. El temperamento sigue siendo la proporción entre curiosidad y cautela.
- **Por qué:** la calibración de la cautela del 18 de noviembre mostró que el curioso se agotaba. La curiosidad, normalizada por su media móvil, estaba "encendida" siempre con la misma fuerza y el hambre se hacía sentir cuando ya era tarde.
- **Primera versión, descartada:** el mismo día, `g(E)` modulaba la curiosidad **y la cautela**. Falló porque la cautela es el freno que protege la energía (ya lo había mostrado la vida con cautela en 0): al multiplicarla por `g(E)`, el freno se soltaba justo cuando la energía bajaba. En el curioso con escala 25, mientras `g(E)` caía de 1,00 a 0,52, el esfuerzo subió de 0,20 a 0,28 y la cautela pasó a cobrar hasta un 40 % menos por unidad de esfuerzo; sin modulación, esa misma vida se mantenía en 0,24 y sana. Con la primera versión, el curioso disparó la regla ética con las escalas 25 y 50 (semilla 901, en la iteración 190). Se corrigió para modular solo la curiosidad, y las vidas de la primera versión quedaron marcadas `superseded` en su `meta.json`.
- **Lo que no se confirmó:** la hipótesis de que la primera versión había empeorado las cosas al soltar el freno **no se confirmó**. Con la versión corregida, en la que la cautela cobra entera, el curioso de la escala 25 (semilla 901) volvió a disparar la regla ética en la iteración 190: energía media 0,49, percentil 10 de 0,05 y mínimo de 0,00, con un esfuerzo de 0,28 a 0,30, mientras la parte de la cautela en la recompensa subía del 6 al 10 %. **La escala 25 falló con las dos versiones de la modulación**: es demasiado débil para frenar al curioso, con o sin modulación. La corrección se mantiene porque tiene sentido por sí misma: el hambre apaga la exploración, no el freno.
- **Un solo cambio a la vez:** la infancia protegida y los pesos de los temperamentos quedan como estaban. `g(E)` se registra en cada iteración junto con la contribución de cada impulso.
- **Los cuerpos uno por uno:** la regla ética mide sobre todos los cuerpos en paralelo, y la media puede verse sana mientras algunos están agotados. Al detenerse, el curioso de la escala 25 tenía una energía media de 0,53, pero 4 de sus 32 cuerpos estaban bajo 0,1 (uno en 0,00) y el percentil 10 era 0,07. Desde entonces, cada vida registra en cada iteración, y por fase en su resumen, la energía mínima y el percentil 10 entre los cuerpos.

**La infancia común, en dos partes (21 de noviembre de 2026, Fase 2b).**

- **Qué es:**
  1. *Aprender el cuerpo:* el modelo predictivo de la Fase 2a, sin cambios (`agent.infancy`).
  2. *Aprender a comer (nueva):* una sola fase compartida, igual para todos los agentes (`experiment=common_feeding`). Parte del checkpoint de la 2a con un actor y un crítico nuevos.
- **Cómo es la fase de comer:**
  - **Se vive con el temperamento equilibrado** (decidido el 21 de noviembre; `feeding.temperament`). Usa sus pesos (curiosidad 0,5 y cautela 0,5) y una escala de cautela propia de la infancia de **50** (`feeding.caution_scale`, **provisional**: es la escala con la que el curioso y el cauteloso se separaron bien en la batería). La curiosidad se modula con `g(E)`, como en las vidas.
    - **Por qué cambió:** en la versión anterior el único impulso era el hambre. El agente aprendió a comer sin pasar hambre, pero con un esfuerzo creciente (de 0,28 a 0,37) y con la variación de la acción reducida a menos de la mitad (×0,46). Además, la escasez impidió completar la rampa (factor final 0,915). El punto de partida que dejaba era intenso y rígido. Se parte de nuevo desde el checkpoint de la 2a, para no heredar esa política. La corrida de solo hambre queda como dato, marcada `discarded_as_start` (ver el historial ético).
    - **Sesgo conocido:** el punto de partida es el temperamento equilibrado, así que el equilibrado empieza más cerca de su carácter final que el curioso y el cauteloso. La batería del punto de partida lo medirá, y la interpretación debe tenerlo en cuenta.
    - Con `feeding.temperament` nulo, la fase vuelve a ser de solo hambre. Es la versión del 21 de noviembre, que se conserva para poder reproducirla.
  - El modelo predictivo sigue aprendiendo, solo con la pérdida de predicción.
  - Los 32 cuerpos se reparten entre las 4 historias (8 en cada una), para que comer no dependa de un mundo en particular.
  - **El costo de vivir sube por competencia, no por calendario** (decidido el 21 de noviembre, tras la detención de la primera corrida; ver el historial ético). El factor de costo empieza en 0,25, como en una vida. En cada iteración solo sube, con la pendiente máxima de la rampa de una vida (0,75 / 720 por iteración), si en las últimas 20 iteraciones, sobre los 32 cuerpos, la fracción bajo 0,1 es menor que el 1 % y el percentil 10 de la energía es ≥ 0,3. Si no se cumple, el factor se mantiene; nunca baja. Llega como mínimo a costo completo en 720 iteraciones, pero puede tardar más. El factor se registra en cada iteración (`cost_factor`, y `ramp_competent` indica si se cumplía la condición). Es un ajuste del mundo, no de la regla: el mundo se mantiene amable mientras el agente aprende y se vuelve exigente solo cuando demuestra que puede sostenerse. **Solo se aplica a la fase de comer.** Las vidas propias conservan su rampa por calendario, porque ya parten de un agente que sabe comer.
  - La duración nominal es de 3.600 iteraciones. La regla ética se aplica igual: si se dispara, se detiene y se avisa a la investigadora.
  - **Por historia:** la energía, la comida y la fracción bajo 0,1 de cada historia se registran en cada iteración. Los reinicios escalonados de los episodios se reparten de forma alternada entre las historias, para que la edad del episodio de un cuerpo no dependa de su historia. En la primera corrida iban en el mismo orden que los bloques de historia y confundían los dos efectos. En las vidas, que tienen una sola historia, el escalonamiento no cambia.
  - Sus semillas son propias: los entornos de entrenamiento salen de `[3·2³¹, 4·2³¹)`, disjunto del entrenamiento de las vidas, de la evaluación y de la normalización. El criterio de término usa `2³¹ + 4.000.000 + i`.
- **Por qué:** con la modulación por hambre corregida, las tres escalas de la cautela (25, 50 y 100) fallaron en la misma ventana, las iteraciones 150 a 250, y con temperamentos distintos (curioso en 25 y 100, equilibrado en 50). Las vidas que pasaban esa ventana terminaban sanas y bien diferenciadas. La causa común es que el actor empezaba sin saber comer mientras subía el costo de vivir. El problema estaba en el punto de partida de la conducta, no en la cautela.
- **Criterio de término (fijado antes de entrenar):**
  - **Cómo se evalúa:** solo cuando el factor de costo llegó al 100 %, desde la primera iteración con costo completo y después cada 100 iteraciones. Con costo completo, energía inicial llena, política estocástica, sin aprendizaje y con semillas de evaluación fijas. Son 8 cuerpos por historia, durante un episodio de 4.096 pasos.
  - **Cuándo pasa:** cada una de las 4 historias debe cumplir, por separado, fracción del tiempo bajo 0,1 < 2 %, fracción bajo 0,3 < 10 % y comida ≥ 1 cada 1.000 pasos.
  - **Al cumplirse** se guarda `common_start.pt`, el **punto de partida único de todas las vidas**.
  - **Tope:** 3.600 iteraciones o 3 horas de reloj, lo que llegue primero (2 horas hasta el 21 de noviembre). Si se alcanza sin que el factor llegue al 100 % o sin cumplir el criterio, se detiene (`timeout`) y se revisan los datos.
- **Qué hereda cada vida** (`agent.start`): la mente con su normalización congelada, el actor, el crítico (con su `log_std`), los dos optimizadores y la media de la curiosidad. Todo es idéntico al punto de partida (hay un test que lo comprueba). Solo cambian el temperamento y la semilla. La normalización de la recompensa (`RewardScaler`) se reinicia igual para todos, porque medía los retornos de otra mezcla de impulsos y no es parte del modelo de sí mismo. Todo lo demás de la vida queda igual: su infancia protegida (la misma rampa), la modulación de la curiosidad, los pesos de los temperamentos, la regla ética, las medidas y los checkpoints de vida corta y vida larga.
- **Se reporta:** la curva del criterio por historia y el `log_std` del actor en `common_start.pt` frente al de su inicialización. Si la fase de comer redujo mucho la exploración, a los temperamentos podría costarles separarse. Solo se reporta, sin ajustar nada por eso.
- **La comprobación de predicción** sigue comparando con el modelo de la 2a (`agent.infancy`).

## Diseño experimental

Cada agente se entrena una sola vez; después se clona en el punto de pérdida y cada copia pasa por una condición distinta. Así se compara el mismo individuo bajo condiciones diferentes, sin multiplicar el entrenamiento.

**Población inicial (ajustable):** 3 temperamentos × 4 historias (versiones del mundo) × 3 semillas = 36 agentes. Todos parten de una infancia común y luego viven una vida propia, con su temperamento y su historia, en la que siguen aprendiendo su cuerpo y forman su personaje (decidido el 17 de noviembre).

**La infancia común tiene dos partes (desde el 21 de noviembre de 2026):** aprender el cuerpo (el modelo predictivo de la Fase 2a) y aprender a comer (una fase compartida, en las 4 historias, vivida con el temperamento equilibrado y una escala de cautela provisional de 50; ver La mente). Su resultado, `common_start.pt`, es el único punto de partida de todas las vidas.

- **Por qué la infancia se vive con el equilibrado:** con solo hambre, el agente aprendió a comer, pero con un esfuerzo creciente y la variación de la acción reducida a menos de la mitad (×0,46), y la escasez impidió completar la rampa.
- **Sesgo conocido:** el equilibrado empieza más cerca de su carácter final que el curioso y el cauteloso. La batería del punto de partida mide cuánto, y la comparación entre temperamentos debe leerse teniéndolo en cuenta.

- **La duración de la infancia común depende del aprendizaje:** el costo de vivir de la fase de comer sube por competencia, no por calendario (ver La mente). Esto no afecta la comparación entre temperamentos, porque es una sola fase compartida por todos.
- **Consecuencia para la interpretación:** el punto de partida de la conducta es compartido. Antes de vivir, la política no depende del temperamento (el temperamento solo cambia la recompensa), así que los tres temperamentos se comportan igual. Las diferencias de carácter que aparezcan vienen, entonces, del temperamento y de la vida propia, no de cómo cada agente aprendió a comer.
- **Para medirlo:** la batería de comportamiento se corre una vez en el punto de partida común. Esa medición es la línea base de los tres temperamentos, y se compara con el final de la vida corta y de la vida larga, para ver cuánto del carácter sale de la vida propia.
- La calibración de la cautela se repite desde este punto de partida, con las mismas escalas (25, 50 y 100), los tres temperamentos, 2 semillas y las mismas tres condiciones. Las vidas de calibración anteriores quedan marcadas `superseded`.

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
  - Resultado (18 de noviembre de 2026, semilla 900, abundancia, 900 iteraciones): ninguna escala cumplió las dos condiciones. Con 25, el curioso se mantuvo sano (0,9 % bajo 0,1), pero el cauteloso no se distinguió del equilibrado en el esfuerzo (g = −0,76 [−5,80; +0,15]). Con 50 y 100, el cauteloso sí se distinguió (g ≈ −18), pero el curioso disparó la regla en las iteraciones 190 y 200 (12,6 % y 10,0 % bajo 0,1 tras el calentamiento), en plena infancia protegida (metabolismo 0,44), con esfuerzo alto (0,24–0,26), comida en descenso y la curiosidad aportando el 64–74 % de su recompensa. Además, el cauteloso con 100 tendió a la quietud (esfuerzo de 0,18 a 0,06). La serie se detuvo y la decisión queda en manos de la investigadora.
  - Decisión de la investigadora: el problema es de la arquitectura de los impulsos, no de la rampa ni de los pesos. Se agregó la modulación por hambre (ver La mente) y se repite la calibración.
  - **Calibración con la modulación por hambre (condiciones escritas antes de lanzar).** Se corrió primero con la versión que modulaba también la cautela y se detuvo: fallaron las escalas 25 y 50 (ver La mente). Se repite desde cero, con las mismas condiciones, modulando solo la curiosidad. Escalas 25, 50 y 100; los tres temperamentos; 2 semillas por combinación (900 y 901); abundancia; las primeras 900 iteraciones de la vida (toda la infancia protegida y 180 iteraciones de adultez); 2 agentes en paralelo con 2 hilos cada uno. Una escala pasa si cumple las tres condiciones:
    1. El cauteloso se distingue del equilibrado en el esfuerzo medio de la batería (8 episodios por agente, en la iteración 900). Se juntan los 16 episodios de cada temperamento (2 semillas): la g de Hedges cauteloso frente a equilibrado debe ser negativa con su intervalo del 95 % entero bajo 0, y además negativa en cada semilla por separado.
    2. Ninguna vida dispara la regla ética.
    3. Ningún temperamento se paraliza: en las últimas 100 iteraciones (801–900), energía media ≥ 0,5 y comida ≥ 0,5 cada 1.000 pasos, en cada una de las 6 vidas.
  - Resultado (19 y 21 de noviembre de 2026): **ninguna escala cumplió.** Las tres se descartaron por la condición 2, siempre al poco de terminar el calentamiento y en plena infancia protegida:
    - Escala 25: el curioso (semilla 901) disparó la regla en la iteración 190 (11,9 % bajo 0,1 tras el calentamiento; energía media 0,49, percentil 10 de 0,05, esfuerzo 0,28–0,30).
    - Escala 50: el equilibrado (semilla 900) la disparó en la iteración 250 (10,06 %; energía media 0,53, percentil 10 de 0,05, mínimo 0,00, esfuerzo 0,23–0,29, comida 0,2–1,2 cada 1.000 pasos). Las otras cuatro vidas que terminaron sus 900 iteraciones estaban sanas en las iteraciones 801–900: los curiosos con energía 0,77 y 0,81 y comida 2,5 y 3,7; los cautelosos con energía 0,75 y 0,78 y comida 1,5 y 1,6. Todas tuvieron en ese tramo algún cuerpo con energía 0,00.
    - Escala 100: el curioso (semilla 900) la disparó en la iteración 190 (10,75 %; energía media 0,59, percentil 10 de 0,08 en la iteración 180, esfuerzo 0,29–0,31).
    - Las vidas que quedaban en cada escala descartada se detuvieron a propósito (`stopped`). La serie se detuvo y la decisión queda en manos de la investigadora.
    - Batería de las cuatro vidas sanas de la escala 50, a pedido de la investigadora (iteración 900, 8 episodios por vida, 16 por temperamento): el cauteloso se distingue del curioso con claridad en el esfuerzo (0,112 frente a 0,341; g = −6,5 [−8,2; −6,2]; también en cada semilla), la distancia, el área, la comida y los contactos con neutros. Ninguno de los 32 episodios llegó a energía 0 ni pasó tiempo bajo 0,1. La reacción a la sonda no los distingue (intervalos que cruzan 0). La condición 1 (cauteloso frente a equilibrado) no se pudo evaluar: ningún equilibrado llegó a la iteración 900. Datos en `runs/calib4/battery_s50_healthy.json`.
  - Se elige la escala más pequeña que cumpla las tres. Si una vida dispara la regla, la serie de esa escala se detiene, la escala se descarta y se pasa a la siguiente. Si ninguna pasa, se detiene todo y se informa, sin probar otros ajustes.
  - Historia: el 21 de noviembre de 2026, la primera fase de comer de la infancia común (`runs/20261121-112141_fase2b_infancia_comer`, semilla 0, 32 cuerpos repartidos en las 4 historias) se detuvo por la regla en la iteración 300 de 3.600, con el 10,2 % del tiempo posterior al calentamiento bajo 0,1. Todavía estaba en la rampa (metabolismo 0,56) y nunca llegó a la primera evaluación del criterio (iteración 720). La energía media bajó de 0,94 a 0,54, la comida se quedó en 0,4–1,0 cada 1.000 pasos y el esfuerzo subió de 0,20 a 0,27, solo con hambre. En el checkpoint de la detención, la energía media por historia era 0,60 (abundancia), 0,13 (escasez), 0,51 (corredores) y 0,69 (oasis). Esta foto está confundida con la edad del episodio de cada cuerpo, porque el escalonamiento de los reinicios va en el mismo orden que los bloques de historia. La regla no se cambió y la decisión queda en manos de la investigadora.
  - Decisión de la investigadora: el problema es la economía de la energía. Con costo completo, un agente ingenuo gasta más de lo que encuentra, y la rampa subía por calendario, aprendiera o no. Hay una pista en la calibración del 18 de noviembre: con cautela, el equilibrado pasó esa ventana en las 4 historias porque gastaba poco (esfuerzo de 0,10 a 0,18); aquí, sin cautela, el esfuerzo se quedó en 0,25. Por eso, en la fase de comer el costo de vivir sube por competencia y no por calendario (ver La mente), y se registra cada historia en cada iteración, con los reinicios escalonados repartidos entre las historias. La regla ética no cambió. Las vidas propias conservan su rampa.
  - Resultado (21 de noviembre de 2026, `runs/20261121-121257_fase2b_infancia_comer_competencia`, semilla 0): la fase llegó a las 3.600 iteraciones sin que el factor de costo alcanzara el 100 % (`timeout`, factor final 0,915). Por eso no hubo evaluación del criterio ni `common_start.pt`.
    - **Fue sana:** el 3,8 % del tiempo posterior al calentamiento bajo 0,1 y el 6,4 % bajo 0,3, con una energía media de 0,73.
    - **Aprendió a comer:** pasó de 0,3–0,6 a 2,2–3,6 veces cada 1.000 pasos según la historia.
    - **Gastó cada vez más:** el esfuerzo subió de 0,28 a 0,37.
    - **La condición de competencia se cumplió cada vez menos:** del 30 % de las iteraciones al principio al 6–14 % al final. Lo que más frenaba era la fracción bajo 0,1, que debía ser menor que el 1 % en la ventana.
    - **Por historia, en las iteraciones 3.201–3.600:**

      | Historia | Energía media | Tiempo bajo 0,1 | Comida cada 1.000 pasos |
      |---|---|---|---|
      | Abundancia | 0,84 | 0,1 % | 3,6 |
      | Escasez | 0,62 | 10,8 % | 2,2 |
      | Corredores | 0,74 | 3,3 % | 2,8 |
      | Oasis | 0,74 | 7,1 % | 3,3 |

    - **La exploración se redujo a menos de la mitad:** la desviación de la acción pasó de 0,607 a 0,276 (×0,46).
    - La serie se detuvo y la decisión queda en manos de la investigadora.
  - Decisión de la investigadora: la corrida cumplió parte de su objetivo (aprender a comer sin pasar hambre), pero el punto de partida que dejaría es intenso y rígido. Queda marcada `discarded_as_start`. La infancia común se vive ahora con el temperamento equilibrado (escala de cautela 50, provisional) y parte de nuevo desde la 2a. Se mantienen la rampa por competencia, el criterio, la regla, el registro por historia, las semillas y la mezcla de historias. El tope pasa a 3.600 iteraciones o 3 horas.
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
