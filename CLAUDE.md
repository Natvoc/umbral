# CLAUDE.md — reglas del proyecto Umbral

Umbral estudia agentes pequeños entrenados desde cero en un mundo 2D: cómo desarrollan un yo corporal y un personaje, y qué pasa con su identidad cuando pierden el cuerpo. **El diseño se rige por [`docs/protocolo.md`](docs/protocolo.md).** Si algo no está en el protocolo, se pregunta antes de inventarlo.

## Restricciones (no negociables)

- **Hardware:** laptop con Windows 64 bits, Intel i5-10210U (4 núcleos / 8 hilos), 8 GB de RAM y sin CUDA. Todo corre en CPU. Hay que cuidar la memoria y el tiempo de cómputo.
- **Sin costos:** nada de servicios de pago, APIs externas ni cuentas en la nube. **Ningún LLM ni modelo preentrenado forma parte del proyecto.** Todo se entrena desde cero.
- **Dependencias mínimas:** Python 3.11+, `numpy` (`<2.4` por Smart App Control), `torch` (CPU, desde `https://download.pytorch.org/whl/cpu`), `pyyaml`, `matplotlib` (solo para el visor) y `pytest`. Para agregar cualquier otra dependencia hay que preguntar primero.
- **Física:** cinemática propia escrita en numpy, sin motor de física.
- **Idioma:** código, identificadores y docstrings en inglés. README, CLAUDE.md y documentación en español.

## Reproducibilidad

- **Todo experimento es reproducible y se registra con `umbral.recorder.Recorder`**, normalmente a través de `umbral run`. No hay resultados válidos fuera de `runs/`.
- Toda corrida parte de `seed_everything` (lo hace `run_experiment`). La semilla y los hilos activos quedan en `meta.json`.
- **La aleatoriedad de numpy sale siempre del `np.random.Generator` explícito** que se recibe como `np_rng`. Nunca se usan las funciones globales `np.random.*`. Si un componente necesita azar de numpy, recibe el generador como argumento.
- `random` y `torch` tienen estado global por proceso. Antes de correr una copia de un agente (de `clone`) se llama a su `activate()`, y el `np_rng` que devuelve es el que se usa.
- Un checkpoint guarda modelos, optimizadores y **todos** los generadores aleatorios. Todo estado nuevo que afecte la trayectoria (entornos, buffers, contadores) tiene que entrar en el checkpoint, y debe haber un test que lo demuestre: reanudar tiene que dar exactamente lo mismo que no haberse detenido.
- Las copias de `clone` son idénticas, generadores incluidos. Cualquier diferencia entre copias debe venir solo de la condición experimental.
- `metrics.jsonl` no lleva horas ni otros datos no deterministas, para poder comparar corridas byte a byte.
- Antes de una corrida que vaya a citarse, se hace commit. `meta.json` registra `git_commit` y `git_dirty`.
- No se usa `set_num_interop_threads` fuera de `seeding.configure_threads`, porque solo puede llamarse una vez por proceso.

## Cuerpo, mundo y vínculo

- **Toda señal entre cuerpo y mente pasa por `BodyLink`** (`src/umbral/body/link.py`). La mente nunca lee el estado del cuerpo ni del mundo directamente: solo recibe `obs` y entrega acciones.
- **`reward` es siempre `0.0`, para siempre.** Los impulsos (curiosidad, cautela) viven dentro de la mente (Fase 2), no en el entorno.
- Las señales aversivas se mantienen en el mínimo necesario. No hay muerte: con energía en cero, el cuerpo se mueve más lento y débil.
- El modo `noise` usa media y desviación estándar congeladas en el instante en que se activa, calculadas solo con la vida encarnada (segmentos en `intact`). El modo `noise` motor no está definido: el protocolo solo lo define para el lado sensorial.
- Con `world.freeze_when_disembodied`, la desconexión total pausa el mundo y el cuerpo. Las pérdidas parciales no lo pausan.
- La acción intentada se registra siempre. Las acciones al vacío son una métrica central.
- La forma de la observación (107 valores) es fija y está documentada en `src/umbral/body/sensors.py`. Cambiarla rompe la comparación entre agentes: no se cambia sin decidirlo de forma explícita.
- Los parámetros de fatiga en `base.yaml` son **provisionales**, todavía sin calibrar.
- Los checkpoints (formato v2) incluyen el mundo completo (`env`), la política (`policy`) y el estado de la mente (`mind_state`). Todo estado nuevo que afecte la trayectoria tiene que entrar ahí, con su test.

## Entornos en paralelo

- `BatchedUmbralEnv` es la única implementación de la física y los sensores; `UmbralEnv` es un lote de tamaño 1.
- **Equivalencia exacta**: el entorno `i` de un lote debe dar la misma trayectoria que uno individual (`tests/test_batched.py`). En los núcleos en lote: solo operaciones elemento a elemento y reducciones sobre el último eje; nada de multiplicación de matrices ni de reducciones sobre el eje del lote. Los objetos se rellenan hasta una capacidad fija. Toda optimización se valida con ese test.
- Cada entorno tiene sus propios generadores (mundo, vínculo, política). Nunca se comparte un generador entre entornos.

## La mente (Fase 2a)

- La GRU recibe `[o_t, a_{t-1}]` y la cabeza recibe `[h_t, a_t]` (documentado en `src/umbral/mind/model.py`). Siempre se usa la acción **intentada**, nunca la aplicada.
- El modelo sin copia eferente es idéntico, pero con acciones en cero. Los dos se entrenan juntos, con los mismos datos y los mismos pesos iniciales, sin compartir parámetros, optimizadores ni gradientes. La política no depende de nada aprendido.
- Las desviaciones estándar que normalizan los cambios se estiman una sola vez (calentamiento con semilla fija), se congelan y se guardan en el `state_dict`. Son las mismas para los dos modelos.
- Semillas disjuntas: entrenamiento en `[0, 2**31)`, evaluación desde `2**31` y normalización desde `2**32`.

## La mente que decide (Fase 2b)

- **El modelo de sí mismo no se deforma por la recompensa.** El codificador, el GRU y la cabeza de predicción se entrenan solo con la pérdida de predicción. El actor y el crítico leen `h_t` con el gradiente detenido (`src/umbral/mind/agent.py`). Hay un test que lo comprueba; no se rompe.
- La recompensa sale solo de los impulsos (`src/umbral/mind/drives.py`), calculados con la observación y la acción intentada. La curiosidad excluye la propiocepción y el auto-contacto. La cautela usa el costo nominal del esfuerzo. El hambre modula solo a la curiosidad, que se multiplica por `g(E)` (`drives.gate_low`), igual para todos los temperamentos. La cautela no se modula: es el freno que protege la energía.
- **La infancia común tiene dos partes:** aprender el cuerpo (el modelo de la 2a, `agent.infancy`) y aprender a comer (`experiment=common_feeding`: con el temperamento equilibrado y una escala de cautela propia, provisional, de 50 (`feeding.*`), las 4 historias mezcladas y registradas por separado, misma regla ética, semillas propias desde `3·2³¹`). Su costo de vivir sube **por competencia**, no por calendario: solo sube si los cuerpos se sostienen, y nunca baja. Las vidas conservan su rampa por calendario. Termina con un criterio fijado en el protocolo y deja `common_start.pt`.
- Todas las vidas parten de `agent.start` (`common_start.pt`), idénticas: solo cambian el temperamento y la semilla, se reinicia la normalización de la recompensa y el `log_std` del actor vuelve a su valor inicial (`life.reset_action_std`: la infancia deja qué hacer, cada vida explora cómo). Hay tests que lo comprueban. La comprobación de predicción sigue comparando con la 2a (`agent.infancy`). Los pesos de los impulsos y de los temperamentos son provisionales y están en `base.yaml`.
- **Regla ética:** más del 10 % de la vida (después del 5 % inicial) con energía bajo 0,1 detiene la vida (`flagged`) y la cola de la población. No se relaja la regla: si se dispara, se avisa a la investigadora y el ajuste va en el mundo.
- La infancia protegida (`life.protected_infancy`) y el metabolismo basal están documentados en el protocolo. Cualquier cambio en la física de la energía obliga a rehacer la línea base del modelo de la 2a (`eval.physics=current`).
- Semillas: la batería de comportamiento usa `2**31 + 3_000_000 + i`; el criterio de la fase de comer, `2**31 + 4_000_000 + i`.
- La población vive en una carpeta fija (`population.dir`) con `population.json`; cada agente corre en su propio proceso. **Un agente a la vez** (`population.parallel: 1`, y `CALIB_PARALLEL=1` en las calibraciones): con 2 en paralelo la memoria se agotó dos veces.

## Fases

- **Ninguna fase avanza sin cumplir su criterio de cierre**, verificado con comandos y tests, no por suposición.
- Fase 0 (base del repositorio): cerrada cuando `umbral run --config configs/base.yaml experiment=noop` deja una carpeta completa en `runs/` y `pytest` pasa.
- Fase 1 (mundo y cuerpo): cerrada cuando `umbral run --config configs/base.yaml experiment=random_agent` deja métricas, `signals.npz` y `run.gif`, y todos los tests pasan.
- Fase 2a (modelo predictivo): equivalencia del lote probada y rendimiento registrado; reanudar y clonar con el mundo probado; una corrida completa de `train_world_model`; en la evaluación, el modelo supera claramente a la persistencia en todos los canales, y al modelo sin copia eferente al menos en propiocepción y tacto; todos los tests pasan.
- Fase 2b (impulsos y temperamentos): intervalos de confianza de la 2a en el README; en el piloto (3 temperamentos × abundancia × 1 semilla), los temperamentos se distinguen en la batería de comportamiento (con tamaños de efecto e intervalos), los agentes comen y respetan la regla ética, y el error de predicción no empeora respecto de la 2a; `train_population` listo y probado, sin lanzar.
- Las corridas largas (de horas) no se lanzan sin el visto bueno de la investigadora.
- **Las corridas largas (piloto, población) se lanzan desde una terminal de PowerShell externa**, abierta por la investigadora, fuera de la sesión de Claude Code, para que no las corte el mecanismo que mata procesos en segundo plano cuando falta memoria. Claude prepara el comando exacto (con `.venv\Scripts\python.exe` y la reanudación de la carpeta de la población), dice qué archivo mirar para saber si terminó o se detuvo, y lee los resultados cuando la investigadora avisa.
- En esta laptop, `umbral.exe` está bloqueado por Smart App Control: se usa `python -m umbral ...`, que es equivalente.

## Estructura

- `src/umbral/`: `config`, `seeding`, `recorder`, `checkpoint` y `cli`.
- `src/umbral/experiments/`: experimentos registrados con `@register("nombre")`.
- `world/`: `arena` (geometría en lote y objetos), `histories` (las 4 historias), `batched` (`BatchedUmbralEnv`), `env` (`UmbralEnv`, al estilo Gymnasium) y `render` (visor y GIF).
- `body/`: `body` (cinemática, movimiento, energía y fatiga), `sensors` (sensores y distribución de la observación) y `link` (`BodyLink`).
- `mind/`: `model` (mente predictiva), `policy` (política OU), `training` y `report` (curvas y orden de aprendizaje).
- `metrics/`: se llena en fases posteriores.
- `configs/base.yaml`: toda clave que se pueda sobrescribir desde la CLI tiene que existir aquí.
- `tests/`: `pytest`.
