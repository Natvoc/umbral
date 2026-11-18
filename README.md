# Umbral

Umbral es un proyecto de investigación. Agentes pequeños, entrenados desde cero en un mundo 2D, desarrollan un "yo corporal" y un personaje propio. Después se les quita el cuerpo y se observa, paso a paso, qué pasa con su identidad. Todo corre en CPU, en una laptop, sin servicios externos ni modelos preentrenados. El diseño del estudio está en [`docs/protocolo.md`](docs/protocolo.md).

**Estado: Fase 2b en curso.** Ya existen la base (configuración, semillas, registro, checkpoints y CLI), el mundo 2D con sus 4 historias, el cuerpo con sus sensores y su energía, el vínculo cuerpo–mente (`BodyLink`), un visor y los entornos en paralelo (`BatchedUmbralEnv`). Los checkpoints incluyen el mundo. La mente tiene un modelo predictivo (Fase 2a) que aprende a anticipar lo que va a sentir según lo que hace. En la Fase 2b el agente empieza a decidir: un actor y un crítico leen el estado del modelo predictivo, guiados por tres impulsos internos (hambre, curiosidad y cautela), cuyos pesos definen tres temperamentos.

## Instalación (Windows, PowerShell)

Requiere Python 3.11 o superior.

```powershell
cd umbral
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # torch solo CPU, primero
pip install -e ".[dev]"                                              # umbral + numpy, pyyaml, matplotlib, pytest
```

Si PowerShell bloquea `Activate.ps1`, ejecuta una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Uso

Para correr el experimento de prueba:

```powershell
umbral run --config configs/base.yaml experiment=noop
```

Para ver un cuerpo sin mente viviendo en el mundo (2000 pasos de acciones aleatorias, con GIF):

```powershell
umbral run --config configs/base.yaml experiment=random_agent
umbral run --config configs/base.yaml experiment=random_agent world.history_id=3 random_agent.steps=5000
```

Para la mente predictiva (Fase 2a):

```powershell
python -m umbral run --config configs/base.yaml experiment=bench_env           # pasos-entorno/s con N = 1, 8, 16, 32
python -m umbral run --config configs/base.yaml experiment=train_world_model   # entrena los dos modelos (con y sin copia eferente)
python -m umbral run --config configs/base.yaml experiment=eval_world_model eval.checkpoint=runs/<corrida>/checkpoints/update_XXXXXXX.pt
```

`train_world_model` se puede reanudar con `run.resume_from=<checkpoint>`. El log muestra el tiempo restante estimado.

Para la mente que decide (Fase 2b):

```powershell
python -m umbral run --config configs/base.yaml experiment=live_agent agent.temperament=curioso agent.history=0 seed=1
python -m umbral run --config configs/base.yaml experiment=behavior_battery behavior.checkpoint=runs/<vida>/checkpoints/life_long.pt
python -m umbral run --config configs/base.yaml experiment=agent_report "report.agents=[runs/<vida1>, runs/<vida2>]"
python -m umbral run --config configs/base.yaml experiment=train_population population.max_agents=6
```

- **`live_agent`:** una vida propia, que parte de la infancia común (`agent.infancy`, el modelo final de la 2a). Guarda `life_short.pt` a la mitad, `life_long.pt` al final y checkpoints intermedios para reanudar. Al terminar, deja `behavior.json` (la batería) y `prediction_check.json` (el modelo predictivo del agente frente al de la infancia).
- **`agent_report`:** compara agentes, con medias e intervalos por bootstrap, g de Hedges entre pares, `behavior.png` y `life_curves.png`.
- **`train_population`:** pone en cola los 36 agentes en una carpeta fija (`population.dir`) con su índice `population.json`. Se puede reanudar: salta los agentes terminados y retoma los interrumpidos desde su último checkpoint. `population.max_agents` limita cuántos corren por sesión y `population.parallel`, cuántos a la vez.
- **Regla ética:** si un agente pasa más del 10 % de su vida (después del 5 % inicial) con energía bajo 0,1, su vida se detiene con `life_status: flagged` y deja `flagged_summary.json`. En la población se detiene toda la cola.

Las 4 historias (`world.history_id`) son `0` abundancia, `1` escasez, `2` corredores y `3` oasis. Están definidas en `configs/base.yaml`, bajo `world.histories`.

Cualquier valor de `configs/base.yaml` se puede sobrescribir con `clave=valor`, usando puntos para las secciones:

```powershell
umbral run --config configs/base.yaml experiment=noop seed=3 run.name=prueba
```

Para reanudar desde un checkpoint:

```powershell
umbral run --config configs/base.yaml run.resume_from=runs/<carpeta>/checkpoints/step_000100.pt
```

**Nota sobre Smart App Control.** Si Windows bloquea `umbral.exe` ("Una directiva de Control de aplicaciones bloqueó este archivo"), usa el comando equivalente `python -m umbral run --config configs/base.yaml experiment=noop`. Por el mismo motivo, numpy está fijado en `<2.4`: los binarios de `numpy.random` 2.5.x también quedan bloqueados.

`umbral list` muestra los experimentos disponibles. Si una clave no existe en el YAML base, el comando falla: así un error de tipeo no pasa en silencio.

## Tests

```powershell
pytest
```

Los tests comprueban lo siguiente:
- Con la misma semilla, `metrics.jsonl` sale idéntico byte a byte.
- Reanudar desde el checkpoint intermedio reproduce exactamente la segunda mitad.
- `clone(checkpoint, 3)` da tres copias idénticas e independientes.
- Dos copias que corren una tras otra en el mismo proceso siguen la misma trayectoria.
- Al reanudar, `meta.json` registra `seed_overridden_by_checkpoint`.
- Cinemática y límites de las articulaciones; el cuerpo nunca atraviesa paredes ni obstáculos.
- Tacto armado a mano, incluido el auto-contacto, que se siente por los dos lados.
- El balance de energía cuadra y la energía nunca baja de cero.
- Mundo reproducible: la misma semilla y la misma historia dan trayectorias idénticas, y las 4 historias son distintas.
- `BodyLink`: `intact`, `cut`, `noise` (escala de la vida encarnada, con estadísticas congeladas) y `fade`. Una extremidad desconectada no se mueve y su acción intentada queda registrada.
- La forma de la observación no cambia en ningún modo.
- El mundo se congela durante la desconexión total y continúa igual al restituir el cuerpo.
- Entornos en paralelo: el entorno `i` de un lote sigue exactamente la misma trayectoria que un entorno individual, en las 4 historias y también con congelamiento.
- Checkpoints con el mundo: reanudar a mitad del entrenamiento reproduce la continuación; los clones siguen trayectorias idénticas; dos copias con distinto modo de `BodyLink` divergen justo en el paso en que se aplica el modo.
- La mente: tamaño, formas, la copia eferente solo cuenta en el modelo que la tiene, el estado oculto se arrastra entre segmentos y la normalización es la misma para los dos modelos.
- La mente que decide: el gradiente de PPO no llega al GRU, al codificador ni a la cabeza de predicción; la log-probabilidad de la gaussiana con tanh es correcta.
- Los impulsos coinciden con casos calculados a mano (hambre, cautela con la acción intentada, curiosidad solo con visión y tacto externo, pesos del temperamento); la escala común de la recompensa es la desviación estándar del retorno.
- La infancia protegida escala el costo de vivir según el calendario.
- Reanudar a mitad de una vida reproduce exactamente la continuación (mente, política, optimizadores, mundos, impulsos y normalizador).
- La regla ética detiene la vida y deja su resumen.
- La sonda de la batería respeta la equivalencia del lote.
- `train_population` salta los agentes terminados y retoma los interrumpidos desde su checkpoint.
- Los intervalos de confianza por bootstrap contienen la estimación y son reproducibles.

## Resultados

Cada corrida crea `runs/<AAAAMMDD-HHMMSS>_<nombre>/`. Git ignora esa carpeta. Contiene:

| Archivo | Contenido |
|---|---|
| `config.yaml` | La configuración completa que se usó |
| `meta.json` | Semilla, commit de git y `git_dirty`, versiones de Python, numpy y torch, sistema, hilos activos, checkpoint de origen y estado final |
| `metrics.jsonl` | Una línea `{"step", "name", "value"}` por evento, sin hora (por eso es comparable entre corridas) |
| `log.txt` | El log legible, con hora |
| `checkpoints/` | Los checkpoints `.pt` del experimento |
| `signals.npz` | (`random_agent`) Todas las señales por paso: observación por canal y plana, acción intentada, aplicada y ejecutada, acciones al vacío, energía, fatiga, contactos, pose del cuerpo y estado de los objetos |
| `run.gif` | (`random_agent`) Animación de los primeros pasos: arena, cuerpo y contactos activos en rojo |
| `learning_curves.png` | (`train_world_model`, `eval_world_model`) Error de predicción por canal, relativo a la persistencia, a lo largo del entrenamiento |
| `eval_results.json` | (`eval_world_model`) Error por canal y por historia del modelo, del modelo sin copia eferente y de la persistencia |

En `random_agent`, `meta.json` agrega `env_steps_per_second` (tiempo dentro de `env.step`) y `loop_steps_per_second` (con el registro incluido).

## Resultados de la Fase 2a

Corrida de referencia: `runs/20261116-152329_fase2a` (commit `332a6fa`, sin cambios locales). Son 98.000 actualizaciones con 32 entornos en paralelo y secuencias de 32 pasos. Tardó 433,5 minutos en la laptop, con un pico de memoria de 351 MB. El 59 % del tiempo se fue en el mundo y el 40 % en los dos modelos.

Rendimiento de `BatchedUmbralEnv` (pasos de entorno por segundo): 791 con N=1, 4.574 con N=8, 7.525 con N=16 y 10.305 con N=32.

Evaluación (`runs/20261117-080216_fase2a_eval_ci`) del checkpoint `update_0098000.pt`, con semillas no vistas en el entrenamiento: 16 entornos por historia (64 semillas) y 2.000 pasos cada uno; tardó 23 s. Los intervalos de confianza del 95 % salen de un bootstrap estratificado por historia sobre las semillas de evaluación (10.000 remuestras). Error cuadrático medio por canal:

| Canal | Con copia eferente | Sin copia eferente | Persistencia | Con copia / persistencia (IC 95 %) | Con copia / sin copia (IC 95 %) |
|---|---|---|---|---|---|
| Propiocepción | 3,11e-3 | 1,55e-2 | 1,93e-2 | 0,16 [0,15; 0,17] | 0,20 [0,19; 0,22] |
| Tacto | 4,48e-3 | 5,07e-3 | 7,14e-3 | 0,63 [0,61; 0,64] | 0,88 [0,87; 0,89] |
| Visión | 1,61e-2 | 1,88e-2 | 2,93e-2 | 0,55 [0,53; 0,57] | 0,86 [0,84; 0,87] |
| Interocepción | 2,25e-5 | 2,33e-5 | 3,16e-5 | 0,71 [0,65; 0,78] | 0,97 [0,95; 0,98] |

Error en los eventos de tacto (cociente con copia / sin copia):

| Evento | Casos | Con copia / sin copia (IC 95 %) |
|---|---|---|
| Inicio de contacto | 16.223 | 0,85 [0,84; 0,86] |
| Inicio de auto-contacto | 5.132 | 0,67 [0,65; 0,69] |
| Inicio de contacto externo | 11.091 | 0,89 [0,89; 0,90] |
| Fin de contacto | 16.163 | 0,80 [0,79; 0,81] |
| Fin de auto-contacto | 5.092 | 0,67 [0,64; 0,69] |
| Fin de contacto externo | 11.071 | 0,83 [0,82; 0,84] |

La primera evaluación de cierre (`runs/20261117-060656_fase2a_eval`, 4 semillas por historia) dio valores compatibles, por ejemplo 0,68 en el inicio del auto-contacto.

Se cumplen los criterios de cierre. El modelo con copia eferente supera a la persistencia en todos los canales y al modelo sin copia en los cuatro; ningún intervalo de confianza toca el 1. La ventaja es mayor en propiocepción y en los eventos de auto-contacto, que es donde saber lo que uno hace sirve para anticipar lo que va a sentir. Al principio del entrenamiento (actualización 5.000), el modelo con copia predecía el inicio del auto-contacto un poco peor que el modelo sin copia (cociente 1,03); esa ventaja se construyó con el entrenamiento. Las figuras están en la carpeta `runs/20261117-060656_fase2a_eval`: `learning_curves.png` y `checkpoint_eval_curves.png`.

Una primera corrida (`runs/20261116-141605_fase2a`) se interrumpió al apagarse la laptop, antes del primer checkpoint. Quedó marcada como `interrupted` y no se cita.
