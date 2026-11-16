# Umbral

Umbral es un proyecto de investigación. Agentes pequeños, entrenados desde cero en un mundo 2D, desarrollan un "yo corporal" y un personaje propio. Después se les quita el cuerpo y se observa, paso a paso, qué pasa con su identidad. Todo corre en CPU, en una laptop, sin servicios externos ni modelos preentrenados. El diseño del estudio está en [`docs/protocolo.md`](docs/protocolo.md).

**Estado: Fase 2a.** Ya existen la base (configuración, semillas, registro, checkpoints y CLI), el mundo 2D con sus 4 historias, el cuerpo con sus sensores y su energía, el vínculo cuerpo–mente (`BodyLink`), un visor y los entornos en paralelo (`BatchedUmbralEnv`). Los checkpoints incluyen el mundo. También existe el núcleo de la mente: una red recurrente que aprende a predecir lo que va a sentir según lo que hace. El cuerpo todavía se mueve con una política aleatoria; los impulsos y los temperamentos llegan en la Fase 2b.

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
