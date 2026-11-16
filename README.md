# Umbral

Umbral es un proyecto de investigación. Agentes pequeños, entrenados desde cero en un mundo 2D, desarrollan un "yo corporal" y un personaje propio. Después se les quita el cuerpo y se observa, paso a paso, qué pasa con su identidad. Todo corre en CPU, en una laptop, sin servicios externos ni modelos preentrenados. El diseño del estudio está en [`docs/protocolo.md`](docs/protocolo.md).

**Estado: Fase 0.** Solo existe la base: configuración, semillas, registro de experimentos, checkpoints con estado completo y una CLI. El mundo, el cuerpo y la mente todavía no existen.

## Instalación (Windows, PowerShell)

Requiere Python 3.11 o superior.

```powershell
cd umbral
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # torch solo CPU, primero
pip install -e ".[dev]"                                              # umbral + numpy, pyyaml, pytest
```

Si PowerShell bloquea `Activate.ps1`, ejecuta una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Uso

Para correr el experimento de prueba:

```powershell
umbral run --config configs/base.yaml experiment=noop
```

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

## Resultados

Cada corrida crea `runs/<AAAAMMDD-HHMMSS>_<nombre>/`. Git ignora esa carpeta. Contiene:

| Archivo | Contenido |
|---|---|
| `config.yaml` | La configuración completa que se usó |
| `meta.json` | Semilla, commit de git y `git_dirty`, versiones de Python, numpy y torch, sistema, hilos activos, checkpoint de origen y estado final |
| `metrics.jsonl` | Una línea `{"step", "name", "value"}` por evento, sin hora (por eso es comparable entre corridas) |
| `log.txt` | El log legible, con hora |
| `checkpoints/` | Los checkpoints `.pt` del experimento |
