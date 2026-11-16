# CLAUDE.md — reglas del proyecto Umbral

Umbral estudia agentes pequeños entrenados desde cero en un mundo 2D: cómo desarrollan un yo corporal y un personaje, y qué pasa con su identidad cuando pierden el cuerpo. **El diseño se rige por [`docs/protocolo.md`](docs/protocolo.md).** Si algo no está en el protocolo, se pregunta antes de inventarlo.

## Restricciones (no negociables)

- **Hardware:** laptop con Windows 64 bits, Intel i5-10210U (4 núcleos / 8 hilos), 8 GB de RAM y sin CUDA. Todo corre en CPU. Hay que cuidar la memoria y el tiempo de cómputo.
- **Sin costos:** nada de servicios de pago, APIs externas ni cuentas en la nube. **Ningún LLM ni modelo preentrenado forma parte del proyecto.** Todo se entrena desde cero.
- **Dependencias mínimas:** Python 3.11+, `numpy`, `torch` (CPU, desde `https://download.pytorch.org/whl/cpu`), `pyyaml` y `pytest`. Para agregar cualquier otra dependencia hay que preguntar primero.
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

## Fases

- **Ninguna fase avanza sin cumplir su criterio de cierre**, verificado con comandos y tests, no por suposición.
- Fase 0 (base del repositorio): cerrada cuando `umbral run --config configs/base.yaml experiment=noop` deja una carpeta completa en `runs/` y `pytest` pasa.

## Estructura

- `src/umbral/`: `config`, `seeding`, `recorder`, `checkpoint` y `cli`.
- `src/umbral/experiments/`: experimentos registrados con `@register("nombre")`.
- `world/`, `body/`, `mind/` y `metrics/`: se llenan en fases posteriores.
- `configs/base.yaml`: toda clave que se pueda sobrescribir desde la CLI tiene que existir aquí.
- `tests/`: `pytest`.
