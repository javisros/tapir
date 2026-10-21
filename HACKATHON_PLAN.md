# TAPIR · Plan de trabajo para el OSI Hackathon (21–22 oct 2026)

> **Para Claude Code (modo plan):** lee este documento entero y la estructura del repositorio.
> **No escribas código todavía.** Tu tarea es devolver un plan de implementación ordenado
> (ver la sección 9, "Qué tienes que devolver"). Si algo de este documento choca con lo que
> ves en el código, dilo antes de planificar.

---

## 1. Qué es TAPIR

**TAPIR — Tracking Absent Pathways in Interstellar Reactions.**

Busca qué reacción falta en una red astroquímica (KIDA) para explicar una molécula observada en
la nube oscura TMC-1 que el modelo no reproduce. Funciona como un detective:

1. **Genera** reacciones candidatas (hoy por analogía con plantillas; en el hackathon, también con un LLM).
2. **Filtra** las que violan la física (conservación de átomos y carga).
3. **Estima** una velocidad para cada candidata.
4. **Simula** cada candidata en un modelo 0D de nube oscura (fase gas).
5. **Ordena** las candidatas según cuánto acercan el modelo a la observación.

El resultado es una **lista corta de hipótesis comprobables**, no reacciones confirmadas.

## 2. Qué ya existe (no se reimplementa)

Estructura actual del repo:

```
tapir/            librería: modelo 0D, química, generador por analogía, pre-ranking por flujo,
                  configuración de knock-out, observaciones de TMC-1, rutas
scripts/          ejecutables (python -m scripts.<nombre>), incluido scripts.generate
tests/            30 tests que pasan
benchmark/        cases.json (los 6 casos de knock-out)
results/          resultados de benchmarks y validaciones
networks/         redes KIDA 2014 y 2024 (en local, fuera de git)
```

Resultados ya validados:

- Modelo 0D validado contra Wakelam et al. 2024 (2,5 s por simulación).
- **Benchmark B (knock-out):** se esconde una reacción real y TAPIR la recupera. 5/6 en rango 1,
  6/6 en el top-5, MRR 0,875 (frente a ~0,39 al azar), 74 s con 4 procesos.
- **Benchmark A (cobertura del generador):** recall 0,45, MRR 0,119 sobre 200 reacciones ocultas.
- Único fallo de B: HNCCC (el generador propone el isómero equivocado, CN + C₂H₂ → HNCCC + H).
- Problema conocido: el estimador da 1e-10 a reacciones neutro + H₂, que a 10 K tienen barrera;
  el pre-ranking por flujo las sobrevalora (p. ej. C₃N + H₂ → HNCCC + H).

Existe una etiqueta git `pre-hackathon`: todo lo posterior se hace durante el evento.

## 3. El resultado que queremos

Al final del jueves:

1. **Descubrimiento de CH₂CCH:** top de hipótesis contra la observación de TMC-1, cada una con la
   **velocidad necesaria** para explicarla, si esa velocidad es **físicamente alcanzable**, si
   **sobrevive** en una segunda configuración del modelo y **qué dice la literatura** (Valency MCP Bond).
2. **Generador LLM** integrado y **comparado** con el de analogía en el mismo benchmark, con una
   medida honesta de memorización (canarios).
3. **App en Marimo** con dos pestañas (Validación y Descubrimiento), publicada en molab.
4. (Opcional) **C₅O** en modo destrucción: el modelo lo sobreproduce, así que se buscan
   reacciones que lo destruyan.
5. Entrega: figura, resumen de una página, README nuevo, vídeo de 4 min, repo público.

**Mínimo imprescindible:** puntos 1, 2 y 3.

## 4. Equipo y reparto

Dos personas, cada una en su portátil, su rama y su sesión de Claude Code.
**Regla de oro: nunca dos personas en el mismo fichero.**

| Persona | Rol | Se encarga de |
|---|---|---|
| Químico | Ciencia | Velocidades, descubrimiento, Valency/Bond, robustez, velocidades publicadas, C₅O, interpretación, resumen |
| Ignacio | Plataforma | Generador LLM y su comparación, app Marimo, molab, README |

**Cómo se pasan el trabajo:** no se envía nada a mano. El descubrimiento guarda sus resultados en
`results/discovery/<especie>.json`; al fusionar en GitHub, la app de Ignacio los lee de ahí.
El generador LLM devuelve candidatas **en el mismo formato que ya usa el generador por analogía**
(no hace falta definir un formato nuevo).

Ficheros nuevos previstos (orientativo; ajusta los nombres a las convenciones del repo):

| Fichero | Responsable |
|---|---|
| `tapir/discovery.py`, `scripts/discover.py` | Químico |
| `tapir/literature.py`, `scripts/check_literature.py` | Químico |
| `tapir/generator_llm.py`, `scripts/bench_llm.py` | Ignacio |
| `app/tapir_app.py`, `scripts/export_app_data.py`, `app/data/` | Ignacio |
| `results/discovery/*.json` | Escribe el Químico, lee Ignacio |

## 5. Calendario (hora de Madrid)

Entrega: jueves 22 a las 19:00 CT = **viernes 23 a las 02:00 en Madrid**. Objetivo interno: enviar a las 00:00.

**Miércoles 21**

| Hora | Químico | Ignacio |
|---|---|---|
| 10:00 | Llamada de arranque (15 min) | |
| 10:15–14:00 | T1 velocidades H₂ → T2 descubrimiento CH₂CCH | T5 generador LLM |
| 14:00 | **Fusionar** (Ignacio recibe los resultados de CH₂CCH) | |
| 14:00–19:00 | T3 Valency/Bond → T4 robustez | T6 app: pestañas Validación y Descubrimiento |
| 19:00 | **Fusionar + T7 integrar el LLM en el descubrimiento (juntos)** | |
| 19:30–23:00 | T8 ejecución final CH₂CCH + velocidades publicadas | T6 pulir app + comparación LLM vs analogía |
| 23:00 | **Revisar resultados juntos y decidir el jueves** | |

**Jueves 22**

| Hora | Químico | Ignacio |
|---|---|---|
| 10:00–13:00 | T9 C₅O (opcional) | T10 publicar en molab |
| 13:00 | **Resultados congelados** (commit "Final discovery results") | |
| 13:00–16:00 | T11 figura + resumen de una página | T12 README |
| 16:00–19:00 | Vídeo (juntos) | |
| 20:00–00:00 | Repo público, etiqueta `v1.0`, envío | |

Fusiones a `main` solo en los puntos de control: 14:00, 19:00, 23:00 (miércoles) y 13:00 (jueves).

## 6. Tareas

Cada tarea en su rama, fusionada por Pull Request con los tests pasando.

### T1 · Velocidades neutro + H₂ (Químico) — rama `fix/h2-rates`
- Nueva clase "neutral + H₂" en el estimador: k = mediana a 10 K de las reacciones neutro-neutro con H₂
  en KIDA 2024, **excluyendo la reacción oculta y sus equivalentes** (anti-fuga). `k_source = "kida_class_median"`.
- **Aceptación:** la mediana es claramente menor que 1e-10; C₃N + H₂ sale del top-10 por flujo en el caso HNCCC;
  benchmark B mantiene los mismos rangos; test anti-fuga nuevo.

### T2 · Descubrimiento de CH₂CCH (Químico) — rama `feat/discovery`
- Modo "observation": el objetivo es la abundancia observada en TMC-1 (no el modelo base).
- Configuración de referencia: KIDA 2024, C/O = 1,2, S = 8e-8, t_ref = 6e6 años, condiciones W24_GAS.
- Candidatas por analogía dirigidas a formar la especie → pre-ranking por flujo → simular top-K = 40.
- Puntuación: D(t_ref) sobre las especies de Gratier+2016 más la especie objetivo, frente a la red base
  (`delta_D`), y `x_ratio = X_cand / X_obs`.
- **k necesaria:** bisección en log10(k) (1e-14 a 1e-7 bimoleculares; hasta 1e-5 recombinación), máximo
  10 simulaciones por candidata, hasta que X_target = X_obs. `capture_feasible = k_needed <= k de captura de su clase`.
- **Unidades:** modelo y observación comparados en la misma unidad (abundancia relativa a H₂; factor 2 respecto a n_H).
- Parámetro `--generators` (de momento solo `analogy`).
- Salida: `results/discovery/CH2CCH.json` (top-20) + CSV resumen. Multiprocessing, 4 procesos.
- **Aceptación:** el fichero existe; informe con tiempo total y número de candidatas alcanzables.

### T3 · Literatura con Valency MCP Bond (Químico) — rama `feat/valency-literature`
- Valency MCP Bond: gratuito, conecta un LLM/agente con ~44,5 M artículos. Conexión como agente con
  **bearer token** (token en `.env`; guía de configuración en `docs/valency.md`).
- Para el top-10 de un fichero de descubrimiento: `literature = {status: measured|proposed|computed|none, refs (≤3, DOI/URL), k_lit}`.
- **No inventar referencias:** si no hay, `status = "none"`. Caché por reacción canónica.
- Backend de reserva con la misma interfaz (LLM con búsqueda web), seleccionable por variable de entorno.
- **Solo en modo descubrimiento:** debe fallar si se llama desde un benchmark (conocería las reacciones ocultas).
- **Aceptación:** 3 ejemplos con referencias para revisión manual; las referencias existen y hablan de la reacción.

### T4 · Robustez (Químico) — rama `feat/robustness`
- Repetir el top-20 en una segunda configuración: KIDA 2024, C/O = 1,2, S = 1,5e-5, t_ref = 4,5e5 años.
- `robust = true` si mejora D y acerca X_target a X_obs en ambas configuraciones.
- **Aceptación:** campo `robust` en el JSON; informe con cuántas sobreviven.

### T5 · Generador LLM (Ignacio) — rama `feat/llm-generator`
- Segundo generador con **la misma salida que el generador por analogía**, más `source = "llm"` y `rationale`.
- Proveedor configurable por variable de entorno (por defecto la API de Anthropic; modelo y clave en `.env`).
  Salida estructurada en JSON. Caché en disco por hash del prompt.
- Prompt: contexto físico (nube oscura, 10 K, fase gas; solo reacciones sin barrera: ion-neutro,
  recombinación disociativa, neutro-neutro entre radicales, asociación radiativa), especie objetivo,
  sus 5 principales reacciones de formación y destrucción en t_ref, y la lista de especies permitidas
  (mundo cerrado). Pide 30 candidatas con reactivos, productos, clase y justificación de una línea.
  *El químico revisa la parte química del prompt con Ignacio a las 12:00.*
- Postproceso: descartar especies fuera de la lista, comprobar conservación, deduplicar por clave canónica,
  contar descartes por motivo.
- Evaluación (`scripts/bench_llm.py`): benchmark B (analogía vs LLM vs unión: Hits@1/5, MRR, rango/n);
  50 reacciones del benchmark A con semilla fija; **contaminación**: 5 canarios (reacciones inventadas
  plausibles) y recall en reacciones antiguas frente a añadidas en KIDA 2024. Tabla + 3 líneas honestas.
- Probar primero con 1 caso y estimar el coste antes de lanzar todo.

### T6 · App en Marimo (Ignacio) — rama `feat/marimo-app`
- `app/tapir_app.py` para `marimo run` y molab. **No simula:** lee resultados precalculados
  (las redes KIDA no van en el repo). `scripts/export_app_data.py` prepara `app/data/` (< 10 MB).
- Pestaña **Validation:** selector de caso del benchmark B; curvas base / knock-out / reinyectada;
  tabla con rango, n, rango/n y azar (H_n/n); MRR frente al azar.
- Pestaña **Discovery:** selector de especie (`results/discovery/*.json`); tabla de hipótesis
  (reacción, source, rationale, delta_D, k_needed, capture_feasible, robust, literatura con enlaces, k_lit);
  al seleccionar una fila, curvas antes/después frente a la observación.
  Aviso fijo: *"Ranked hypotheses, not confirmed reactions."*
- Pestaña **Generator:** comparación analogía vs LLM y nota de contaminación (resultados de T5).

### T7 · Integrar el LLM en el descubrimiento (juntos, 19:00) — rama `feat/integrate-llm`
- `--generators analogy,llm` en `scripts/discover.py`: unir candidatas (deduplicar por clave canónica,
  conservar ambos `source` si coinciden) y pasarlas por el mismo pre-ranking, simulación y k necesaria.
- No cambiar la lógica de ninguno de los dos generadores.

### T8 · Ejecución final de CH₂CCH + velocidades publicadas (Químico) — rama `feat/literature-rates`
- Orden: descubrimiento con analogía + LLM → T3 (literatura) → T4 (robustez) → velocidades publicadas.
- Para candidatas con `k_lit`: volver a simular con esa k (`k_source = "literature"`), guardar ambos
  resultados (captura y literatura). No modificar la función de simulación.

### T9 · C₅O en modo destrucción (Químico, opcional) — rama `feat/destruction-mode`
- Modo "destruction" en el generador: candidatas con la especie entre los reactivos, mismo anti-fuga.
- Si X_mod > X_obs, usar destrucción automáticamente; k necesaria para bajar X_target hasta X_obs.
- Salida `results/discovery/C5O.json`; test: todas las candidatas contienen la especie como reactivo.

### T10 · Publicar en molab (Ignacio) — rama `deploy/molab`
- La app funciona solo con `app/data/`; dependencias declaradas; botón "Open in molab" en el README.
- **Aceptación:** se abre desde una ventana privada sin iniciar sesión.

### T11 · Figura y resumen (Químico) — rama `release/figure`
- `scripts/make_figure.py` → `figure.png`: (A) esquema del pipeline; (B) curvas de CH₂CCH: base,
  mejor hipótesis y observación; (C) analogía vs LLM vs unión (Hits@1, MRR) con línea del azar.
- Resumen de una página: problema → TAPIR → validación → descubrimiento → limitaciones → qué estaba hecho antes del evento.

### T12 · README definitivo (Ignacio) — rama `release/readme`
- Arriba: nombre, una frase, figura, resultados (benchmark B frente al azar, LLM vs analogía, descubrimiento).
- Después: botón de molab, quickstart, cómo funciona, dónde entran Valency y Marimo, limitaciones,
  trabajo previo (etiqueta `pre-hackathon`), cita y licencia. Lo técnico antiguo pasa a `docs/technical.md`.

## 7. Reglas que no se rompen

- **No modificar el modelo de simulación** ni romper los tests existentes.
- **Anti-fuga:** en benchmarks, la reacción oculta y sus equivalentes nunca entran en plantillas,
  medianas, prompts ni búsquedas de literatura. Bond y la literatura **solo en modo descubrimiento**.
- **No inventar datos, resultados ni referencias.** Si algo no cuadra, parar y avisar.
- **Claves de API solo en `.env`** (en `.gitignore`). Revisar el historial antes de hacer el repo público.
- Las redes KIDA no se suben al repo.
- Código, nombres y salidas en inglés.
- Commits pequeños; ramas cortas; fusión solo en los puntos de control.
- Se presentan **hipótesis ordenadas**, no reacciones confirmadas; la memorización del LLM se declara.

## 8. Planes B

| Si pasa | Entonces |
|---|---|
| Bond no funciona a tiempo | T3 usa el backend de reserva (LLM con búsqueda web) |
| El LLM da basura o cuesta demasiado | Se queda como prototipo con su tabla honesta; el peso lo lleva analogía + Bond |
| Ninguna hipótesis explica CH₂CCH | Se presenta como resultado ("ninguna reacción en fase gas lo cierra con k alcanzable") y C₅O pasa a principal |
| molab pide login o falla | Vídeo de la app en local + capturas en el README |

## 9. Qué tienes que devolver (modo plan)

1. **Comprobación del repo:** qué módulos y funciones existentes vas a reutilizar para cada tarea
   (generador por analogía, estimador de velocidades, pre-ranking por flujo, simulación, observaciones),
   con sus rutas reales. Señala cualquier diferencia con lo que dice este documento.
2. **Plan por tareas (T1–T12):** para cada una, ficheros a crear o modificar, funciones principales,
   dependencias con otras tareas, tests, criterio de aceptación y estimación de tiempo realista.
3. **Riesgos técnicos** que veas (rendimiento de la bisección, unidades, coste del LLM, tamaño de `app/data/`, etc.).
4. **Preguntas** que necesites resolver antes de empezar (máximo 5, concretas).
5. Si ves que el calendario no cabe, **propón qué recortar** respetando el mínimo imprescindible (sección 3).

No escribas código hasta que el plan esté aprobado.
