# Guía de entrevista — RadIA (para Gustavo)

Lee esto antes de cualquier entrevista donde aparezca el proyecto. Si te preguntan, di la verdad: **lo desarrollaste con Claude Code**, que es justamente lo que pones en el CV como "usuario avanzado de IA". Lo valioso es que entiendes qué hace, cómo se mide y cuáles son sus límites.

## El pitch en 30 segundos
"Hice RadIA, un proyecto de IA aplicada a salud. Tiene dos partes: un modelo que mira radiografías de tórax y detecta neumonía, con un mapa de calor que muestra en qué zona del pulmón se fijó; y un clasificador que, a partir de síntomas descritos en texto, sugiere los diagnósticos más probables. En datos de prueba, el de radiografías detectó el 98,5 % de las neumonías con 92 % de especificidad. Lo construí con Claude Code, y el código y las métricas están en GitHub."

## Cómo funciona (con tus palabras)
- **Datos:** 5.856 radiografías públicas de niños (normal / neumonía). Se separan en entrenamiento, validación y prueba. El modelo **nunca ve** las de prueba mientras aprende.
- **Modelo:** ResNet-18, una red neuronal que ya sabía reconocer formas en fotos comunes (ImageNet). La "reentrenamos" (*fine-tuning*) con las radiografías. Se entrenó en la tarjeta gráfica del computador en unos 10 minutos.
- **Grad-CAM:** pinta en rojo las zonas de la imagen que más influyeron en la decisión. Sirve para ver si el modelo mira el pulmón o algo irrelevante.
- **Síntomas:** convierte el texto en números según las palabras que aparecen (TF-IDF) y un modelo estadístico (regresión logística) elige entre 22 diagnósticos.

## Las métricas (lo que más preguntan)
- **Sensibilidad 98,5 %:** de 390 radiografías con neumonía, detectó 384. Se le escaparon 6 (falsos negativos).
- **Especificidad 92,3 %:** de 234 radiografías normales, dijo "normal" en 216. En 18 dio una falsa alarma (falsos positivos).
- **AUC 0,993:** qué tan bien separa ambos grupos en todos los umbrales posibles (1 = perfecto, 0,5 = azar).
- **¿Por qué priorizar sensibilidad?** En tamizaje es peor perder una neumonía que pedir una segunda revisión de un sano.
- **Umbral:** el corte de decisión (0,92) se eligió con los datos de validación, no con los de prueba, para no "hacer trampa".

## Preguntas difíciles y cómo responder
- **"¿Lo usarías en un hospital?"** No tal cual. Es educativo: radiografías pediátricas de un solo centro y solo normal vs neumonía. Para uso clínico se necesitaría validación con datos locales, aprobación regulatoria (ISP) y siempre supervisión médica.
- **"¿Lo programaste tú?"** Lo desarrollé con Claude Code: definí el objetivo, elegí los datos, revisé el código, entrené el modelo y validé las métricas. Es el mismo modo de trabajo que uso en soporte TI y documentación.
- **"¿Qué mejorarías?"** Probar con radiografías de adultos, agregar más patologías (por ejemplo con ChestX-ray14), y conectarlo a un visor PACS en un piloto controlado.
- **"¿Qué es un falso positivo?"** Cuando el modelo dice "neumonía" y el paciente está sano. En el README hay un ejemplo: el mapa muestra que el modelo miró el borde del corazón.

## Cómo mostrarlo en vivo
En el computador: `cd radia`, `.venv\Scripts\activate`, `python app.py` y abrir http://127.0.0.1:7860. Sube `results/ejemplo_neumonia.png` y `results/ejemplo_normal.png`, y en la pestaña Síntomas escribe "fiebre alta, tos con flema, falta de aire y dolor de pecho".
