# Contexto 1er problema
Queremos analizar el rendimiento académico de diferentes escuelas basado en los puntajes promedio obtenidos por los estudiantes en un examen de ingreso. Cada escuela tiene un número diferente de estudiantes que rinden el examen, y la precisión de sus promedios varía en función del tamaño de la muestra.

Nuestro objetivo es modelar la verdadera habilidad académica promedio de cada escuela utilizando un modelo jerárquico bayesiano, que nos permitirá:

- Inferir el rendimiento verdadero de cada escuela teniendo en cuenta la variabilidad en los datos.
- Ajustar por incertidumbre, ya que algunas escuelas tienen más estudiantes que otras.
- Obtener estimaciones más precisas, combinando información a nivel individual y grupal.

# Contexto 2do problema
Nuevo problema: Calidad de focos en una fábrica
Queremos estimar la calidad promedio de los focos producidos en varias líneas de producción. Algunas líneas pueden producir focos de mejor calidad que otras, y la medición de calidad tiene cierto ruido. Modelaremos este problema usando un enfoque bayesiano.

📌 Modelo
Cada línea de producción tiene una calidad real promedio.
La variabilidad entre líneas de producción sigue una distribución normal.
Se hacen mediciones ruidosas de la calidad de los focos en cada línea.
Queremos estimar la calidad real de cada línea usando inferencia bayesiana.

# Instalar y activar el entorno virtual
/virtualenv env     /env/Scripts/activate.bat

# requisitos
pip install -U emcee
pip install matplotlib
pip install scipy
pip install arviz
pip install tqdm

# ejecutar
py .\markov_chain_ejemplo.py
py .\markov_chain_ejemplo2.py

# libreria emcee
https://emcee.readthedocs.io/en/stable/

