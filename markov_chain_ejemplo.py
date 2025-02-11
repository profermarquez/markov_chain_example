import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import scipy.stats as stats
import emcee

# 🔧 Configuración de datos
np.random.seed(42)
L = 12  # Número de líneas de producción
n = np.random.randint(50, 500, L)  # Número de focos evaluados por línea
true_quality = np.random.normal(80, 10, L)  # Calidad real promedio por línea
y = np.random.normal(true_quality, 5, L)  # Mediciones observadas con ruido
"""
Reducir sigma_y a 2 / np.sqrt(n) hizo que el modelo considerara que 
las mediciones observadas (y) tienen menos incertidumbre, lo que lleva 
a estimaciones que no están tan fuertemente atadas a los datos observados. Esto 
permite que la calidad estimada fluctúe más libremente y se diferencie mejor de 
la observada.
"""
sigma_y = 2 / np.sqrt(n)  # Desviación estándar basada en la cantidad de muestras

# 🎯 Definir el modelo bayesiano
def log_prior(params):
    mu_quality, sigma_quality, *quality = params
    if sigma_quality <= 0:
        return -np.inf  # La desviación estándar debe ser positiva

    lp = stats.norm.logpdf(mu_quality, 80, 15)  # Prior para la calidad global
    lp += stats.halfnorm.logpdf(sigma_quality, scale=15)  # Prior para la variabilidad entre líneas
    lp += np.sum(stats.norm.logpdf(quality, mu_quality, sigma_quality))  # Priors para cada línea
    return lp

def log_likelihood(params):
    _, _, *quality = params
    return np.sum(stats.norm.logpdf(y, quality, sigma_y))  # Likelihood

def log_posterior(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params)

# 🔥 Configuración del MCMC con emcee
ndim = L + 2  # Parámetros: mu_quality, sigma_quality y L valores de calidad
nwalkers = 2*ndim
nsteps = 5000

# Inicialización de los walkers
initial_pos = np.random.normal(80, 15, (nwalkers, ndim))
initial_pos[:, 1] = np.abs(np.random.normal(15, 5, nwalkers))  # sigma_quality positivo
for i in range(L):
    initial_pos[:, 2 + i] = np.random.normal(80, 10, nwalkers)  # Calidad de cada línea

# 🚀 Ejecutar el muestreo MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
sampler.run_mcmc(initial_pos, nsteps, progress=True)

# 📊 Extraer resultados
samples = sampler.get_chain(discard=1000, thin=10, flat=True)  # Quemar primeros 1000 pasos
posterior_mu_quality = np.median(samples[:, 0])
posterior_sigma_quality = np.median(samples[:, 1])
posterior_quality = np.median(samples[:, 2:], axis=0)

# 📋 Mostrar resultados
print("\n===== RESULTADOS DEL MODELO BAYESIANO =====")
print(f"Número de líneas de producción analizadas: {L}")
print(f"Calidad global promedio estimada: {posterior_mu_quality:.2f}")
print(f"Variabilidad estimada entre líneas: {posterior_sigma_quality:.2f}\n")

print("Resultados estimados por línea de producción:")
for i in range(L):
    print(f"  - Línea {i+1}: {posterior_quality[i]:.2f} (real: {true_quality[i]:.2f}, observada: {y[i]:.2f})")

# 📈 Graficar comparaciones entre calidad real, observada y estimada
plt.figure(figsize=(12, 6))
x_labels = [f"Línea {i+1}" for i in range(L)]
plt.plot(x_labels, true_quality, 'bo-', label="Calidad Real")
plt.plot(x_labels, y, 'ro-', label="Calidad Observada")
plt.plot(x_labels, posterior_quality, 'go-', label="Calidad Estimada (Bayes)")
plt.xlabel("Líneas de Producción")
plt.ylabel("Calidad de los Focos")
plt.title("Comparación entre Calidad Real, Observada y Estimada")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# 📊 Mostrar gráfico de la distribución de los parámetros globales
az.plot_trace({"mu_quality": samples[:, 0], "sigma_quality": samples[:, 1]})
plt.show()
