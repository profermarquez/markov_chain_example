import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import scipy.stats as stats
import emcee   #Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble sampler for a given number of walkers and dimensions.

# Configuración de datos
np.random.seed(42)
J = 8  # Número de escuelas
n = np.random.randint(50, 500, J)  # Número de estudiantes por escuela
true_ability = np.random.normal(75, 10, J)  # Habilidad real promedio en cada escuela
# Observaciones con ruido si aumenta la variabilidad 
# de los datos sintéticos Si quieres que el modelo haga ajustes más significativos, puedes aumentar la variabilidad en y
y = np.random.normal(true_ability, 5, J) # y = np.random.normal(true_ability, 5, J) genera menos variabilidad en
"""
Reducir sigma_y a 2 / np.sqrt(n) hizo que el modelo considerara que 
las mediciones observadas (y) tienen menos incertidumbre, lo que lleva 
a estimaciones que no están tan fuertemente atadas a los datos observados. Esto 
permite que la calidad estimada fluctúe más libremente y se diferencie mejor de 
la observada.
"""
sigma_y = 2 / np.sqrt(n)  # Desviación estándar basada en cantidad de estudiantes

# Definir el modelo bayesiano
def log_prior(params):
    mu_theta, sigma_theta, *theta = params
    if sigma_theta <= 0:
        return -np.inf  # La desviación estándar debe ser positiva

    lp = stats.norm.logpdf(mu_theta, 75, 10)  # Prior para mu_theta
    lp += stats.halfnorm.logpdf(sigma_theta, scale=10)  # Prior para sigma_theta
    lp += np.sum(stats.norm.logpdf(theta, mu_theta, sigma_theta))  # Priors para theta
    return lp

def log_likelihood(params):
    _, _, *theta = params
    return np.sum(stats.norm.logpdf(y, theta, sigma_y))  # Likelihood

def log_posterior(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params)

# Configuración del MCMC con emcee
ndim = J + 2  # Parámetros: mu_theta, sigma_theta y J valores theta
nwalkers = 20
nsteps = 3000

# Inicialización de los walkers
initial_pos = np.random.normal(75, 10, (nwalkers, ndim))
initial_pos[:, 1] = np.abs(np.random.normal(10, 2, nwalkers))  # sigma_theta positivo
for i in range(J):
    initial_pos[:, 2 + i] = np.random.normal(75, 10, nwalkers)  # theta valores

# Ejecutar el muestreo MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
sampler.run_mcmc(initial_pos, nsteps, progress=True)

# Extraer resultados
samples = sampler.get_chain(discard=500, thin=10, flat=True)  # Quemar primeros 500 pasos
posterior_mu_theta = samples[:, 0].mean()
posterior_sigma_theta = samples[:, 1].mean()
posterior_theta = samples[:, 2:].mean(axis=0)

# Mostrar resultados
print("\n===== RESULTADOS DEL MODELO BAYESIANO =====")
print(f"Número de escuelas analizadas: {J}")
print(f"Promedio global de habilidad estimado: {posterior_mu_theta:.2f}")
print(f"Variabilidad estimada entre escuelas: {posterior_sigma_theta:.2f}\n")

print("Resultados estimados por escuela:")
for i in range(J):
    print(f"  - Escuela {i+1}: {posterior_theta[i]:.2f} (real: {true_ability[i]:.2f}, observada: {y[i]:.2f})")

# Graficar comparaciones entre habilidades reales, observadas y estimadas
plt.figure(figsize=(10, 6))
x_labels = [f"Escuela {i+1}" for i in range(J)]
plt.plot(x_labels, true_ability, 'bo-', label="Habilidad Real")
plt.plot(x_labels, y, 'ro-', label="Habilidad Observada")
plt.plot(x_labels, posterior_theta, 'go-', label="Habilidad Estimada (Bayes)")
plt.xlabel("Escuelas")
plt.ylabel("Habilidad Estimada")
plt.title("Comparación entre Habilidad Real, Observada y Estimada por Escuela")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Mostrar gráfico de la distribución de los parámetros globales
az.plot_trace({"mu_theta": samples[:, 0], "sigma_theta": samples[:, 1]})
plt.show()
