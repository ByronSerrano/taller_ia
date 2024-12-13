# Base de conocimientos
reglas = {
    "gripe": ["fiebre", "tos"],
    "meningitis": ["fiebre", "dolor de cabeza"]
}

# Observaciones del paciente
sintomas = ["fiebre", "tos"]

# Generación de hipótesis
hipotesis = []

for enfermedad, sintomas_enfermedad in reglas.items():
    if all(sintoma in sintomas for sintoma in sintomas_enfermedad):
        hipotesis.append(enfermedad)

print("Posibles diagnósticos:", hipotesis)
