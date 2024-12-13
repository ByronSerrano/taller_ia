import random
from deap import base, creator, tools, algorithms

# Definir la clase de fitness y el tipo de individuo
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Función de evaluación (fitness)
def evaluate(individual, opponent):
    score = 0
    # Estrategia dinámica del oponente, puede ser aleatoria o provenir de un individuo
    for i, action in enumerate(individual):
        oponent_action = opponent[i]  # Acción del oponente en la ronda i
        if action == 'C' and oponent_action == 'C':
            score += 3  # Ambos cooperan
        elif action == 'C' and oponent_action == 'D':
            score += 0  # El oponente traiciona
        elif action == 'D' and oponent_action == 'C':
            score += 5  # El jugador traiciona
        elif action == 'D' and oponent_action == 'D':
            score += 1  # Ambos traicionan
    return score,

# Crear un individuo (estrategia)
def create_individual():
    return [random.choice(['C', 'D']) for _ in range(5)]  # Estrategias de 5 rondas

# Registrar los operadores genéticos
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Crear la población del jugador
population = toolbox.population(n=50)

# Crear una población para los oponentes (oponentes también son individuos)
opponent_population = toolbox.population(n=50)

# Ejecutar el algoritmo evolutivo
for gen in range(50):  # 50 generaciones
    for ind, opponent in zip(population, opponent_population):
        ind.fitness.values = evaluate(ind, opponent)  # Evaluar el individuo con su oponente

    # Selección, cruce y mutación para ambos, jugadores y oponentes
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.7:  # Probabilidad de cruce
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < 0.2:  # Probabilidad de mutación
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluar los descendientes
    for ind, opponent in zip(offspring, opponent_population):
        ind.fitness.values = evaluate(ind, opponent)

    population[:] = offspring  # Reemplazar la población actual por los descendientes

    # También evolucionamos la población de oponentes
    for gen_opponent in range(50):  # Evolucionar oponentes de manera similar
        for ind, opponent in zip(opponent_population, population):
            opponent.fitness.values = evaluate(opponent, ind)

        offspring_opponent = toolbox.select(opponent_population, len(opponent_population))
        offspring_opponent = list(map(toolbox.clone, offspring_opponent))

        for child1, child2 in zip(offspring_opponent[::2], offspring_opponent[1::2]):
            if random.random() < 0.7:  # Probabilidad de cruce
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring_opponent:
            if random.random() < 0.2:  # Probabilidad de mutación
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluar los descendientes de los oponentes
        for ind, opponent in zip(offspring_opponent, population):
            ind.fitness.values = evaluate(ind, opponent)

        opponent_population[:] = offspring_opponent  # Reemplazar la población de oponentes

# Imprimir las mejores estrategias
best_individual = tools.selBest(population, 1)[0]
best_opponent = tools.selBest(opponent_population, 1)[0]
print(f"La mejor estrategia del jugador es: {best_individual}")
print(f"La mejor estrategia del oponente es: {best_opponent}")