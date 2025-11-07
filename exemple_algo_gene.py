import numpy as np
import random
import matplotlib.pyplot as plt

def random_matrix(nrows, ncols):
    matrix = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            matrix[i,j] = random.uniform(2, 20)
    return matrix

# matrice des distances (Exemple de 20 clients)
distances = random_matrix(20, 20)




# Définition de la fonction objectif que l'on cherche à minimiser
def objectif(solution):
    # Exemple : la distance parcouru
    cout = 0
    for i in range(len(solution)-1):
        cout+=distances[solution[i],solution[i+1]]
        
    return cout   


# Définition de la fonction de sélection par tournoi
def selection(population, taille_tournoi):
    participants = random.sample(population, taille_tournoi)
    return min(participants, key=lambda x: objectif(x))

# Définition de la fonction de croisement à un point
def croisement(parent1, parent2):
    point_croisement = random.randint(1, len(parent1) - 1)
    enfant1 = parent1[:point_croisement] + parent2[point_croisement:]
    enfant2 = parent2[:point_croisement] + parent1[point_croisement:]
    return enfant1, enfant2

# Définition de la fonction de mutation
def mutation(solution, taux_mutation):
    for i in range(len(solution)):
        if random.random() < taux_mutation:
            solution[i] = max(solution) - solution[i]  # Inversion de la valeur de l'élément
    return solution

# Définition de l'algorithme génétique
def genetique(taille_population, taille_solution, taux_mutation, max_iterations):
    population = [[random.randint(0, 9) for j in range(taille_solution)] for i in range(taille_population)]
    cout_history = []

    
    for i in range(max_iterations):
        meilleure_solution = max(population, key=lambda x: objectif(x))
        meilleure_valeur = objectif(meilleure_solution)
        #print(meilleure_solution)
        if len(cout_history)!=0:
            cout_history.append(min(meilleure_valeur ,cout_history[-1]))
        else:
            cout_history.append(meilleure_valeur)
                                
        parents = [selection(population, 2) for j in range(taille_population)]
        aux = [croisement(parents[j], parents[j+1]) for j in range(0, taille_population, 2)]
        enfants1 = [x[0] for x in aux]
        enfants2 = [x[1] for x in aux]
        enfants = enfants1+enfants2
        population = [mutation(enfants[j], taux_mutation) for j in range(taille_population)]
         
        
        
    return  meilleure_solution, cout_history[-1], cout_history

# Exemple d'utilisation
taille_population = 200
taille_solution = 20
taux_mutation = 0.01
max_iterations = 200
meilleure_solution, meilleure_valeur, cout_history = genetique(taille_population, taille_solution, taux_mutation, max_iterations)
plt.plot(cout_history)
print("Meilleure solution :", meilleure_solution)
print("Meilleure valeur :", meilleure_valeur)
plt.show()