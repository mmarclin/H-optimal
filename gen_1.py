import numpy as np
import random as rd
import matplotlib.pyplot as plt


#### Algorithmes génétiques ####

# ordonanncement de la date d'admission sur le planning de lit
def generate_schedule_bed(predict_time, add_date):
    return [predict_time, add_date]


# création de la population de planning
def generate_population_schedule_bed(current_date, stay_time):
    pop = []
    for i in range(size_pop) :
        predict_time = rd.randint(6, 365-current_date-stay_time-1)
        add_date = rd.randint(current_date, current_date + predict_time)
        pop.append(generate_schedule_bed(predict_time, add_date))
    return pop

def fill_pop(current_date, stay_time, pop_selected, nb_select) :
    pop = pop_selected.copy()
    for i in range(size_pop - nb_select) :
        predict_time = rd.randint(2, 365-current_date-stay_time-1)
        add_date = rd.randint(current_date, current_date + predict_time)
        pop.append(generate_schedule_bed(predict_time, add_date))
    return pop

# Calcul d'un score
def compute_fitness(schedule_bed_genome, current_date, current_schedule, stay_time):
    # score = disponibilité + variation moyenne  
    score = 0
    current_ecart_type = np.std(current_schedule[current_date:current_date + schedule_bed_genome[0]])
    current_moy = np.mean(current_schedule[current_date:current_date + schedule_bed_genome[0]])

    # calcul disponibilite lit : 0 ou 1
    #disponibility = 0
    schedule_bed = current_schedule.copy()
    #for i in range(1,stay_time) :
    #    if schedule_bed[schedule_bed_genome[1] + i] == 0 : 
    #        disponibility == 0
    #        break
    #    else :
    #        disponibility = 1
    
    disponibility = all(schedule_bed[schedule_bed_genome[1] + i] != 0 for i in range(1, stay_time))
    if disponibility : 
        for i in range(1,stay_time) : 
            schedule_bed[schedule_bed_genome[1] + i] += 1

    # calcul disponibilite chirurchien et bloc 
            

    # écart type par unité de temps : valeur mise 0 et 1
    ecart_type = np.std(schedule_bed[current_date:current_date + schedule_bed_genome[0]])
    if (ecart_type != 0) :
        var_ecart_type = schedule_bed_genome[0]/ecart_type
        var_normalized = 1/(1 + np.exp(-var_ecart_type))
    else : 
        var_normalized = 1
    
    # calcul variation moyenne : valeur mise 0 et 1
    # new_moy = np.mean(schedule_bed[current_date:current_date + schedule_bed_genome[0]])
    # var_moy = abs(current_moy - new_moy)/1000

    score = disponibility + var_normalized
    return score


def cross_over():
    return 0

def selection(pop, scores, size_pop, nb_select):
    pop_last = []
    
    scores_sorted_indices = np.argsort(scores)[::-1]  # Tri décroissant, renvoie les indices

    # Sélection des meilleurs individus
    pop_last = [pop[i] for i in scores_sorted_indices[:nb_select]]
    best_score = scores[scores_sorted_indices[0]]
    print("-------------",best_score,"-------------")
    return pop_last, best_score


####### test #######

current_schedule = [rd.randint(220,250) for i in range(365)]

plt.title("Evolution du nombre de lits occupés")
plt.plot([i for i in range(365)],current_schedule)
plt.show()

max_occupancy = 250
current_date = 120
size_pop = 50
stay_time = 20

stays_time = [rd.randint(1,10) for i in range(20)]

pop = generate_population_schedule_bed(current_date, stay_time)
current_ecart_type = np.std(current_schedule[current_date:])
best_score = 0
scores = []
ite = 0
nb_select = 20

best_score_evol = []
while (best_score < 1.99999999998737) :
    ite += 1
    scores = []
    for i in range(size_pop) :
        scores.append(compute_fitness(pop[i], current_date, current_schedule, stay_time))
        print("predict_time :", pop[i][0], "date d'admission :", pop[i][1], "score :", scores[i])
    pop_selected, best_score = selection(pop, scores, size_pop, nb_select)
    best_score_evol.append(best_score)
    pop = []
    pop = fill_pop(current_date, stay_time, pop_selected, nb_select)
    print("------------------------------------------------------")

plt.title("Evolution du score maximal")
plt.plot([i for i in range(ite)], best_score_evol)
plt.show()

plt.title("Scores de chaque planning de la population")
plt.plot([i for i in range(size_pop)], scores)
plt.show()

