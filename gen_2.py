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
    
    disponibility = all(schedule_bed[schedule_bed_genome[1] + i] != 0 for i in range(1, stay_time))
    if disponibility : 
        for i in range(1,stay_time) : 
            schedule_bed[schedule_bed_genome[1] + i] += 1

    # calcul disponibilite chirurchien et bloc 
            

    # écart type par unité de temps : valeur mise 0 et 1
    ecart_type = np.std(schedule_bed[current_date:current_date + schedule_bed_genome[0]])
    #if (ecart_type != 0) :
    #    var_ecart_type = schedule_bed_genome[0]/ecart_type
    #    var_normalized = 1/(1 + np.exp(-var_ecart_type))
    #else : 
    #    var_normalized = 1
    
    
    # evalue si l'écart type dans l'intervalle diminue
    #bin_ecart = 1/(1 + np.exp(-(ecart_type - current_ecart_type)))

    # evalue si l'écart type global diminue
    ecart_type_global = np.std(schedule_bed[current_date:])
    var_normalized = 1/(1 + np.exp(-ecart_type_global))
    print("-------------",best_score,"-------------")

    score = disponibility + var_normalized #+ bin_ecart
    return score


def cross_over():
    return 0

def selection(pop, scores, size_pop, nb_select):
    pop_last = []
    
    scores_sorted_indices = np.argsort(scores)[::-1]  # Tri décroissant, renvoie les indices

    # Sélection des meilleurs individus
    pop_last = [pop[i] for i in scores_sorted_indices[:nb_select]]
    best_score = scores[scores_sorted_indices[0]]
    #print("-------------",best_score,"-------------")
    return pop_last, best_score

def update_schedule(current_schedule, pop_selected, stay_time) :
    for i in range(stay_time) :
        current_schedule[pop_selected[0][1] + i] += 1
    return current_schedule
    

####### test #######

current_schedule_init = [rd.randint(50,100) for i in range(365)]
max_occupancy = 100
current_date = 120
size_pop = 50
current_ecart_type_init = np.std(current_schedule_init[current_date:])
stays_time = [rd.randint(1,10) for i in range(1000)]

current_schedule = current_schedule_init.copy()
for stay_time in stays_time :
    pop = generate_population_schedule_bed(current_date, stay_time)
    ecart_type_global = np.std(current_schedule_init[current_date:])
    score_aim = 1 + 1/(1+np.exp(-ecart_type_global*0.8))
    best_score = 0
    scores = []
    ite = 0
    nb_select = 20

    best_score_evol = []
    # value = 1.99999999998737
    while (best_score < score_aim) :
        ite += 1
        scores = []
        for i in range(size_pop) :
            scores.append(compute_fitness(pop[i], current_date, current_schedule, stay_time))
            #print("predict_time :", pop[i][0], "date d'admission :", pop[i][1], "score :", scores[i])
        pop_selected, best_score = selection(pop, scores, size_pop, nb_select)
        best_score_evol.append(best_score)
        pop = []
        pop = fill_pop(current_date, stay_time, pop_selected, nb_select)
        print("------------------------------------------------------")
    pop_selected, best_score = selection(pop, scores, size_pop, nb_select)

    current_schedule = update_schedule(current_schedule, pop_selected, stay_time)

    #plt.title("Scores de chaque planning de la population")
    #plt.plot([i for i in range(size_pop)], scores)
    #plt.show()

current_ecart_type = np.std(current_schedule[current_date:])

print(current_ecart_type_init, current_ecart_type)
plt.title("Evolution du nombre de lits occupés")
plt.plot([i for i in range(365)],current_schedule_init, label="init")
plt.plot([i for i in range(365)],current_schedule, label="final")
plt.legend()
plt.show()