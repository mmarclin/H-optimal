############################################################

import numpy as np
import random as rd
import matplotlib.pyplot as plt

############################################################
################### Algorithmes génétiques #################

# création de la population de planning
def generate_population_schedule_bed(current_date, stay_time, size_pop, size_study):
    pop = []
    for i in range(size_pop) :
        predict_time = rd.randint(6, size_study-current_date-stay_time-50)
        add_date = rd.randint(current_date, current_date + predict_time)
        pop.append(generate_schedule_bed(predict_time, add_date))
    return pop

# ordonanncement de la date d'admission sur le planning de lit
def generate_schedule_bed(predict_time, add_date):
    return [predict_time, add_date]

# Calcul de score
def compute_fitness(schedule_bed_genome, current_date, current_schedule, stay_time, max_occupancy, f_dispo = 1, f_ecart = 1, f_malus = 1, f_bonus = 1):
    # score = disponibilité + variation ecart type + sous la courbe + malus
    score = 0

    # calcul disponibilite lit : 0 ou 1
    schedule_bed = current_schedule.copy()
    current_moy = np.mean(schedule_bed[current_date:])
    
    disponibility = all(schedule_bed[schedule_bed_genome[1] + i] <= max_occupancy for i in range(1, stay_time))
    for i in range(1,stay_time) : 
        schedule_bed[schedule_bed_genome[1] + i] += 1

    # calcul disponibilite chirurchien et bloc 
    
    # malus : jour où les lits sont déjà bien remplis
    # malus : week end
    #malus_bed_point = 0
    malus_week_end = 0
    moy_sejour = np.mean(schedule_bed[schedule_bed_genome[1]:schedule_bed_genome[1]+stay_time])
    for i in range(stay_time) :
        #if moy_sejour < schedule_bed[schedule_bed_genome[1] + i] :
        #    malus_bed_point += 1/stay_time
        if (schedule_bed_genome[1] + i)%7 == 0 or (schedule_bed_genome[1] + i)%7 == 6 :
            malus_week_end += 1/stay_time

    bonus_min = 1/( 1+np.exp(moy_sejour - min(schedule_bed[current_date:])) )

    # evalue si l'écart type global diminue
    ecart_type_global = np.std(schedule_bed[current_date:])
    if (ecart_type_global != 0):
        var_normalized = 1/(1 + np.exp(-(1/ecart_type_global)))
    else : 
        var_normalized = 1

    score = f_dispo*disponibility + f_ecart*var_normalized - f_bonus*bonus_min
    return score

# Sélection des élements ayant les meilleurs scores
def selection(pop, scores, nb_select):
    pop_last = []
    scores_sorted_indices = np.argsort(scores)[::-1]  # Tri décroissant, renvoie les indices
    # Sélection des meilleurs individus
    pop_last = [pop[i] for i in scores_sorted_indices[:nb_select]]
    best_score = scores[scores_sorted_indices[0]]
    return pop_last, best_score

# Croisement entre des élements de la population sélectionné
def cross_over(parent1, parent2):
    num_gen_crossed = rd.randint(0,1)
    if verify_cross_gene(parent1, parent2):
        enfant1 = [parent1[0], parent2[1]]
        enfant2 = [parent1[0], parent2[1]]
        return enfant1, enfant2
    else : 
        return parent1, parent2    
def verify_cross_gene(parent1, parent2):
    return ( (parent1[1]<parent2[0]) and (parent2[1]<parent1[0]) )

# Mutation de la population sélectionné
def mutation(pop_selected, taux_mutation, current_date, stay_time):
    for i in range(len(pop_selected)):
        if rd.random() < taux_mutation:
            num_gen_mutated = rd.randint(0,1)
            if num_gen_mutated == 0 :
                predict_time = rd.randint(6, size_study-current_date-stay_time-50) 
                if (pop_selected[i][1] <= predict_time - stay_time) :
                    pop_selected[i][0] = predict_time
            else :
                add_date = rd.randint(current_date, current_date + pop_selected[i][0])
                pop_selected[i][1] = add_date
    return pop_selected

# remplir la population restant
def fill_pop(current_date, stay_time, pop_selected, nb_select, size_pop) :
    pop = pop_selected.copy()
    for i in range(size_pop - nb_select) :
        predict_time = rd.randint(2, size_study-current_date-stay_time)
        add_date = rd.randint(current_date, current_date + predict_time)
        pop.append(generate_schedule_bed(predict_time, add_date))
    return pop

# Mise à jour du planning après ajout des patients
def update_schedule(current_schedule, pop_selected, stay_time) :
    for i in range(stay_time) :
        current_schedule[pop_selected[0][1] + i] += 1
    return current_schedule

# lancement algorithme génétique
def genetique(size_pop, max_iteration, taux_mutation, max_occupancy, current_date, current_schedule, stay_time, date_rdv, f_dispo = 1, f_ecart = 1, f_malus = 1, f_bonus = 1):
    # sélection > croisement > mutation > selection
    pop = generate_population_schedule_bed(date_rdv, stay_time, size_pop, size_study)
    # nombre de plannings gardés dans la sélection
    nb_select = 20
    current_ecart_type = np.std(current_schedule[date_rdv:])
    for k in range(max_iteration) :
        # A partir de la 2ieme génération
        if k != 0 :
            # croisement
            best_parent = [pop_selected[0]]
            aux = []
            if (nb_select-1)%2 == 1 :
                for j in range(1, nb_select-1, 2) :
                    aux.append(cross_over(pop_selected[j], pop_selected[j+1]))
                enfants1 = [x[0] for x in aux]
                enfants2 = [x[1] for x in aux]
                pop_selected = enfants1 + enfants2 + [pop_selected[-1]]              
            else : 
                for j in range(1, nb_select, 2) :
                    aux.append(cross_over(pop_selected[j], pop_selected[j+1]))
                enfants1 = [x[0] for x in aux]
                enfants2 = [x[1] for x in aux]
                pop_selected = enfants1 + enfants2

            # mutation de la population sélectionné
            pop_selected = best_parent + mutation(pop_selected, taux_mutation, current_date, stay_time)

            # remplir la population sélectionné
            pop = fill_pop(date_rdv, stay_time, pop_selected, nb_select, size_pop)

        # évolution des scores
        scores = []
        for i in range(size_pop) :
            scores.append(compute_fitness(pop[i], date_rdv, current_schedule, stay_time, max_occupancy, f_dispo, f_ecart, f_malus, f_bonus = 1))
        
        # selection des meilleurs résultats
        pop_selected, best_score = selection(pop, scores, nb_select)
        scores.append(best_score)
        #print("------------------------------------------------------")

    current_schedule = update_schedule(current_schedule, pop, stay_time)
    current_ecart_type = np.std(current_schedule[date_rdv:])
    return current_schedule, scores, current_ecart_type, pop[0][0], pop[0][1]

def create_new_schedule(size_pop, max_iteration, taux_mutation, max_occupancy, current_date, current_schedule_init, stays_time, dates_rdv, f_dispo = 1, f_ecart = 1, f_malus = 1, f_bonus = 1):
    # copie du planning de départ
    current_schedule = current_schedule_init.copy()

    # evolution de l'écart type
    ecart_type_evol = []
    ecart_type_evol.append(np.std(current_schedule_init[current_date:]))

    # evolution des scores
    scores = [0]

    # evolution du génome
    predict_time_evol = [0]
    add_date_evol = [0]

    # ordonnancement des différents patients
    for k in range(len(stays_time)) :
        print("**********", k, "*********")
        current_schedule, score, ecart_type, predict_time, add_date = genetique(size_pop, max_iteration, taux_mutation, max_occupancy, current_date, current_schedule, stays_time[k], dates_rdv[k], f_dispo, f_ecart, f_malus)
        scores.append(score)
        ecart_type_evol.append(ecart_type)
        predict_time_evol.append(predict_time)
        add_date_evol.append(add_date)
    return current_schedule, scores, ecart_type_evol, predict_time_evol, add_date_evol

######################################################
######################## test ########################

# données initiales
size_pop = 50                   # taille de la population
size_patients_a_placer = 50     # nombre de patients à déposer
size_chirurgien = 20            # nombre de chirurgien
size_blocs = 100                # nombre de blocs opératoire
size_study = 365                # intervalle de test
max_iteration = 400             # nombre de générations de la population
taux_mutation = 0.1             # taux de mutation  
max_occupancy = 50              # nombre de lits disponibles
max_work_chirurgien = 8         # durée de travail d'un chirurgien par jour
max_work_bloc = 15              # durée disponible par jour d'un bloc
current_date = 50               # date actuelle

# nombre de lits occupés en fonction du temps
current_schedule_init = [rd.randint(20,25) for i in range(size_study)]
#current_schedule_init = [0]*size_study

# temps occupés des chirurgiens
current_schedule_chirurgien_init = []*size_study
for i in range(size_study) :
    if (i%6 == 0) or (i%7 == 0) :
        current_schedule_chirurgien_init = max_work_chirurgien
    else : 
        rd.randint(0,max_work_chirurgien-3)

# temps occupés des blocs
current_schedule_blocs_init = []*size_study
for i in range(size_study) :
    if (i%6 == 0) or (i%7 == 0) :
        current_schedule_blocs_init = max_work_bloc
    else : 
        rd.randint(0,max_work_bloc-10)

# nombre de patients à placer
stays_time = [rd.randint(1,2) for i in range(size_patients_a_placer)]
# stays_time = [1 for i in range(size_patients_a_placer)]
date_rdv = [current_date for i in range(size_patients_a_placer)]

# nouveau planning des lits avec l'évolution de l'écart-type
current_schedule, scores, ecart_type_evol, predict_time_evol, add_date_evol = create_new_schedule(size_pop, max_iteration, taux_mutation, max_occupancy, current_date, current_schedule_init, stays_time, date_rdv)

######################### résultats test #########################

print("avant :", ecart_type_evol[0], "après :", ecart_type_evol[len(stays_time)-1])
print("taux de variation :", 1 - ecart_type_evol[len(stays_time)-1]/ecart_type_evol[0])
print("moyenne predict_time :", np.mean(predict_time_evol))

plt.subplot(2,2,1)
plt.title("Evolution des dates d'admission")
plt.plot([i for i in range(size_patients_a_placer+1)],add_date_evol)

plt.subplot(2,2,2)
plt.title("Evolution du predict time")
plt.plot([i for i in range(size_patients_a_placer+1)],predict_time_evol)

plt.subplot(2,2,3)
plt.title("Evolution de l'écart_type")
plt.plot([i for i in range(size_patients_a_placer+1)],ecart_type_evol)

plt.subplot(2,2,4)
plt.title("Evolution du nombre de lits occupés")
plt.plot([i for i in range(size_study)],current_schedule_init, label="init")
plt.plot([i for i in range(size_study)],current_schedule, label="final")
plt.legend()

plt.tight_layout()  # Ajuste automatiquement la disposition des sous-graphes pour éviter les chevauchements
plt.show()