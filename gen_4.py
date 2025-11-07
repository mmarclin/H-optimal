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
        predict_time = rd.randint(1, size_study-current_date-stay_time)
        add_date = rd.randint(current_date, current_date + predict_time)
        pop.append(generate_schedule_bed(predict_time, add_date))
    return pop

# ordonanncement de la date d'admission sur le planning de lit
def generate_schedule_bed(predict_time, add_date):
    return [predict_time, add_date]

# Calcul de score
def compute_fitness(schedule_bed_genome, current_date, current_schedule, stay_time, max_occupancy, f_dispo = 1, f_ecart = 3, f_bonus = 1):
    # score = disponibilité + variation ecart type + sous la courbe + malus
    score = 0

    # calcul disponibilite lit : 0 ou 1
    schedule_bed = current_schedule.copy()
    current_moy = np.mean(schedule_bed[current_date:])
    
    disponibility = all(schedule_bed[schedule_bed_genome[1] + i] <= max_occupancy for i in range(1, stay_time))
    for i in range(1,stay_time) : 
        schedule_bed[schedule_bed_genome[1] + i] += 1

    # calcul disponibilite chirurchien et bloc 
    
    # bonus : jour où les lits sont peu remplis
    # malus : week end
    malus_week_end = 0
    moy_sejour = np.mean(schedule_bed[schedule_bed_genome[1]:schedule_bed_genome[1]+stay_time])
    for i in range(stay_time) :
        if (schedule_bed_genome[1] + i)%7 == 0 or (schedule_bed_genome[1] + i)%7 == 6 :
            malus_week_end += 1/stay_time

    bonus_min = 1/( 1+np.exp(moy_sejour - min(schedule_bed[current_date:])) )

    # evalue si l'écart type global diminue
    ecart_type_global = np.std(schedule_bed[current_date:])
    if (ecart_type_global != 0):
        var_normalized = 1/(1 + np.exp(-(1/ecart_type_global)))
    else : 
        var_normalized = 1

    score = f_dispo*disponibility + f_ecart*var_normalized + f_bonus*bonus_min
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
def mutation(pop_selected, taux_mutation, current_date, stay_time, size_study):
    for i in range(len(pop_selected)):
        if rd.random() < taux_mutation:
            num_gen_mutated = rd.randint(0,1)
            if num_gen_mutated == 0 :
                predict_time = rd.randint(1, size_study-current_date-stay_time)
                if (pop_selected[i][1] <= predict_time - stay_time) :
                    pop_selected[i][0] = predict_time
            else :
                add_date = rd.randint(current_date, current_date + pop_selected[i][0])
                pop_selected[i][1] = add_date
    return pop_selected

# remplir la population restant
def fill_pop(current_date, stay_time, pop_selected, nb_select, size_pop, size_study) :
    pop = pop_selected.copy()
    for i in range(size_pop - nb_select) :
        predict_time = rd.randint(1, size_study-current_date-stay_time)
        add_date = rd.randint(current_date, current_date + predict_time)
        pop.append(generate_schedule_bed(predict_time, add_date))
    return pop

# Mise à jour du planning après ajout des patients
def update_schedule(current_schedule, pop_selected, stay_time) :
    for i in range(stay_time) :
        current_schedule[pop_selected[0][1] + i] += 1
    return current_schedule

# lancement algorithme génétique
def genetique(size_pop, max_iteration, taux_mutation, max_occupancy, current_date, current_schedule, stay_time, date_rdv, size_study, f_dispo = 1, f_ecart = 3, f_bonus = 1):
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
            pop_selected = best_parent + mutation(pop_selected, taux_mutation, current_date, stay_time, size_study)

            # remplir la population sélectionné
            pop = fill_pop(date_rdv, stay_time, pop_selected, nb_select, size_pop, size_study)

        # évolution des scores
        scores = []
        for i in range(size_pop) :
            scores.append(compute_fitness(pop[i], date_rdv, current_schedule, stay_time, max_occupancy, f_dispo, f_ecart, f_bonus))
        
        # selection des meilleurs résultats
        pop_selected, best_score = selection(pop, scores, nb_select)
        scores.append(best_score)
        #print("------------------------------------------------------")

    current_schedule = update_schedule(current_schedule, pop, stay_time)
    current_ecart_type = np.std(current_schedule[date_rdv:])
    return current_schedule, scores, current_ecart_type, pop[0][0], pop[0][1]

def create_new_schedule(size_pop, max_iteration, taux_mutation, max_occupancy, current_date, current_schedule_init, stays_time, dates_rdv, size_study, f_dispo = 1, f_ecart = 3, f_bonus = 1):
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
        #print("**********", k, "*********")
        current_schedule, score, ecart_type, predict_time, add_date = genetique(size_pop, max_iteration, taux_mutation, max_occupancy, current_date, current_schedule, stays_time[k], dates_rdv[k], size_study, f_dispo, f_ecart, f_bonus)
        scores.append(score)
        ecart_type_evol.append(ecart_type)
        predict_time_evol.append(predict_time)
        add_date_evol.append(add_date)
    return current_schedule, scores, ecart_type_evol, predict_time_evol, add_date_evol

def search_combi_preponderance(start_num, end_num, pas) :
    # liste de [f_dispo, f_ecart_type, f_undercurb, f_we]

    size_pop = 50                   # taille de la population
    size_patients_a_placer = 10     # nombre de patients à déposer
    size_chirurgien = 20            # nombre de chirurgien
    size_blocs = 100                # nombre de blocs opératoire
    size_study = 50                 # intervalle de test
    max_iteration = 300             # nombre de générations de la population
    taux_mutation = 0.1             # taux de mutation  
    max_occupancy = 30              # nombre de lits disponibles
    max_work_chirurgien = 8         # durée de travail d'un chirurgien par jour
    max_work_bloc = 15              # durée disponible par jour d'un bloc
    current_date = 0                # date actuelle

    current_schedule_init = [rd.randint(20,25) for i in range(size_study)]
    stays_time = [rd.randint(1,2) for i in range(size_patients_a_placer)]
    dates_rdv = [current_date for i in range(size_patients_a_placer)]

    # meilleur combinaison
    list_combi = []
    evol_combi = []
    evol_best_score = [0]
    best_score = 0

    # création des combinaisons
    values_dispo = np.arange(start_num,end_num,pas)
    for i in values_dispo :
        print(i)
        for j in values_dispo  :
            for k in values_dispo  :
                list_combi.append([i,j,k])

    L = len(list_combi)
    print("---------------", L, "----------------")
    
    evol_combi.append(list_combi[0])

    # test
    for i in range(L):
        print("---------------", i, "----------------")
        current_schedule, scores, ecart_type_evol, predict_time_evol, add_date_evol = create_new_schedule(size_pop, max_iteration, taux_mutation, max_occupancy, current_date, current_schedule_init, stays_time, dates_rdv, size_study, list_combi[i][0], list_combi[i][1], list_combi[i][2])
        score = ecart_type_evol[0] - ecart_type_evol[len(stays_time)-1]
        if (score > best_score) :
            best_score = score
            evol_best_score.append(best_score)
            evol_combi.append(list_combi[i])


    return evol_combi, evol_best_score, current_schedule_init, current_schedule

def affichage(evol_combi, evol_best_score, current_schedule_init, current_schedule):
    L = len(evol_combi)
    index = [i for i in range(L)]

    evol_dispo = []
    evol_ecart = []
    evol_malus = []
    for i in range(L) :
        evol_dispo.append(evol_combi[i][0])
        evol_ecart.append(evol_combi[i][1])
        evol_malus.append(evol_combi[i][2])
    
    print("Facteur dispo      ", evol_combi[-1][0])
    print("Facteur ecart type ", evol_combi[-1][1])
    print("Facteur bonus      ", evol_combi[-1][2])
    print("Meilleur score     ", evol_best_score[-1])

    plt.subplot(2,2,1)
    plt.title("Facteur dispo")
    plt.plot(index, evol_dispo, label="dispo")
    plt.plot(index, evol_ecart, label="ecart")
    plt.plot(index, evol_malus, label="bonus")
    plt.legend()

    plt.subplot(2,2,3)
    plt.title("Evolution du nombre de lits")
    plt.plot([i for i in range(len(current_schedule))] ,current_schedule_init, label="init")
    plt.plot([i for i in range(len(current_schedule))] ,current_schedule, label = "final")
    plt.legend()

    plt.tight_layout()  # Ajuste automatiquement la disposition des sous-graphes pour éviter les chevauchements
    plt.show()
    return 0

######################################################
######################## test ########################

start_num = 1
end_num = 1
pas = 1
evol_combi, evol_best_score, current_schedule_init, current_schedule = search_combi_preponderance(start_num, end_num, pas)
affichage(evol_combi, evol_best_score, current_schedule_init, current_schedule)
