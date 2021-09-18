# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:19:55 2021

@author: mLLL
"""

import numpy as np
import itertools as itr
from matplotlib import pyplot as plt
from numba import jit, njit
rng = np.random.default_rng()

types = np.array(["Normal  ", "Fire    ", "Water   ", "Electric", "Grass   ",
                  "Ice     ", "Fighting", "Poison  ", "Ground  ", "Flying  ", 
                  "Psychic ", "Bug     ", "Rock    ", "Ghost   ", "Dragon  ",
                  "Dark    ", "Steel   ", "Fairy   ", "None    "])

[Normal, Fire, Water, Electric, Grass, Ice,
 Fighting, Poison, Ground, Flying, Psychic,
 Bug, Rock, Ghost, Dragon, Dark, Steel, Fairy, NoneType] = [i for i in range(19)]

damage_array = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1/2, 0, 1, 1, 1/2, 1, 1],
                    [1, 1/2, 1/2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1/2, 1, 1/2, 1, 2, 1, 1],
                    [1, 2, 1/2, 1, 1/2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1/2, 1, 1, 1, 1],
                    [1, 1, 2, 1/2, 1/2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 1/2, 1, 1, 1, 1],
                    [1, 1/2, 2, 1, 1/2, 1, 1, 1/2, 2, 1/2, 1, 1/2, 2, 1, 1/2, 1, 1/2, 1, 1],
                    [1, 1/2, 1/2, 1, 2, 1/2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1/2, 1, 1],
                    [2, 1, 1, 1, 1, 2, 1, 1/2, 1, 1/2, 1/2, 1/2, 2, 0, 1, 2, 2, 1/2, 1],
                    [1, 1, 1, 1, 2, 1, 1, 1/2, 1/2, 1, 1, 1, 1/2, 1/2, 1, 1, 0, 2, 1],
                    [1, 2, 1, 2, 1/2, 1, 1, 2, 1, 0, 1, 1/2, 2, 1, 1, 1, 2, 1, 1],
                    [1, 1, 1, 1/2, 2, 1, 2, 1, 1, 1, 1, 2, 1/2, 1, 1, 1, 1/2, 1, 1],
                    [1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1/2, 1, 1, 1, 1, 0, 1/2, 1, 1],
                    [1, 1/2, 1, 1, 2, 1, 1/2, 1/2, 1, 1/2, 2, 1, 1, 1/2, 1, 2, 1/2, 1/2, 1],
                    [1, 2, 1, 1, 1, 2, 1/2, 1, 1/2, 2, 1, 2, 1, 1, 1, 1, 1/2, 1, 1],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1/2, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1/2, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1/2, 1, 1, 1, 2, 1, 1, 2, 1, 1/2, 1, 1/2, 1],
                    [1, 1/2, 1/2, 1/2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1/2, 2, 1],
                    [1, 1/2, 1, 1, 1, 1, 2, 1/2, 1, 1, 1, 1, 1, 1, 2, 2, 1/2, 1, 1]])

### building keys ###
'''
every combination of 2 pokemon types (including none) has a unique type ID
'''
pokemon_types = np.zeros((171,2), dtype = int)
types_to_type_id = np.zeros((19,19),dtype=int) - 1
for n,(i,j) in zip(range(171),itr.combinations(range(19),r=2)):
    pokemon_types[n] = [i,j]
    types_to_type_id[i,j] = n
    types_to_type_id[j,i] = n
    
'''
Every combination of 4 attack types has a unique attack ID
for simplicity duplicate attack types are used
'''
pokemon_movesets = np.empty((3060,4),dtype = int)
attks_to_Move_set_id = np.zeros((18,18,18,18),dtype=int) - 1
for n,atks in zip(range(3060),itr.combinations(range(18),r=4)):
    pokemon_movesets[n] = atks
    for i,j,k,l in itr.permutations(atks):
        attks_to_Move_set_id[i,j,k,l] = n

################################
### Precalculation of Damage ###
################################
damage_array_2_type = np.zeros((171,18)) #defender, attack,
'''the multiplier for a pokemon getting attacked (without stab)'''
for [i,j],[[dt1,dt2],atk] in zip(itr.product(range(171),range(18)),
                                 itr.product(itr.combinations(range(19),r=2),
                                             range(18))):
    damage_array_2_type[i,j] = damage_array[atk,dt1] * damage_array[atk,dt2]

pokemon_dmg_calc = np.zeros((171,171,18)) #defender,attacker,attacks
'''the multiplier for a pokemon getting attacked (with stab of the attacker considered)'''
block = zip(itr.product(range(171),range(18)),
            itr.product(itr.combinations(range(19),r=2),range(18)))
for [j,k],[[at1,at2],atk] in block:
    stab = 1 + (np.equal(atk, at1) + np.equal(atk, at2))/2 #stab
    pokemon_dmg_calc[:,j,k] = damage_array_2_type[:,atk] * stab

pokemon_fight_calc = np.zeros((171,171,3060)) #defender,attacker,moveset
#the maximum damage a pokemon with a given moveset can do to the defender
for i,atks in zip(range(3060),itr.combinations(range(18),r=4)):
    pokemon_fight_calc[:,:,i] = np.max(pokemon_dmg_calc[:,:,atks],axis = -1)

##########################
#### helper functions ####
##########################

def other_vs_sample_payout_tensor(types,movesets):
    '''
    for finding out how a particular collection of pokemon of types, with movesets 
    fare against the general population
    
    types: an array of m pokemon type ids
    movesets: an array of m pokemon moveset ids

    Returns:
    an m x 171 x 3060 tensor, with a 1 if the other wins, a 0 for draw and a -1
    for loss
    '''
    sample_attack_other = pokemon_fight_calc[:,types,movesets].T
    other_attack_sample = pokemon_fight_calc[types,:,:]
    return np.sign(other_attack_sample - sample_attack_other[:,:,np.newaxis])
    
@njit(parallel = True)
def project_into_pdf(x):
    '''
    projects an array into a pdf (or very close) at least
    outputs y such that ||x-y|| is minimized, 
    this is used for projected gradient decent
    '''
    p = x - (np.sum(x) - 1)/x.size
    dp = np.sum(np.fmin(0,p))/np.sum(p>0)
    while (np.sum(np.fmin(dp,p))**2>1e-16):
        dp -= np.sum(np.fmin(dp,p))/np.sum(p>dp)
    return p-np.fmin(p,dp)

#@njit(parallel = True)
def counter(pop):
    '''
    calculates, a counter to the population input by pop
    this isnt necciarily the best counter population, but it is at least a
    good enough one
    '''
    return project_into_pdf(pop@payout_matrix*100+pop)

#@njit(parallel = True)
def update_pop(pop,counter_pop,grad1,r=0.9):  
    '''
    uses projected gradient decent with momentum to maximize the winrate against
    pop and the counter pop, with r as the weighting ratio between them
    
    this method converges better than ADAM but is slower initially
    
    '''
    grad = ((pop * r + counter_pop * (1-r)) @ payout_matrix) #unprojected gradient
    new_pop_1 = project_into_pdf(grad * 0.001 + pop) 
    proj_grad = grad1*0.9 + 0.1*(new_pop_1 - pop)/0.001
    new_pop = project_into_pdf(proj_grad * 0.001 + pop) 
    counter_new_pop = counter(new_pop)
    return new_pop,counter_new_pop,proj_grad

def update_pop_adam(pop,counter_pop,grad_m,grad_v,t,r=0.9,alpha0=0.001):  
    '''
    uses projected gradient decent with ADAM to maximize the winrate against
    pop and the counter pop, with r as the weighting ratio between them
    this method converges better than ADAM but is slower initially
    '''
    grad = ((pop * r + counter_pop * (1-r)) @ payout_matrix) #unprojected gradient
    
    proj_grad = (project_into_pdf(grad * 0.001 + pop) - pop)/0.001
    m = grad_m* 0.9 +   0.1* proj_grad
    v = grad_v* 0.99 + 0.01* proj_grad**2
    
    alpha = alpha0 * np.sqrt(1 - .99**t)/(1-.9**t)
    new = pop + alpha * m/(np.sqrt(v)+1e-8)
    new_pop = project_into_pdf(new)
    
    counter_new_pop = counter(new_pop)
    return new_pop,counter_new_pop,m,v

############################
##### simulation begins ####
############################

num_trials = 300000
pokes = np.array([i for i in range(171) for j in range(6)])
n = pokes.size
movesets  = rng.choice(range(3060),size=n)
payout_matrix = other_vs_sample_payout_tensor(pokes,movesets)[:,pokes,movesets]

pop = rng.dirichlet(np.ones(n))
counter_pop = counter(pop)

prev_pop = np.zeros_like(pop)
exploitability = pop@payout_matrix@counter_pop
grad = np.zeros_like(pop)
grad_sq = np.zeros_like(pop)
reset = 0
p = 0.8
for i in range(num_trials):
    #new_pop,new_counter,grad = update_pop(pop,counter_pop,grad,p)
    new_pop,new_counter, grad,grad_sq = update_pop_adam(pop,counter_pop,grad,grad_sq,i+1,p)   

    exploitability = (new_pop) @ payout_matrix @ new_counter
    payout = pop @ payout_matrix @ new_pop
    pop = new_pop
    counter_pop = new_counter 
    
    print("\riteration:%i,"%i,
          "benifit from previous population:%+1.5e,"%np.abs(payout),
          "exploitability:%+0.5e"%exploitability,end='   ')
    
    if (exploitability<1e-3): 
        '''
        when the exploitability goes below a threshold swap out 10% the pokemon
        the only valid canidates are those with 0 population for 2 iterations
        and arent being used by the counter population
        '''
        canidates = np.logical_and(np.logical_and(pop == 0,prev_pop == 0),counter_pop==0)
        n_new = min(n//10,sum(canidates))
        rerolls = rng.choice(np.arange(n)[canidates],n_new,False)
        attempts = 0
        
        
        print("\ncurrent population")
        for i,j,k in zip(pokes,movesets,pop):
            if k>0:
                print(*types[pokemon_types[i]],":",*types[pokemon_movesets[j]],k)
            
        
        outsiders = other_vs_sample_payout_tensor(pokes[pop>0],movesets[pop>0])
        outsider_scores = np.tensordot(pop[pop>0],outsiders,(0,0))
        num_better_than = np.sum(outsider_scores > 0)
        worst_matchup = np.unravel_index(np.argmax(outsider_scores),outsider_scores.shape)
        print("the above strategy is worse than %i other type movesets combinations"%num_better_than)
        print("it's worst matchup is:",
              *types[pokemon_types[worst_matchup[0]]],':',
              *types[pokemon_movesets[worst_matchup[1]]],end='\n\n')
        print("searching for new canidates")
        while (exploitability<1e-3 and attempts < 30):
            attempts += 1
            movesets[rerolls] = rng.choice(3060,n_new)
            
            payout_matrix = other_vs_sample_payout_tensor(pokes,movesets)[:,pokes,movesets]
            
            exploitability = (pop) @ payout_matrix @ counter(pop)
        prev_pop = pop 
        print("new exploitability", exploitability,"  number of attempsts:",attempts)



     
        
        
    
#'''
