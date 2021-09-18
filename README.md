# Pokemon-Simulations
This program is a test of an algorithm to try to solve the mixed strategy nash equilibrium, (p) of a large payout matrix for a zero sum, symmetric game.
it does this by using projected gradient decent of the population's expected payout vs a linear combination of the population and a "counter" population
right now the code is designed to work on a meta where:
  -all pokemon type combinations are possible, and all pokemon have equal stats
  -all pokemon have acess to a complete movepool, and all moves do the same amount of damage, and never miss
  -there are no abilites or status effects
  -matches are 1v1
under these conditions, the winner of the battle is determined by which pokemon has the higher multiplier (super effective * stab bonus) so it is possible to precalculate the payoff matrix, which makes calculating the payout a matter of matrix multiplication with a payout matrix M

the code has 2 projected gradient decent methods implemented, Projected gradient descent with momentum and Projected gradient decent with ADAM
the implementations for both are taken with insipration from this stack exchange
https://datascience.stackexchange.com/questions/31709/adam-optimizer-for-projected-gradient-descent
