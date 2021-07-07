# genetic algorithm search of the one max optimization problem
from numpy.random import randint
from numpy.random import rand
 
# objective function
def onemax(x):
	return -sum(x)
 
# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]
 
# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]
 
# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

def get_binary_vals(percent, num_bits):
    import random
    ret_val = [0] * num_bits
    for i in range(num_bits):
        if random.uniform(0, 1) < percent:
            ret_val[i] = 1
    return ret_val

# genetic algorithm
def genetic_algorithm(objective, X, y, n_bits, n_iter, n_pop, r_cross, r_mut):
	# initial population of random bitstring
	pop = []
	for _ in range(n_pop):
	    protein_feature_selection = get_binary_vals(0.273, 55)
	    ligand_feature_selection = get_binary_vals(0.273/8, n_bits - 55)
	    pop += [protein_feature_selection + ligand_feature_selection]
	protein_f_s = sum([sum(_[:55]) for _ in pop])/len(pop)
	ligand_f_s = sum([sum(_[55:]) for _ in pop])/len(pop)
	print("Average number of features(protein, ligand) selected = (", protein_f_s, ", ", ligand_f_s, ")")
	# keep track of best solution
	best, best_eval = 0, 100 #objective(pop[0], X, y)
	# enumerate generations
	for gen in range(n_iter):
		print("Generation - ", gen)
		# evaluate all candidates in the population
		scores = objective(pop, X, y)
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				#print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
				print(">%d, new best = %.3f. Features(%d, %d)" % (gen, scores[i], sum(best[:55]), sum(best[55:])))
		# select parents
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
	return [best, best_eval]

if False:
    # define the total iterations
    n_iter = 100
    # bits
    n_bits = 500 #20
    # define the population size
    n_pop = n_bits * 5 #100
    # crossover rate
    r_cross = 0.9
    # mutation rate
    r_mut = 1.0 / float(n_bits)
    # perform the genetic algorithm search
    best, score = genetic_algorithm(onemax, n_bits, n_iter, n_pop, r_cross, r_mut)
    print('Done!')
    print('f(%s) = %f' % (best, score))
