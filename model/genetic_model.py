# genetic algorithm search of the one max optimization problem
from numpy.random import randint
from numpy.random import rand
import random
 
# objective function
def onemax(x):
	return -sum(x)
 
# tournament selection
def selection(pop, scores, k=10):
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
 
# mutation operator --> r_mut is unused so does not afect us
def mutation(bitstring, r_mut):
    if random.uniform(0, 1) < 0.5:
        # Mutation in proteins features.
        for i in range(55):
            if rand() < 1/55:
                # flip the bit
                bitstring[i] = 1 - bitstring[i]
    else:
        # Mutation in the ligand features.
        flipped = False
        bit = 0
        for i in range(55, len(bitstring)):
            if rand() < 1/(len(bitstring) - 55):
                # flip the bit
                bit = bitstring[i]
                bitstring[i] = 1 - bitstring[i]
        if flipped:
            i_prime = random.randint(55, len(bitstring))
            while bitstring[i_prime] != bit:
                i_prime = random.randint(55, len(bitstring))
            bitstring[i_prime] = bit

def get_index_vals(num, start, end):
    ret_list = []
    for i in range(num):
        new_val = random.randint(start, end)
        while new_val in ret_list:
            new_val = random.randint(start, end)
        ret_list += [new_val]
    return sorted(ret_list)

# r_mut here signifies how many indices to change per population element
def mutation_with_index(index_list, r_mut):
    mutation_rate = r_mut / len(index_list)
    for i in range(len(index_list)):
        if random.uniform(0, 1) < mutation_rate:
            if(index_list[i] < 55):
                new_val = random.randint(0, 55-1)
                while new_val in index_list:
                    new_val = random.randint(0, 55-1)
            else:
                new_val = random.randint(55, 457-1)
                while new_val in index_list:
                    new_val = random.randint(55, 457-1)
            index_list[i] = new_val

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
	    protein_feature_selection = get_index_vals(15, 0, 55-1)
	    ligand_feature_selection = get_index_vals(15, 55, n_bits-1)
	    pop += [protein_feature_selection + ligand_feature_selection]
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
				print(">%d, new best = %.3f." % (gen, scores[i]))
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
				mutation_with_index(c, r_mut)
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
