class Genetic:
    
    def __init__(self,pop_size,nlayers,max_nfilters,max_sfilters):
        self.pop_size = pop_size
        self.nlayers = nlayers
        self.max_nfilters = max_nfilters
        self.max_sfilters = max_sfilters
        self.max_acc = 0
        self.best_arch = np.zeros((1,6))
        self.gen_acc = []
    
    def generate_population(self):
        np.random.seed(0)
        pop_nlayers = np.random.randint(1,self.max_nfilters,(self.pop_size,self.nlayers))
        pop_sfilters = np.random.randint(1,self.max_sfilters,(self.pop_size,self.nlayers))
        pop_total = np.concatenate((pop_nlayers,pop_sfilters),axis=1)
        return pop_total
    
    def select_parents(self,pop,nparents,fitness):
        parents = np.zeros((nparents,pop.shape[1]))
        for i in range(nparents):
            best = np.argmax(fitness)
            parents[i] = pop[best]
            fitness[best] = -99999
        return parents
    
    def crossover(self,parents):
        nchild = self.pop_size - parents.shape[0]
        nparents = parents.shape[0]
        child = np.zeros((nchild,parents.shape[1]))
        for i in range(nchild):
            first = i % nparents
            second = (i+1) % nparents
            child[i,:2] = parents[first][:2]
            child[i,2] = parents[second][2]
            child[i,3:5] = parents[first][3:5]
            child[i,5] = parents[second][5]
        return child

    def mutation(self,child):
        for i in range(child.shape[0]):
            val = np.random.randint(1,6)
            ind = np.random.randint(1,4) - 1
            if child[i][ind] + val > 100:
                child[i][ind] -= val
            else:
                child[i][ind] += val
            val = np.random.randint(1,4)
            ind = np.random.randint(4,7) - 1
            if child[i][ind] + val > 20:
                child[i][ind] -= val
            else:
                child[i][ind] += val
        return child
    
    def fitness(self,pop,X,Y,epochs):
        pop_acc = []
        for i in range(pop.shape[0]):
            nfilters = pop[i][0:3]
            sfilters = pop[i][3:]
            model = CNN(nfilters,sfilters)
            H = model.fit(X,Y,batch_size=32,epochs=epochs)
            acc = H.history['accuracy']
            pop_acc.append(max(acc)*100)
        if max(pop_acc) > self.max_acc:
            self.max_acc = max(pop_acc)
            self.best_arch = pop[np.argmax(pop_acc)]
        self.gen_acc.append(max(pop_acc))
        return pop_acc
    
    def smooth_curve(self,factor,gen):
        smoothed_points = []
        for point in self.gen_acc:
            if smoothed_points:
                prev = smoothed_points[-1]
                smoothed_points.append(prev*factor + point * (1-factor))
            else:
                smoothed_points.append(point)
        plt.plot(range(gen+1),smoothed_points,'g',label='Smoothed training acc')
        plt.xticks(np.arange(gen+1))
        plt.legend()
        plt.title('Fitness Accuracy vs Generations')
        plt.xlabel('Generations')
        plt.ylabel('Fitness (%)')
        plt.show()
