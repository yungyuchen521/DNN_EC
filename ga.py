import random
import gc

from sklearn.metrics import accuracy_score

from individual import Individual


class GA:
    EVAL_BY_ACC = "acc"
    EVAL_BY_LOSS = "loss"

    def __init__(
        self,
        pop_sz: int,
        offspr_sz: int,      # number of offsprings to be created per _evolve
        recomb_prob: float,
        mutate_prob: float,
        X,                   # embeddings of images = feature_extractor(imgs)
        y,                   # labels of images
        **individual_kwargs, # arguments for Individual
    ):
        assert pop_sz > offspr_sz
        assert 0 <= recomb_prob <= 1
        assert 0 <= mutate_prob <= 1

        self.pop_sz: int = pop_sz
        self.population: list[Individual] = [
            Individual(**individual_kwargs) for _ in range(pop_sz)
        ]

        self.X = X
        self.y_true = y

        self.offspr_sz: int = offspr_sz
        self.recomb_prob: float = recomb_prob
        self.mutate_prob: float = mutate_prob

        self._best_individual = max(self.population, key=lambda x: self._evaluate(x))

    @property
    def best_individual(self) -> Individual:
        return self._best_individual

    @property
    def best_fitness(self) -> float:
        return self._evaluate(self._best_individual)

    def run(self, max_iter: int, goal: float) -> Individual:
        for i in range(max_iter):
            print(f"===== Iteration {i+1} =====", end=" ")
            self._evolve()
            print(f"Best fitness = {self.best_fitness}")

            if self.best_fitness >= goal:
                break

        return self._best_individual

    def _evolve(self):
        offspr_lst = []
        while len(offspr_lst) < self.offspr_sz:
            p1 = self._select()
            p2 = self._select()

            if random.random() > self.recomb_prob:
                offspr_lst += [Individual.copy(p1), Individual.copy(p2)]
            else:
                c1, c2 = p1.recombine(p2)
                offspr_lst += [c1, c2]

        self._eliminate_oldest()
        for ind in self.population:
            ind.increment_age()
            if random.random() <= self.mutate_prob:
                ind.mutate()

        self.population += offspr_lst

    def _select(self) -> Individual:
        tournament_size = 2
        tournament_pool = random.sample(self.population, tournament_size)
        return max(tournament_pool, key=lambda ind: self._evaluate(ind))

    def _eliminate_oldest(self):
        key = lambda x: x.age
        self.population.sort(key=key, reverse=True)

        del self.population[: self.pop_sz]
        gc.collect()

    def _eliminate_worst(self):
        key = lambda x: self._evaluate(x)
        self.population.sort(key=key)

        del self.population[: self.pop_sz]
        gc.collect()

    def _evaluate(self, ind: Individual) -> float:
        if ind.acc is None:
            y_pred = [(1 if ind.predict(x) > 0.5 else 0) for x in self.X]
            ind.set_acc(accuracy_score(self.y_true, y_pred))

        return ind.acc
