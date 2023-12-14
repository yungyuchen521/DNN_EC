import random
import gc
from copy import deepcopy

import numpy as np
from sklearn.metrics import accuracy_score

from .individual import BaseIndividual, GaIndividual, EsIndividual


class EC:
    EVAL_BY_ACC = "acc"
    EVAL_BY_LOSS = "loss"

    def __init__(self, pop_sz: int, offspr_sz: int, X: np.ndarray, y: np.ndarray):
        """
        offspr_sz: number of offsprings to be created per _evolve
        X: embeddings of images = feature_extractor(imgs)
        y: labels of images
        """

        self.pop_sz: int = pop_sz
        self.population: list[BaseIndividual]

        self.offspr_sz: int = offspr_sz

        self.X = X
        self.y_true = y

        self.best_individual: BaseIndividual = None
        self._update_best_individual()

    @property
    def best_fitness(self) -> float:
        return self._evaluate(self.best_individual)

    def run(self, max_iter: int, goal: float, report_period: int = 100) -> list[float]:
        history = []

        for i in range(max_iter):
            self._evolve()
            assert len(self.population) == self.pop_sz
            self._update_best_individual()
            history.append(self.best_fitness)

            if (i + 1) % report_period == 0:
                print(f"===== Iteration {i+1} ===== Best fitness = {self.best_fitness}")

            if self.best_fitness >= goal:
                break

        return history

    def _evolve(self):
        raise NotImplementedError

    def _update_best_individual(self):
        ind = max(self.population, key=lambda x: self._evaluate(x))

        if (self.best_individual is None) or (ind.fitness > self.best_fitness):
            self.best_individual = deepcopy(ind)  # escape from _eliminate_oldest

    def _eliminate_oldest(self, n: int):
        key = lambda x: x.age
        self.population.sort(key=key, reverse=True)

        del self.population[: n]
        gc.collect()

    def _eliminate_worst(self, n: int):
        key = lambda x: self._evaluate(x)
        self.population.sort(key=key)

        del self.population[: n]
        gc.collect()

    def _evaluate(self, ind: BaseIndividual) -> float:
        if ind.acc is None:
            y_pred = [(1 if ind.predict(x) > 0.5 else 0) for x in self.X]
            ind.set_acc(accuracy_score(self.y_true, y_pred))

        return ind.acc


class GeneticAlgorithm(EC):
    ELIMINATE_BY_AGE = "age"
    ELIMINATE_BY_FITNESS = "fitness"

    def __init__(
        self,
        pop_sz: int,
        offspr_sz: int,
        recomb_prob: float,
        mutate_prob: float,
        eliminate_by: str,
        X: np.ndarray,
        y: np.ndarray,
        **ind_kwargs,
    ):
        """
        ind_kwargs
            - dim_list
            - ga_operator_callbacks
        """
        assert pop_sz > offspr_sz
        assert 0 <= recomb_prob <= 1
        assert 0 <= mutate_prob <= 1
        assert eliminate_by in (self.ELIMINATE_BY_AGE, self.ELIMINATE_BY_FITNESS)

        self.recomb_prob: float = recomb_prob
        self.mutate_prob: float = mutate_prob
        self.eliminate_by = eliminate_by
        self.population: list[GaIndividual] = [
            GaIndividual(**ind_kwargs) for _ in range(pop_sz)
        ]

        super().__init__(
            pop_sz=pop_sz,
            offspr_sz=offspr_sz,
            X=X,
            y=y,
        )

    def _evolve(self):
        offspr_lst = []
        while len(offspr_lst) < self.offspr_sz:
            p1 = self._select()
            p2 = self._select()

            if random.random() > self.recomb_prob:
                offspr_lst += [GaIndividual.copy(p1), GaIndividual.copy(p2)]
            else:
                c1, c2 = p1.recombine(p2)
                offspr_lst += [c1, c2]

        if self.eliminate_by == self.ELIMINATE_BY_AGE:
            self._eliminate_oldest(self.offspr_sz)
        else:
            self._eliminate_worst(self.offspr_sz)

        for ind in self.population:
            ind.increment_age()
            if random.random() <= self.mutate_prob:
                ind.mutate()

        self.population += offspr_lst

    def _select(self) -> GaIndividual:
        tournament_size = 2
        tournament_pool = random.sample(self.population, tournament_size)
        return max(tournament_pool, key=lambda ind: self._evaluate(ind))


class EvolutionaryStrategy(EC):
    ES_COMMA = "comma"
    ES_PLUS = "plus"

    def __init__(self, pop_sz: int, offspr_sz: int, select_by: str, X, y, **individual_kwargs):
        """
        individual_kwargs:
            - dim_list,
            - step_size
            - tau: coordinate-wise learning rate
            - tau_prime: general learning rate
            - eps: min step size
        """
        assert pop_sz <= offspr_sz
        assert select_by in (self.ES_COMMA, self.ES_PLUS)

        self.select_by: str = select_by
        self.population: list[EsIndividual] = [
            EsIndividual(**individual_kwargs) for _ in range(pop_sz)
        ]

        super().__init__(
            pop_sz=pop_sz,
            offspr_sz=offspr_sz,
            X=X,
            y=y,
        )

    def _evolve(self):
        parents = random.choices(self.population, k=self.offspr_sz)
        offspr_lst = [ind.mutate() for ind in parents]

        if self.select_by == self.ES_COMMA:
            del self.population
            self.population = offspr_lst
            self._eliminate_worst(len(self.population) - self.pop_sz)
        else:
            self.population += offspr_lst
            self._eliminate_worst(len(self.population) - self.pop_sz)
