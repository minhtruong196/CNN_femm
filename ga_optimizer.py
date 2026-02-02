"""
GA + NGNet + FEMM Topology Optimization for IPM Motor

Workflow:
1. GA tạo population (mỗi individual = weights của NGNet)
2. Với mỗi individual:
   - Gọi dxf_part.py để tạo geometry từ weights
   - Gọi plot_fem.py để chạy FEMM simulation
   - Quét adv từ 35-55 độ để tìm max torque
   - Fitness = max torque
3. Selection, crossover, mutation
4. Lặp lại cho đến khi đạt số generations
"""

import os
import sys
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Callable
from pathlib import Path

# Import modules từ project
from dxf_part import generate_geometry, get_ngnet_size
from plot_fem import setup_model, sweep_adv_for_max_torque, close_femm


@dataclass
class GAConfig:
    """Cấu hình cho Genetic Algorithm."""
    population_size: int = 20
    generations: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.2  # Gaussian mutation std
    tournament_size: int = 3
    elitism: int = 2  # Số lượng best individuals giữ lại

    # Bounds cho weights
    weight_min: float = -1.0
    weight_max: float = 1.0

    # Advance angle sweep
    adv_list: List[float] = field(default_factory=lambda: [35, 40, 45, 50, 55])

    # Motor parameters
    Im: float = 11.0   # Biên độ dòng điện (A)
    ini: float = -15.0  # Góc ban đầu (độ)


@dataclass
class Individual:
    """Một cá thể trong population."""
    weights: np.ndarray
    fitness: float = 0.0
    best_adv: float = 0.0

    def copy(self) -> 'Individual':
        return Individual(
            weights=self.weights.copy(),
            fitness=self.fitness,
            best_adv=self.best_adv
        )


class GAOptimizer:
    """Genetic Algorithm optimizer cho topology optimization."""

    def __init__(
        self,
        config: GAConfig,
        tra_path: str,
        base_fem: str,
        output_dir: str = ".",
    ):
        self.config = config
        self.tra_path = tra_path
        self.base_fem = base_fem
        self.output_dir = output_dir

        # Lấy số lượng weights cần thiết
        self.n_weights = get_ngnet_size(tra_path)
        print(f"NGNet weights size: {self.n_weights}")

        # History
        self.history: List[dict] = []
        self.best_individual: Individual = None

        # Counter cho evaluations
        self.eval_count = 0

    def initialize_population(self) -> List[Individual]:
        """Tạo population ban đầu."""
        population = []
        for _ in range(self.config.population_size):
            weights = np.random.uniform(
                self.config.weight_min,
                self.config.weight_max,
                self.n_weights
            )
            population.append(Individual(weights=weights))
        return population

    def evaluate_fitness(self, individual: Individual, verbose: bool = False) -> float:
        """
        Đánh giá fitness của một individual.
        Fitness = max torque khi quét các góc adv.
        """
        self.eval_count += 1

        try:
            # 1. Tạo geometry từ weights
            dxf_path, json_path = generate_geometry(
                weights=individual.weights,
                tra_path=self.tra_path,
                output_dir=self.output_dir,
                verbose=verbose
            )

            # 2. Setup FEMM model
            fem_file, _ = setup_model(
                base_fem=self.base_fem,
                rotor_dxf=dxf_path,
                centroids_json=json_path,
                verbose=verbose
            )

            # 3. Quét adv để tìm max torque
            best_adv, max_torque, _ = sweep_adv_for_max_torque(
                fem_file=fem_file,
                adv_list=self.config.adv_list,
                Im=self.config.Im,
                ini=self.config.ini,
                verbose=verbose
            )

            individual.fitness = max_torque
            individual.best_adv = best_adv

            if verbose:
                print(f"  Eval #{self.eval_count}: fitness={max_torque:.4f} N.m (adv={best_adv}°)")

            return max_torque

        except Exception as e:
            print(f"  Eval #{self.eval_count}: ERROR - {e}")
            individual.fitness = 0.0
            return 0.0

    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Chọn individual bằng tournament selection."""
        tournament = np.random.choice(
            len(population),
            size=self.config.tournament_size,
            replace=False
        )
        best_idx = max(tournament, key=lambda i: population[i].fitness)
        return population[best_idx].copy()

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """BLX-alpha crossover cho real-valued chromosomes."""
        if np.random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()

        alpha = 0.5
        child1_weights = np.zeros(self.n_weights)
        child2_weights = np.zeros(self.n_weights)

        for i in range(self.n_weights):
            p1, p2 = parent1.weights[i], parent2.weights[i]
            min_val, max_val = min(p1, p2), max(p1, p2)
            range_val = max_val - min_val

            low = min_val - alpha * range_val
            high = max_val + alpha * range_val

            child1_weights[i] = np.random.uniform(low, high)
            child2_weights[i] = np.random.uniform(low, high)

        # Clip to bounds
        child1_weights = np.clip(child1_weights, self.config.weight_min, self.config.weight_max)
        child2_weights = np.clip(child2_weights, self.config.weight_min, self.config.weight_max)

        return Individual(weights=child1_weights), Individual(weights=child2_weights)

    def mutate(self, individual: Individual) -> Individual:
        """Gaussian mutation."""
        mutated = individual.copy()

        for i in range(self.n_weights):
            if np.random.random() < self.config.mutation_rate:
                mutated.weights[i] += np.random.normal(0, self.config.mutation_sigma)

        # Clip to bounds
        mutated.weights = np.clip(mutated.weights, self.config.weight_min, self.config.weight_max)

        return mutated

    def evolve(self, verbose: bool = True) -> Individual:
        """
        Chạy GA optimization.

        Returns:
            Best individual sau tất cả generations.
        """
        print("=" * 60)
        print("GA + NGNet + FEMM Topology Optimization")
        print("=" * 60)
        print(f"Population size: {self.config.population_size}")
        print(f"Generations: {self.config.generations}")
        print(f"Crossover rate: {self.config.crossover_rate}")
        print(f"Mutation rate: {self.config.mutation_rate}")
        print(f"Elitism: {self.config.elitism}")
        print(f"Adv sweep: {self.config.adv_list}")
        print("=" * 60)

        # Initialize
        population = self.initialize_population()

        start_time = time.time()

        for gen in range(self.config.generations):
            gen_start = time.time()
            print(f"\n--- Generation {gen + 1}/{self.config.generations} ---")

            # Evaluate fitness cho tất cả individuals
            for i, ind in enumerate(population):
                if ind.fitness == 0.0:  # Chưa evaluate
                    print(f"  Individual {i + 1}/{len(population)}", end=" ")
                    self.evaluate_fitness(ind, verbose=False)
                    print(f"-> fitness={ind.fitness:.4f} N.m")

            # Sort by fitness (descending)
            population.sort(key=lambda x: x.fitness, reverse=True)

            # Track best
            if self.best_individual is None or population[0].fitness > self.best_individual.fitness:
                self.best_individual = population[0].copy()

            # Log
            fitnesses = [ind.fitness for ind in population]
            gen_stats = {
                "generation": gen + 1,
                "best_fitness": max(fitnesses),
                "avg_fitness": np.mean(fitnesses),
                "worst_fitness": min(fitnesses),
                "best_adv": population[0].best_adv,
            }
            self.history.append(gen_stats)

            gen_time = time.time() - gen_start
            print(f"  Best: {gen_stats['best_fitness']:.4f} N.m (adv={gen_stats['best_adv']}°)")
            print(f"  Avg:  {gen_stats['avg_fitness']:.4f} N.m")
            print(f"  Time: {gen_time:.1f}s")

            # Last generation - không cần tạo next gen
            if gen == self.config.generations - 1:
                break

            # Create next generation
            next_population = []

            # Elitism - giữ lại best individuals
            for i in range(self.config.elitism):
                next_population.append(population[i].copy())

            # Fill còn lại bằng selection + crossover + mutation
            while len(next_population) < self.config.population_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)

                child1, child2 = self.crossover(parent1, parent2)

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                next_population.append(child1)
                if len(next_population) < self.config.population_size:
                    next_population.append(child2)

            population = next_population

        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Total evaluations: {self.eval_count}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Best fitness: {self.best_individual.fitness:.4f} N.m")
        print(f"Best adv: {self.best_individual.best_adv}°")
        print(f"Best weights: {self.best_individual.weights}")

        return self.best_individual

    def save_results(self, filename: str = "ga_results.json"):
        """Lưu kết quả optimization."""
        results = {
            "config": {
                "population_size": self.config.population_size,
                "generations": self.config.generations,
                "crossover_rate": self.config.crossover_rate,
                "mutation_rate": self.config.mutation_rate,
                "elitism": self.config.elitism,
                "adv_list": self.config.adv_list,
            },
            "best": {
                "weights": self.best_individual.weights.tolist(),
                "fitness": self.best_individual.fitness,
                "best_adv": self.best_individual.best_adv,
            },
            "history": self.history,
            "total_evaluations": self.eval_count,
        }

        path = Path(self.output_dir) / filename
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {path}")


def main():
    """Main function để chạy GA optimization."""                
    # Paths                                                     #file paths
    tra_path = "rotor_2.TRA"        
    base_fem = "basic.FEM"
    output_dir = "."

    # Check files exist
    if not os.path.exists(tra_path):
        print(f"ERROR: TRA file not found: {tra_path}")
        sys.exit(1)
    if not os.path.exists(base_fem):
        print(f"ERROR: FEM file not found: {base_fem}")
        sys.exit(1)

    # GA config
    config = GAConfig(
        population_size=10,                                     # Nhỏ để test nhanh, tăng lên 20-50 cho production
        generations=5,                                          # Nhỏ để test, tăng lên 50-100 cho production
        crossover_rate=0.8,
        mutation_rate=0.15,
        mutation_sigma=0.2,
        tournament_size=3,
        elitism=2,
        adv_list=[35, 40, 45, 50, 55],                          #current sweeping angles
        Im=11.0,
        ini=-15.0,
    )

    # Create optimizer
    optimizer = GAOptimizer(
        config=config,
        tra_path=tra_path,
        base_fem=base_fem,
        output_dir=output_dir,
    )

    try:
        # Run optimization
        best = optimizer.evolve(verbose=True)

        # Save results
        optimizer.save_results("ga_results.json")

        # Generate final geometry với best weights
        print("\n--- Generating final geometry with best weights ---")
        dxf_path, json_path = generate_geometry(
            weights=best.weights,
            tra_path=tra_path,
            output_dir=output_dir,
            verbose=True
        )
        print(f"Final DXF: {dxf_path}")
        print(f"Final JSON: {json_path}")

    finally:
        # Đóng FEMM
        close_femm()

    print("\nDone!")


if __name__ == "__main__":
    main()
