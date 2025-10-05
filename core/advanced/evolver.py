"""
J.A.R.V.I.S. Evolver
Genetic algorithm-based code evolution system
"""

import os
import time
import ast
import random
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import copy


@dataclass
class CodeIndividual:
    """Represents an individual in the genetic algorithm population"""
    code: str
    fitness: float = 0.0
    generation: int = 0
    mutations: List[str] = None

    def __post_init__(self):
        if self.mutations is None:
            self.mutations = []


class GeneticAlgorithm:
    """Genetic algorithm for code evolution"""

    def __init__(self, population_size: int = 20, generations: int = 10, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def evolve(self, initial_code: str, fitness_function) -> CodeIndividual:
        """Run genetic algorithm evolution"""
        # Initialize population
        population = self._initialize_population(initial_code)

        best_individual = None

        for generation in range(self.generations):
            # Evaluate fitness
            for individual in population:
                individual.fitness = fitness_function(individual.code)
                individual.generation = generation

            # Sort by fitness (higher is better)
            population.sort(key=lambda x: x.fitness, reverse=True)

            # Keep track of best
            if best_individual is None or population[0].fitness > best_individual.fitness:
                best_individual = population[0]

            # Create next generation
            new_population = population[:2]  # Keep top 2 (elitism)

            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                # Crossover
                child_code = self._crossover(parent1.code, parent2.code)

                # Mutation
                child_code = self._mutate(child_code)

                # Create child individual
                child = CodeIndividual(
                    code=child_code,
                    generation=generation + 1,
                    mutations=[]
                )

                new_population.append(child)

            population = new_population

        return best_individual

    def _initialize_population(self, initial_code: str) -> List[CodeIndividual]:
        """Initialize population with mutated versions of initial code"""
        population = [CodeIndividual(code=initial_code)]

        for _ in range(self.population_size - 1):
            mutated_code = self._mutate(initial_code)
            population.append(CodeIndividual(code=mutated_code))

        return population

    def _tournament_selection(self, population: List[CodeIndividual], tournament_size: int = 3) -> CodeIndividual:
        """Tournament selection"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(self, code1: str, code2: str) -> str:
        """Crossover between two code strings"""
        try:
            # Parse AST
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)

            # Simple crossover: swap function bodies
            functions1 = [node for node in ast.walk(tree1) if isinstance(node, ast.FunctionDef)]
            functions2 = [node for node in ast.walk(tree2) if isinstance(node, ast.FunctionDef)]

            if functions1 and functions2:
                # Swap bodies of first functions
                func1_body = functions1[0].body
                func2_body = functions2[0].body

                # Create new trees
                new_tree1 = copy.deepcopy(tree1)
                new_tree2 = copy.deepcopy(tree2)

                # Find and replace bodies
                for node in ast.walk(new_tree1):
                    if isinstance(node, ast.FunctionDef) and node.name == functions1[0].name:
                        node.body = func2_body
                        break

                # Convert back to code
                return ast.unparse(new_tree1)

        except Exception:
            # Fallback: random split and combine
            lines1 = code1.split('\n')
            lines2 = code2.split('\n')

            split_point = random.randint(1, min(len(lines1), len(lines2)) - 1)

            new_lines = lines1[:split_point] + lines2[split_point:]
            return '\n'.join(new_lines)

        return code1  # Return original if crossover fails

    def _mutate(self, code: str) -> str:
        """Mutate code with various transformations"""
        if random.random() > self.mutation_rate:
            return code

        try:
            tree = ast.parse(code)
            mutated_tree = copy.deepcopy(tree)

            # Apply random mutations
            mutations = [
                self._mutate_variable_names,
                self._mutate_function_names,
                self._mutate_constants,
                self._add_error_handling,
                self._optimize_loops,
                self._add_logging
            ]

            # Apply one random mutation
            mutation = random.choice(mutations)
            mutated_tree = mutation(mutated_tree)

            return ast.unparse(mutated_tree)

        except Exception:
            return code  # Return original if mutation fails

    def _mutate_variable_names(self, tree: ast.AST) -> ast.AST:
        """Mutate variable names"""
        class VariableRenamer(ast.NodeTransformer):
            def __init__(self):
                self.var_map = {}

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    if node.id not in self.var_map:
                        self.var_map[node.id] = f"var_{len(self.var_map)}"
                    node.id = self.var_map[node.id]
                elif isinstance(node.ctx, ast.Load):
                    if node.id in self.var_map:
                        node.id = self.var_map[node.id]
                return node

        renamer = VariableRenamer()
        return renamer.visit(tree)

    def _mutate_function_names(self, tree: ast.AST) -> ast.AST:
        """Mutate function names"""
        class FunctionRenamer(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                node.name = f"func_{hash(node.name) % 1000}"
                return node

        renamer = FunctionRenamer()
        return renamer.visit(tree)

    def _mutate_constants(self, tree: ast.AST) -> ast.AST:
        """Mutate numeric constants"""
        class ConstantMutator(ast.NodeTransformer):
            def visit_Num(self, node):
                if isinstance(node.n, int):
                    node.n = int(node.n * random.uniform(0.5, 1.5))
                elif isinstance(node.n, float):
                    node.n = node.n * random.uniform(0.8, 1.2)
                return node

        mutator = ConstantMutator()
        return mutator.visit(tree)

    def _add_error_handling(self, tree: ast.AST) -> ast.AST:
        """Add try-except blocks"""
        class ErrorHandler(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # Wrap function body in try-except
                try_body = node.body
                except_body = [
                    ast.Return(value=ast.Name(id='None', ctx=ast.Load()))
                ]

                try_except = ast.Try(
                    body=try_body,
                    handlers=[ast.ExceptHandler(
                        type=ast.Name(id='Exception', ctx=ast.Load()),
                        name='e',
                        body=except_body
                    )],
                    orelse=[],
                    finalbody=[]
                )

                node.body = [try_except]
                return node

        handler = ErrorHandler()
        return handler.visit(tree)

    def _optimize_loops(self, tree: ast.AST) -> ast.AST:
        """Optimize loops (simplified)"""
        class LoopOptimizer(ast.NodeTransformer):
            def visit_For(self, node):
                # Add enumerate if using range(len())
                if (isinstance(node.iter, ast.Call) and
                    isinstance(node.iter.func, ast.Name) and
                    node.iter.func.id == 'range' and
                    len(node.iter.args) == 1 and
                    isinstance(node.iter.args[0], ast.Call) and
                    isinstance(node.iter.args[0].func, ast.Name) and
                    node.iter.args[0].func.id == 'len'):

                    # Replace with enumerate
                    node.iter = ast.Call(
                        func=ast.Name(id='enumerate', ctx=ast.Load()),
                        args=[node.iter.args[0].args[0]],
                        keywords=[]
                    )

                    # Update loop variable to tuple
                    if isinstance(node.target, ast.Name):
                        node.target = ast.Tuple(
                            elts=[
                                ast.Name(id=node.target.id, ctx=ast.Store()),
                                ast.Name(id=f"{node.target.id}_idx", ctx=ast.Store())
                            ],
                            ctx=ast.Store()
                        )

                return node

        optimizer = LoopOptimizer()
        return optimizer.visit(tree)

    def _add_logging(self, tree: ast.AST) -> ast.AST:
        """Add logging statements"""
        class LoggingAdder(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # Add logging at start of function
                log_stmt = ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='logger', ctx=ast.Load()),
                            attr='info',
                            ctx=ast.Load()
                        ),
                        args=[ast.Str(s=f"Calling {node.name}")],
                        keywords=[]
                    )
                )

                node.body.insert(0, log_stmt)
                return node

        adder = LoggingAdder()
        return adder.visit(tree)


class Evolver:
    """Evolves code using genetic algorithms"""

    def __init__(self, development_engine):
        self.engine = development_engine
        self.jarvis = development_engine.jarvis
        self.logger = logging.getLogger('JARVIS.Evolver')

        # Genetic algorithm parameters
        self.ga = GeneticAlgorithm(
            population_size=20,
            generations=10,
            mutation_rate=0.1
        )

        # Evolution history
        self.evolution_history = []

    async def initialize(self):
        """Initialize evolver"""
        try:
            self.logger.info("Initializing genetic algorithm evolver...")

            # Test genetic algorithm
            test_code = "def test(): return 1"
            result = self.ga.evolve(test_code, lambda x: 1.0)

            if result:
                self.logger.info("✓ Genetic algorithm initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing evolver: {e}")
            raise

    async def evolve_code(self, code: str, task_type: str, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve code for better performance using genetic algorithms"""
        try:
            self.logger.info(f"Starting genetic evolution for {task_type}")

            start_time = time.time()

            # Define fitness function based on task type and metrics
            fitness_function = self._create_fitness_function(task_type, performance_metrics)

            # Run genetic algorithm
            best_individual = self.ga.evolve(code, fitness_function)

            evolution_time = time.time() - start_time

            # Check if improvement was made
            original_fitness = fitness_function(code)
            improvement_made = best_individual.fitness > original_fitness

            # Record evolution
            evolution_record = {
                "original_code": code,
                "evolved_code": best_individual.code,
                "original_fitness": original_fitness,
                "evolved_fitness": best_individual.fitness,
                "generations": best_individual.generation,
                "evolution_time": evolution_time,
                "task_type": task_type,
                "timestamp": time.time()
            }

            self.evolution_history.append(evolution_record)

            self.logger.info(f"Evolution completed: fitness {original_fitness:.3f} -> {best_individual.fitness:.3f}")

            return {
                "improvements_made": improvement_made,
                "improved_code": best_individual.code,
                "evolution_cycles": best_individual.generation,
                "fitness_improvement": best_individual.fitness - original_fitness,
                "evolution_time": evolution_time
            }

        except Exception as e:
            self.logger.error(f"Error in genetic evolution: {e}")
            return {
                "improvements_made": False,
                "improved_code": code,
                "error": str(e)
            }

    def _create_fitness_function(self, task_type: str, performance_metrics: Dict[str, Any]):
        """Create fitness function based on task type and metrics"""

        def fitness_function(code: str) -> float:
            try:
                fitness = 0.0

                # Parse code to check structure
                try:
                    tree = ast.parse(code)
                    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                except:
                    return 0.0  # Invalid code

                # Base fitness from code structure
                fitness += len(functions) * 2  # More functions = potentially better
                fitness += len(classes) * 5    # Classes are good

                # Check for error handling
                has_try_except = 'try:' in code and 'except' in code
                if has_try_except:
                    fitness += 10

                # Check for logging
                has_logging = 'logger' in code.lower()
                if has_logging:
                    fitness += 5

                # Check for type hints
                has_type_hints = '->' in code or ': ' in code
                if has_type_hints:
                    fitness += 5

                # Task-specific fitness
                if task_type == "performance":
                    # Favor code with optimizations
                    if 'enumerate(' in code:
                        fitness += 8
                    if 'list(' not in code and '[]' in code:  # Avoid unnecessary list() calls
                        fitness += 3

                elif task_type == "reliability":
                    # Favor code with error handling
                    error_handling_score = code.count('try:') + code.count('except') + code.count('finally')
                    fitness += error_handling_score * 2

                elif task_type == "maintainability":
                    # Favor readable code
                    lines = code.split('\n')
                    avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
                    if avg_line_length < 100:  # Reasonable line length
                        fitness += 10
                    if '"""' in code:  # Has docstrings
                        fitness += 5

                # Performance metrics bonus
                if performance_metrics:
                    execution_time = performance_metrics.get('execution_time', 0)
                    if execution_time > 0:
                        fitness += max(0, 20 - execution_time * 10)  # Faster = better

                    memory_usage = performance_metrics.get('memory_usage', 0)
                    fitness += max(0, 10 - memory_usage / 1024 / 1024)  # Lower memory = better

                return fitness

            except Exception as e:
                self.logger.debug(f"Fitness evaluation error: {e}")
                return 0.0

        return fitness_function

    async def evolve_algorithm(self, algorithm_code: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evolve an algorithm using test cases"""

        def algorithm_fitness(code: str) -> float:
            try:
                # Execute code against test cases
                fitness = 0.0

                for test_case in test_cases:
                    try:
                        # This would need a safe execution environment
                        # For now, just check code structure
                        if 'def ' in code and 'return' in code:
                            fitness += 10

                        # Check if algorithm handles test case requirements
                        input_keys = test_case.get('input', {}).keys()
                        for key in input_keys:
                            if key in code:
                                fitness += 2

                    except Exception:
                        continue

                return fitness

            except Exception:
                return 0.0

        return await self.evolve_code(algorithm_code, "algorithm", {})

    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get evolution statistics"""
        if not self.evolution_history:
            return {"total_evolutions": 0}

        total_improvements = sum(1 for record in self.evolution_history
                               if record['evolved_fitness'] > record['original_fitness'])

        avg_improvement = sum(record['evolved_fitness'] - record['original_fitness']
                            for record in self.evolution_history) / len(self.evolution_history)

        return {
            "total_evolutions": len(self.evolution_history),
            "successful_evolutions": total_improvements,
            "success_rate": total_improvements / len(self.evolution_history),
            "average_improvement": avg_improvement,
            "best_improvement": max((record['evolved_fitness'] - record['original_fitness']
                                   for record in self.evolution_history), default=0)
        }

    async def shutdown(self):
        """Shutdown evolver"""
        try:
            self.logger.info("Shutting down genetic evolver...")

            # Save evolution history
            await self._save_evolution_history()

            self.logger.info("Evolver shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down evolver: {e}")

    async def _save_evolution_history(self):
        """Save evolution history to file"""
        try:
            history_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'evolution_history.json')

            os.makedirs(os.path.dirname(history_file), exist_ok=True)

            with open(history_file, 'w') as f:
                json.dump({
                    "history": self.evolution_history,
                    "stats": self.get_evolution_stats(),
                    "last_updated": time.time()
                }, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving evolution history: {e}")