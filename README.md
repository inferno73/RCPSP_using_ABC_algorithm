Resource-Constrained Project Scheduling Problem (RCPSP) Solver
This project implements a solution for the Resource-Constrained Project Scheduling Problem (RCPSP) using the Artificial Bee Colony (ABC) algorithm. The solver efficiently schedules tasks while adhering to resource constraints and task precedence, ensuring feasible and optimized solutions.

Key Features
Task Precedence Management: Ensures all tasks are scheduled after their prerequisites.
Resource Constraint Handling: Allocates resources effectively within defined limits.
Parallelism Optimization: Improves schedules by identifying opportunities for task parallelization.
ABC Algorithm Implementation: Simulates the behavior of employed, onlooker, and scout bees to explore and exploit the solution space.
Validation Checks: Ensures the final schedule satisfies all constraints.
Project Structure
RCPSP Class: Defines the problem structure, including tasks, durations, and resources.
ABC Algorithm: Implements the Artificial Bee Colony logic for finding optimized solutions.
Validation: Ensures precedence and resource constraints are respected in the generated schedule.
Visualization: Uses Gantt chart plotting to visualize schedules.
How It Works
Initialization: Tasks are topologically sorted, and initial task orders are shuffled to generate diverse solutions.
Optimization:
Employed bees explore new solutions near existing ones.
Onlooker bees focus on promising solutions.
Scout bees introduce diversity by generating new solutions.
Validation: Final schedules are checked to ensure feasibility and constraints satisfaction.
Output: The algorithm returns the best task order, makespan, and task start times.

All of the example tests are available in each script.
