# Import the OR-Tools CP-SAT model
from ortools.sat.python import cp_model
from optimisedVersion import RCPSP, abc, validate_solution
import random

def add_resource_constraints(model, num_tasks, durations, start_vars, project):
    # Iterate over each resource
    for resource_idx, availability in enumerate(project.resource_availabilities):
        # Define cumulative constraint data
        intervals = []
        demands = []

        for task in range(num_tasks):
            # Create an interval variable for the task
            start = start_vars[task]
            duration = durations[task]
            end = model.NewIntVar(0, 100, f"end_{task}")
            interval = model.NewIntervalVar(start, duration, end, f"interval_{task}")
            intervals.append(interval)

            # Add resource demand for this task and resource
            demands.append(project.resource_requirements[task][resource_idx])

        # Add cumulative constraint for this resource
        model.AddCumulative(intervals, demands, availability)

def compare_scheduling_algorithms(project, abc_params):
    # Compare ABC with Google OR-Tools
    num_tasks = project.num_tasks
    durations = project.durations
    predecessors = [task['predecessors'] for task in project.tasks.values()]

    # === OR-TOOLS MODEL ===
    model = cp_model.CpModel()

    # Variables for tasks
    start_vars = {}
    end_vars = {}

    # Create variables for start and end times
    for i in range(num_tasks):
        start_vars[i] = model.NewIntVar(0, 100, f"start_{i}")
        end_vars[i] = model.NewIntVar(0, 100, f"end_{i}")
        model.Add(end_vars[i] == start_vars[i] + durations[i])

    # Add precedence constraints
    for task, preds in enumerate(predecessors):
        for pred in preds:
            model.Add(end_vars[pred] <= start_vars[task])

    # Add resource constraints
    add_resource_constraints(model, num_tasks, durations, start_vars, project)

    # Define makespan variable
    makespan = model.NewIntVar(0, 100, "makespan")
    for i in range(num_tasks):
        model.Add(makespan >= end_vars[i])

    # Minimize makespan
    model.Minimize(makespan)

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    or_makespan = None
    or_schedule = None
    if status == cp_model.OPTIMAL:
        or_makespan = solver.Value(makespan)
        or_schedule = {i: solver.Value(start_vars[i]) for i in range(num_tasks)}

        # === VALIDATE OR-TOOLS SOLUTION ===
        print("Validating OR-Tools solution...")
        task_start_times = {task: or_schedule[task] for task in range(num_tasks)}
        is_valid = validate_solution(project, or_schedule, task_start_times)

        if not is_valid:
            print("OR-Tools solution is invalid. Skipping comparison with ABC.")
            return  # Exit early if OR-Tools solution is invalid
        else:
            print("OR-Tools solution is valid.")

    # === ABC ALGORITHM ===
    best_schedule, best_makespan, task_start_times = abc(
        abc_params['population_size'],
        abc_params['scouts'],
        abc_params['max_trial'],
        project,
        max_iterations=abc_params.get('max_iterations', 100)
    )

    # === COMPARE RESULTS ===
    comparison = {
        "OR-Tools": {
            "Makespan": or_makespan,
            "Schedule": or_schedule
        },
        "ABC": {
            "Makespan": best_makespan,
            "Schedule": task_start_times
        }
    }

    if or_makespan is not None:
        if or_makespan < best_makespan:
            print(f"OR-Tools performed better. ABC makespan: {best_makespan}   ORTools makespan: {or_makespan}\n")
        elif or_makespan > best_makespan:
            print("ABC algorithm performed better.\n")
        else:
            print(f"Both methods achieved the same makespan and that is: {best_makespan}\n")

# a wrapper to run tests and label each one
def run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors, test_name):
    print(f"\n{test_name}")
    #pack it into the RCPSP instance
    tasks = {
        i: {'duration': durations[i], 'predecessors': predecessors[i], 'resources': resource_requirements[i]} for i in
        range(num_tasks)
    }
    project = RCPSP(
        num_tasks=len(tasks),
        durations=[task['duration'] for task in tasks.values()],
        resource_requirements=[task['resources'] for task in tasks.values()],
        resource_availabilities=resource_availabilities
    )
    project.tasks = tasks
    #wrap the simulation parameters
    abc_params = {
        "population_size": 100,
        "scouts": 15,
        "max_trial": 20,
        "max_iterations": 100
    }
    compare_scheduling_algorithms(project, abc_params)


if __name__ == "__main__":
#     # TEST 1: Basic Dependencies
#     run_test(
#         num_tasks=4,
#         durations=[3, 2, 4, 1],
#         resource_requirements=[[2, 1], [1, 2], [3, 3], [1, 1]],
#         resource_availabilities=[4, 4],
#         predecessors=[[], [0], [0], [1, 2]],
#         test_name="TEST 1: Basic Dependencies"
#     )
#
#     # TEST 2: No Dependencies
#     run_test(
#         num_tasks=3,
#         durations=[2, 3, 4],
#         resource_requirements=[[1, 2], [2, 1], [3, 2]],
#         resource_availabilities=[4, 3],
#         predecessors=[[], [], []],
#         test_name="TEST 2: No Dependencies"
#     )
#
#     # TEST 3: Linear Dependency Chain
#     run_test(
#         num_tasks=4,
#         durations=[2, 3, 1, 4],
#         resource_requirements=[[1, 1], [2, 2], [1, 2], [3, 1]],
#         resource_availabilities=[4, 4],
#         predecessors=[[], [0], [1], [2]],
#         test_name="TEST 3: Linear Dependency Chain"
#     )
#
#     # TEST 4: Complex Dependencies
#     run_test(
#         num_tasks=5,
#         durations=[3, 2, 1, 4, 2],
#         resource_requirements=[[2, 1], [1, 2], [1, 1], [3, 2], [2, 3]],
#         resource_availabilities=[5, 5],
#         predecessors=[[], [0], [0], [1, 2], [2, 3]],
#         test_name="TEST 4: Complex Dependencies"
#       )
#     # TEST 5: High Resource Constraints
#     run_test(
#         num_tasks=4,
#         durations=[4, 3, 2, 5],
#         resource_requirements=[[4, 2], [3, 3], [2, 4], [5, 1]],
#         resource_availabilities=[5, 5],
#         predecessors = [[],[],[0],[1, 2]],
#         test_name="TEST 5: High Resource Constraints"
#       )
#     # TEST 6: Larger Project with Complex Dependencies
#     run_test(
#         num_tasks=6,
#         durations = [2, 3, 4, 1, 2, 3],
#         resource_requirements = [[2, 1], [3, 2], [1, 3], [2, 1], [3, 2], [2, 2]],
#         resource_availabilities = [6, 5],
#         predecessors = [
#         [],  # Task 0
#         [0],  # Task 1 depends on Task 0
#         [0],  # Task 2 depends on Task 0
#         [1, 2],  # Task 3 depends on Tasks 1 and 2
#         [3],  # Task 4 depends on Task 3
#         [2, 4],],
#         test_name="TEST 6: Larger Project with Complex Dependencies"
#     )
#     # TEST 7: All Tasks Have Same Duration and Resources
#     run_test(
#         num_tasks=4,
#         durations=[3, 3, 3, 3],
#         resource_requirements = [[2, 1], [2, 1], [2, 1], [2, 1]],
#         resource_availabilities = [4, 2],
#         predecessors = [
#             [],  # Task 0
#             [0],  # Task 1 depends on Task 0
#             [1],  # Task 2 depends on Task 1
#             [1],  # Task 3 depends on Task 1
#         ],
#         test_name="TEST 7: All Tasks Have Same Duration and Resources",
#     )
#     # TEST 8 Example for a circular dependency
#     try:
#         run_test(
#             num_tasks=3,
#             durations=[3, 2, 4],
#             resource_requirements=[[1, 2], [2, 1], [1, 1]],
#             resource_availabilities=[3, 3],
#             predecessors=[[2], [0], [1]],
#             test_name="TEST 8: Circular Precedence Edge Case (Invalid Input)"
#         )
#     except ValueError as e:
#         print(f"Error in TEST 8: {e}")
#
#     # Test 9: Large Project with Limited Resources
#     run_test(
#         num_tasks=6,
#         durations = [2, 3, 4, 1, 5, 3],
#         resource_requirements = [[2, 3], [3, 1], [2, 2], [1, 1], [3, 2], [2, 3]],
#         resource_availabilities = [5, 4],
#         predecessors = [
#             [],  # Task 0
#             [0],  # Task 1 depends on Task 0
#             [0],  # Task 2 depends on Task 0
#             [1],  # Task 3 depends on Task 1
#             [2, 3],  # Task 4 depends on Tasks 2 and 3
#             [4],  # Task 5 depends on Task 4
#         ],
#         test_name="Test 9: Large Project with Limited Resources",
#     )
#     # Test 10: All Tasks Share One Resource Exclusively
#     run_test(
#         num_tasks=5,
#         durations = [3, 2, 4, 1, 2],
#         resource_requirements = [[3], [3], [3], [3], [3]],
#         resource_availabilities = [3],
#         predecessors = [
#             [],  # Task 0
#             [0],  # Task 1 depends on Task 0
#             [0],  # Task 2 depends on Task 0
#             [1, 2],  # Task 3 depends on Tasks 1 and 2
#             [3],  # Task 4 depends on Task 3
#         ],
#         test_name="Test 10: All Tasks Share One Resource Exclusively",
#     )
#     # Test 11: High Resource Demand with Overlapping Tasks
#     run_test(
#         num_tasks=5,
#         durations = [4, 3, 5, 2, 3],
#         resource_requirements = [[3, 2], [2, 3], [4, 1], [1, 2], [2, 2]],
#         resource_availabilities = [5, 5],
#         predecessors = [
#             [],  # Task 0
#             [0],  # Task 1 depends on Task 0
#             [0, 1],  # Task 2 depends on Tasks 0 and 1
#             [1],  # Task 3 depends on Task 1
#             [2, 3],  # Task 4 depends on Tasks 2 and 3
#         ],
#         test_name="Test 11: High Resource Demand with Overlapping Tasks",
#     )
#     # Test 12: Tasks with Minimal Dependencies
#     run_test(
#         num_tasks=6,
#         durations = [2, 4, 3, 5, 2, 4],
#         resource_requirements = [[4, 1], [3, 2], [2, 3], [5, 1], [4, 2], [3, 3]],
#         resource_availabilities = [6, 5],
#         predecessors = [
#             [],  # Task 0
#             [],  # Task 1
#             [0],  # Task 2 depends on Task 0
#             [1, 2],  # Task 3 depends on Tasks 1 and 2
#             [2],  # Task 4 depends on Task 2
#             [3, 4],  # Task 5 depends on Tasks 3 and 4
#         ],
#         test_name="Test 12: Tasks with Minimal Dependencies",
#     )
#     # Test 13: Very Large Project
#     run_test(
#         num_tasks=10,
#         durations = [3, 2, 4, 1, 5, 3, 4, 2, 6, 3],
#         resource_requirements = [[2, 1], [1, 2], [3, 1], [2, 2], [4, 1], [2, 3], [1, 2], [3, 3], [2, 2], [4, 1]],
#         resource_availabilities = [6, 6],
#         predecessors = [
#             [],  # Task 0
#             [0],  # Task 1 depends on Task 0
#             [0],  # Task 2 depends on Task 0
#             [1, 2],  # Task 3 depends on Tasks 1 and 2
#             [2],  # Task 4 depends on Task 2
#             [3, 4],  # Task 5 depends on Tasks 3 and 4
#             [0],  # Task 6 depends on Task 0
#             [6],  # Task 7 depends on Task 6
#             [5, 7],  # Task 8 depends on Tasks 5 and 7
#             [8],  # Task 9 depends on Task 8
#         ],
#         test_name="Test 13: Very large project",
#     )
# #BATCH2
#     # Test 14: Simple Project with 3 tasks
#     run_test(
#         num_tasks=3,
#         durations = [2, 3, 1],
#         resource_requirements = [[1, 1], [2, 1], [1, 1]],
#         resource_availabilities = [3, 3],
#         predecessors = [[], [0], [1]],
#         test_name="Test 14: Simple Project with 3 tasks",
#     )
#     # Test 15: Simple Project with 3 tasks
#     run_test(
#         num_tasks=4,
#         durations = [3, 2, 2, 4],
#         resource_requirements = [[2, 1], [1, 2], [2, 2], [1, 1]],
#         resource_availabilities = [3, 3],
#         predecessors = [[], [0], [1], [2]],
#         test_name="Test 15: Simple project with resource conflict",
#     )
#     # Test 16: Complex project with more tasks and tighter resources
#     run_test(
#         num_tasks=5,
#         durations = [4, 3, 5, 2, 3],
#         resource_requirements = [[3, 2], [2, 3], [4, 1], [1, 2], [2, 2]],
#         resource_availabilities = [5, 5],
#         predecessors = [
#             [],  # Task 0
#             [0],  # Task 1 depends on Task 0
#             [0, 1],  # Task 2 depends on Tasks 0 and 1
#             [1],  # Task 3 depends on Task 1
#             [2, 3],  # Task 4 depends on Tasks 2 and 3
#         ],
#         test_name="Test 16: Complex project with more tasks and tighter resources",
#     )
#     # Test 17: Complex project with three sets of resources
#     run_test(
#         num_tasks=5,
#         durations = [4, 3, 5, 2, 3],
#         resource_requirements = [[3, 2, 6], [2, 3, 8], [4, 1, 2], [1, 2, 8], [2, 2, 3]],
#         resource_availabilities = [5, 5, 10],
#         predecessors = [
#             [2],  # Task 0
#             [0],  # Task 1 depends on Task 0
#             [0, 1],  # Task 2 depends on Tasks 0 and 1
#             [],  # Task 3 depends on Task 1
#             [2],  # Task 4 depends on Tasks 2 and 3
#         ],
#         test_name="Test 17: Complex project with three sets of resources",
#     )
#     # Test 18: Overlapping Tasks with Heavy Resource Demands
#     run_test(
#         num_tasks=5,
#         durations = [4, 3, 5, 2, 3],
#         resource_requirements = [[3, 2], [2, 3], [4, 1], [1, 2], [2, 2]],
#         resource_availabilities = [5, 5],
#         predecessors = [
#             [],  # Task 0
#             [0],  # Task 1 depends on Task 0
#             [0, 1],  # Task 2 depends on Tasks 0 and 1
#             [1],  # Task 3 depends on Task 1
#             [2, 3],  # Task 4 depends on Tasks 2 and 3
#         ],
#         test_name="Test 18: Overlapping Tasks with Heavy Resource Demands",
#     )
#     # Test 19: Tight Resource Constraints with Long Duration Tasks
#     run_test(
#         num_tasks=6,
#         durations = [6, 5, 7, 3, 4, 6],
#         resource_requirements = [[3, 2], [2, 3], [3, 4], [2, 1], [3, 2], [1, 3]],
#         resource_availabilities = [6, 6],
#         predecessors = [
#             [],  # Task 0
#             [0],  # Task 1 depends on Task 0
#             [1],  # Task 2 depends on Task 1
#             [2],  # Task 3 depends on Task 2
#             [3],  # Task 4 depends on Task 3
#             [4],  # Task 5 depends on Task 4
#         ],
#         test_name="Test 19: Tight Resource Constraints with Long Duration Tasks",
#     )
#     # Test 20: Complex Project with Multiple Predecessor Chains
#     run_test(
#         num_tasks=7,
#         durations = [4, 2, 3, 6, 2, 5, 7],
#         resource_requirements = [[2, 3], [3, 1], [2, 2], [1, 3], [4, 2], [1, 3], [3, 2]],
#         resource_availabilities = [8, 6],
#         predecessors = [
#             [],  # Task 0
#             [0],  # Task 1 depends on Task 0
#             [1],  # Task 2 depends on Task 1
#             [2],  # Task 3 depends on Task 2
#             [2],  # Task 4 depends on Task 2
#             [3],  # Task 5 depends on Task 3
#             [4, 5],  # Task 6 depends on Tasks 4 and 5
#         ],
#         test_name="Test 20: Complex Project with Multiple Predecessor Chains",
#     )
#     # Test 21: Tight Resource Constraints with Heavy Task Duration Overlap
#     run_test(
#         num_tasks=8,
#         durations = [7, 4, 5, 3, 2, 6, 5, 4],
#         resource_requirements = [[4, 3], [3, 4], [4, 2], [3, 3], [2, 2], [3, 4], [2, 3], [4, 3]],
#         resource_availabilities = [10, 7],
#         predecessors = [
#             [],  # Task 0
#             [0],  # Task 1 depends on Task 0
#             [0, 1],  # Task 2 depends on Tasks 0 and 1
#             [1],  # Task 3 depends on Task 1
#             [2],  # Task 4 depends on Task 2
#             [3],  # Task 5 depends on Task 3
#             [4],  # Task 6 depends on Task 4
#             [5],  # Task 7 depends on Task 5
#         ],
#         test_name="Test 21: Tight Resource Constraints with Heavy Task Duration Overlap",
#     )

#30 tasks, BF cant handle
    run_test(
        num_tasks=30,
        durations=[
            3, 5, 7, 4, 6, 8, 2, 5, 6, 3,
            4, 7, 6, 2, 5, 8, 3, 4, 7, 6,
            5, 6, 8, 4, 3, 7, 6, 2, 4, 5
        ],
        resource_requirements=[
            [3, 4], [2, 5], [3, 3], [4, 2], [3, 4], [2, 3], [3, 3], [4, 5], [3, 4], [2, 3],
            [4, 2], [3, 5], [2, 3], [3, 4], [4, 3], [2, 5], [3, 3], [4, 2], [3, 5], [4, 4],
            [2, 3], [3, 5], [4, 2], [3, 4], [2, 3], [4, 3], [3, 4], [4, 2], [2, 3], [4, 4]
        ],
        resource_availabilities=[20, 15],
        predecessors=[
            [], [0], [0, 1], [1], [2], [3], [4], [5], [6], [7],
            [8], [9], [10], [11], [12], [13], [14], [15], [16], [17],
            [18], [19], [20], [21], [22], [23], [24], [25], [26], [27]
        ],
        test_name="Test 1: Moderate Constraints with 30 Tasks"
    )

