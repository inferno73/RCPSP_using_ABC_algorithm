import time
from itertools import permutations
from optimisedVersion import RCPSP, abc

#bruteForce RCPSP solver used to compare results with demonstrated ABC algorithm
#includes a lot of tests, batch1 and 2

# Function to convert inputs into `tasks` format
def convert_to_tasks(num_tasks, durations, resource_requirements, predecessors=None):
    if predecessors is None:
        predecessors = [[] for _ in range(num_tasks)]
    return {
        i: {
            "duration": durations[i],
            "predecessors": predecessors[i],
            "resources": resource_requirements[i],
        }
        for i in range(num_tasks)
    }

# Brute force implementation for makespan calculation
def brute_force_makespan(project):
    best_schedule = None
    best_makespan = float('inf')

    task_ids = list(project.tasks.keys())
    for task_order in permutations(task_ids):
        if project.validate_task_order(task_order):
            # Unpack or process the schedule correctly
            schedule = project.serial_schedule_generation_scheme(task_order)
            if isinstance(schedule, tuple):  # If it returns a tuple, adjust accordingly
                schedule = schedule[0]  # Assuming the first element is the schedule dictionary

            makespan = project.compute_makespan(schedule)
            if makespan < best_makespan:
                best_schedule = schedule
                best_makespan = makespan

    return best_schedule, best_makespan


# Function to run the test
def run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors=None):
    # Initialize project
    project = RCPSP(
        num_tasks=num_tasks,
        durations=durations,
        resource_requirements=resource_requirements,
        resource_availabilities=resource_availabilities
    )
    project.tasks = convert_to_tasks(num_tasks, durations, resource_requirements, predecessors)

    # Brute Force Testing
    start_time = time.time()
    bf_schedule, bf_makespan = brute_force_makespan(project)
    bf_time = time.time() - start_time
    #print(f"\nBrute Force Results:")
    #print(f"  Schedule: {bf_schedule}")
    print(f"Brute Force Makespan: {bf_makespan}")
    #print(f"  Time: {bf_time:.4f} seconds")

    # ABC Algorithm Testing
    start_time = time.time()
    abc_schedule, abc_makespan, startTimes = abc(population_size=20, scouts=5, max_trial=10, project=project)
    abc_time = time.time() - start_time
    #print(f"\nABC Algorithm Results:")
    #print(f"  Schedule: {abc_schedule}")
    print(f"ABC Algorithm Makespan: {abc_makespan}")
    #print(f"  Time: {abc_time:.4f} seconds")

    # Comparison
    if bf_makespan == abc_makespan:
        print("✅ ABC algorithm matches brute force result!\n")
    else:
        print("❌ ABC algorithm result differs from brute force!\n")

# Example test case

if __name__ == "__main__":
#BF vs ABC batch1
    print("TEST 1: Basic Dependencies")
    num_tasks = 4
    durations = [3, 2, 4, 1]
    resource_requirements = [[2, 1], [1, 2], [3, 3], [1, 1]]
    resource_availabilities = [4, 4]
    predecessors = [
        [],  # Task 0
        [0],  # Task 1 depends on Task 0
        [0],  # Task 2 depends on Task 0
        [1, 2],  # Task 3 depends on Tasks 1 and 2
    ]
    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

    print("\nTEST 2: No Dependencies")
    num_tasks = 3
    durations = [2, 3, 4]
    resource_requirements = [[1, 2], [2, 1], [3, 2]]
    resource_availabilities = [4, 3]
    predecessors = [[], [], []]  # No dependencies
    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

    print("\nTEST 3: Linear Dependency Chain")
    num_tasks = 4
    durations = [2, 3, 1, 4]
    resource_requirements = [[1, 1], [2, 2], [1, 2], [3, 1]]
    resource_availabilities = [4, 4]
    predecessors = [
        [],  # Task 0
        [0],  # Task 1 depends on Task 0
        [1],  # Task 2 depends on Task 1
        [2],  # Task 3 depends on Task 2
    ]
    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

    print("\nTEST 4: Complex Dependencies")
    num_tasks = 5
    durations = [3, 2, 1, 4, 2]
    resource_requirements = [[2, 1], [1, 2], [1, 1], [3, 2], [2, 3]]
    resource_availabilities = [5, 5]
    predecessors = [
        [],  # Task 0
        [0],  # Task 1 depends on Task 0
        [0],  # Task 2 depends on Task 0
        [1, 2],  # Task 3 depends on Tasks 1 and 2
        [2, 3],  # Task 4 depends on Tasks 2 and 3
    ]
    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

    print("\nTEST 5: High Resource Constraints")
    num_tasks = 4
    durations = [4, 3, 2, 5]
    resource_requirements = [[4, 2], [3, 3], [2, 4], [5, 1]]
    resource_availabilities = [5, 5]
    predecessors = [
        [],  # Task 0
        [],  # Task 1
        [0],  # Task 2 depends on Task 0
        [1, 2],  # Task 3 depends on Tasks 1 and 2
    ]
    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

    print("\nTEST 6: Larger Project with Complex Dependencies")
    num_tasks = 6
    durations = [2, 3, 4, 1, 2, 3]
    resource_requirements = [[2, 1], [3, 2], [1, 3], [2, 1], [3, 2], [2, 2]]
    resource_availabilities = [6, 5]
    predecessors = [
        [],  # Task 0
        [0],  # Task 1 depends on Task 0
        [0],  # Task 2 depends on Task 0
        [1, 2],  # Task 3 depends on Tasks 1 and 2
        [3],  # Task 4 depends on Task 3
        [2, 4],  # Task 5 depends on Tasks 2 and 4
    ]
    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

    print("\nTEST 7: All Tasks Have Same Duration and Resources")
    num_tasks = 4
    durations = [3, 3, 3, 3]
    resource_requirements = [[2, 1], [2, 1], [2, 1], [2, 1]]
    resource_availabilities = [4, 2]
    predecessors = [
        [],  # Task 0
        [0],  # Task 1 depends on Task 0
        [1],  # Task 2 depends on Task 1
        [1],  # Task 3 depends on Task 1
    ]
    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

    # Test 8: Circular Precedence Edge Case (Invalid Input)
    print("Test 8\n")
    # Description: This test checks if the system gracefully handles an invalid input with circular dependencies.
    num_tasks = 3
    durations = [3, 2, 4]
    resource_requirements = [[1, 2], [2, 1], [1, 1]]
    resource_availabilities = [3, 3]
    predecessors = [
        [2],  # Task 0 depends on Task 2
        [0],  # Task 1 depends on Task 0
        [1],  # Task 2 depends on Task 1 (circular dependency)
    ]

    try:
        run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)
    except ValueError as e:
        print(f"Error: {e}\n")

    # Test 9: Large Project with Limited Resources
    print("Test 9")
    # Description: A larger project with tight resource constraints.
    num_tasks = 6
    durations = [2, 3, 4, 1, 5, 3]
    resource_requirements = [[2, 3], [3, 1], [2, 2], [1, 1], [3, 2], [2, 3]]
    resource_availabilities = [5, 4]
    predecessors = [
        [],  # Task 0
        [0],  # Task 1 depends on Task 0
        [0],  # Task 2 depends on Task 0
        [1],  # Task 3 depends on Task 1
        [2, 3],  # Task 4 depends on Tasks 2 and 3
        [4],  # Task 5 depends on Task 4
    ]

    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

    # Test 10: All Tasks Share One Resource Exclusively
    print("Test 10")
    # Description: All tasks require the same resource, creating a bottleneck.
    num_tasks = 5
    durations = [3, 2, 4, 1, 2]
    resource_requirements = [[3], [3], [3], [3], [3]]
    resource_availabilities = [3]
    predecessors = [
        [],  # Task 0
        [0],  # Task 1 depends on Task 0
        [0],  # Task 2 depends on Task 0
        [1, 2],  # Task 3 depends on Tasks 1 and 2
        [3],  # Task 4 depends on Task 3
    ]

    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

    # Test 11: High Resource Demand with Overlapping Tasks
    print("Test 11")
    # Description: Tasks overlap heavily, testing resource constraint enforcement.
    num_tasks = 5
    durations = [4, 3, 5, 2, 3]
    resource_requirements = [[3, 2], [2, 3], [4, 1], [1, 2], [2, 2]]
    resource_availabilities = [5, 5]
    predecessors = [
        [],  # Task 0
        [0],  # Task 1 depends on Task 0
        [0, 1],  # Task 2 depends on Tasks 0 and 1
        [1],  # Task 3 depends on Task 1
        [2, 3],  # Task 4 depends on Tasks 2 and 3
    ]

    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

    # Test 12: Tasks with Minimal Dependencies
    print("Test 12")
    # Description: Tasks have few dependencies but high resource demands.
    num_tasks = 6
    durations = [2, 4, 3, 5, 2, 4]
    resource_requirements = [[4, 1], [3, 2], [2, 3], [5, 1], [4, 2], [3, 3]]
    resource_availabilities = [6, 5]
    predecessors = [
        [],  # Task 0
        [],  # Task 1
        [0],  # Task 2 depends on Task 0
        [1, 2],  # Task 3 depends on Tasks 1 and 2
        [2],  # Task 4 depends on Task 2
        [3, 4],  # Task 5 depends on Tasks 3 and 4
    ]

    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

    # Test 13: Very Large Project
    print("Test 13")
    # Description: A very large project with multiple dependencies and tight resource constraints.
    num_tasks = 10
    durations = [3, 2, 4, 1, 5, 3, 4, 2, 6, 3]
    resource_requirements = [[2, 1], [1, 2], [3, 1], [2, 2], [4, 1], [2, 3], [1, 2], [3, 3], [2, 2], [4, 1]]
    resource_availabilities = [6, 6]
    predecessors = [
        [],  # Task 0
        [0],  # Task 1 depends on Task 0
        [0],  # Task 2 depends on Task 0
        [1, 2],  # Task 3 depends on Tasks 1 and 2
        [2],  # Task 4 depends on Task 2
        [3, 4],  # Task 5 depends on Tasks 3 and 4
        [0],  # Task 6 depends on Task 0
        [6],  # Task 7 depends on Task 6
        [5, 7],  # Task 8 depends on Tasks 5 and 7
        [8],  # Task 9 depends on Task 8
    ]
    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

# #BF vs ABC batch2
    # Test 1: Simple Project with 3 tasks
    print("Test 14")
    # Description: Simple 3-task project to validate basic feasibility.
    num_tasks = 3
    durations = [2, 3, 1]
    resource_requirements = [[1, 1], [2, 1], [1, 1]]
    resource_availabilities = [3, 3]
    predecessors = [[], [0], [1]]  # Task 1 depends on Task 0, Task 2 depends on Task 1

    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

    # Test 2: Simple project with resource conflict
    print("Test 15")
    # Description: Tests resource conflict handling for tasks that require overlapping resources.
    num_tasks = 4
    durations = [3, 2, 2, 4]
    resource_requirements = [[2, 1], [1, 2], [2, 2], [1, 1]]
    resource_availabilities = [3, 3]
    predecessors = [[], [0], [1], [2]]  # Task 1 depends on Task 0, Task 2 depends on Task 1, Task 3 depends on Task 2

    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

    # Test 3: Complex project with more tasks and tighter resources
    print("Test 16")
    # Description: A project with more tasks and tight resource constraints for validation.
    num_tasks = 5
    durations = [4, 3, 5, 2, 3]
    resource_requirements = [[3, 2], [2, 3], [4, 1], [1, 2], [2, 2]]
    resource_availabilities = [5, 5]
    predecessors = [
        [],  # Task 0
        [0],  # Task 1 depends on Task 0
        [0, 1],  # Task 2 depends on Tasks 0 and 1
        [1],  # Task 3 depends on Task 1
        [2, 3],  # Task 4 depends on Tasks 2 and 3
    ]

    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)
# Test 4: Complex project with three sets of resources
    print("Test 17")
    # Description: A project with more tasks and tight resource constraints for validation.
    num_tasks = 5
    durations = [4, 3, 5, 2, 3]
    resource_requirements = [[3, 2, 6], [2, 3, 8], [4, 1, 2], [1, 2, 8], [2, 2, 3]]
    resource_availabilities = [5, 5, 10]
    predecessors = [
        [2],  # Task 0
        [0],  # Task 1 depends on Task 0
        [0, 1],  # Task 2 depends on Tasks 0 and 1
        [],  # Task 3 depends on Task 1
        [2],  # Task 4 depends on Tasks 2 and 3
    ]

    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

    # Test 5: Overlapping Tasks with Heavy Resource Demands
    print("Test 18")
    # Description: Tasks overlap heavily, testing resource constraint enforcement.
    num_tasks = 5
    durations = [4, 3, 5, 2, 3]
    resource_requirements = [[3, 2], [2, 3], [4, 1], [1, 2], [2, 2]]
    resource_availabilities = [5, 5]
    predecessors = [
        [],  # Task 0
        [0],  # Task 1 depends on Task 0
        [0, 1],  # Task 2 depends on Tasks 0 and 1
        [1],  # Task 3 depends on Task 1
        [2, 3],  # Task 4 depends on Tasks 2 and 3
    ]

    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

    # Test 6: Tight Resource Constraints with Long Duration Tasks
    print("Test 19")
    # Description: Project with high resource demand and tight resource constraints.
    num_tasks = 6
    durations = [6, 5, 7, 3, 4, 6]
    resource_requirements = [[3, 2], [2, 3], [3, 4], [2, 1], [3, 2], [1, 3]]
    resource_availabilities = [6, 6]
    predecessors = [
        [],  # Task 0
        [0],  # Task 1 depends on Task 0
        [1],  # Task 2 depends on Task 1
        [2],  # Task 3 depends on Task 2
        [3],  # Task 4 depends on Task 3
        [4],  # Task 5 depends on Task 4
    ]

    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

    # Test 7: Complex Project with Multiple Predecessor Chains
    print("Test 20")
    # Description: A complex project with long chains of dependencies to test precedence handling.
    num_tasks = 7
    durations = [4, 2, 3, 6, 2, 5, 7]
    resource_requirements = [[2, 3], [3, 1], [2, 2], [1, 3], [4, 2], [1, 3], [3, 2]]
    resource_availabilities = [8, 6]
    predecessors = [
        [],  # Task 0
        [0],  # Task 1 depends on Task 0
        [1],  # Task 2 depends on Task 1
        [2],  # Task 3 depends on Task 2
        [2],  # Task 4 depends on Task 2
        [3],  # Task 5 depends on Task 3
        [4, 5],  # Task 6 depends on Tasks 4 and 5
    ]

    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

    # Test 8: Tight Resource Constraints with Heavy Task Duration Overlap
    print("Test 21")
    # Description: A project with tight resource constraints and heavy task duration overlap to test resource usage.
    num_tasks = 8
    durations = [7, 4, 5, 3, 2, 6, 5, 4]
    resource_requirements = [[4, 3], [3, 4], [4, 2], [3, 3], [2, 2], [3, 4], [2, 3], [4, 3]]
    resource_availabilities = [10, 7]
    predecessors = [
        [],  # Task 0
        [0],  # Task 1 depends on Task 0
        [0, 1],  # Task 2 depends on Tasks 0 and 1
        [1],  # Task 3 depends on Task 1
        [2],  # Task 4 depends on Task 2
        [3],  # Task 5 depends on Task 3
        [4],  # Task 6 depends on Task 4
        [5],  # Task 7 depends on Task 5
    ]
    run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)

#Example of a test that cannot be conducted using brute force
    # print("Here TEST")
    # num_tasks=30
    # durations=[
    #     3, 5, 7, 4, 6, 8, 2, 5, 6, 3,
    #     4, 7, 6, 2, 5, 8, 3, 4, 7, 6,
    #     5, 6, 8, 4, 3, 7, 6, 2, 4, 5
    # ]
    # resource_requirements=[
    #     [3, 4], [2, 5], [3, 3], [4, 2], [3, 4], [2, 3], [3, 3], [4, 5], [3, 4], [2, 3],
    #     [4, 2], [3, 5], [2, 3], [3, 4], [4, 3], [2, 5], [3, 3], [4, 2], [3, 5], [4, 4],
    #     [2, 3], [3, 5], [4, 2], [3, 4], [2, 3], [4, 3], [3, 4], [4, 2], [2, 3], [4, 4]
    # ]
    # resource_availabilities=[20, 15]
    # predecessors=[
    #     [], [0], [0, 1], [1], [2], [3], [4], [5], [6], [7],
    #     [8], [9], [10], [11], [12], [13], [14], [15], [16], [17],
    #     [18], [19], [20], [21], [22], [23], [24], [25], [26], [27]
    # ]
    # run_test(num_tasks, durations, resource_requirements, resource_availabilities, predecessors)
