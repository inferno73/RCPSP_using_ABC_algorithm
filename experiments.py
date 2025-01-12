from collections import Counter
from optimisedVersion import RCPSP, abc

# Initialize the results list
results = []

# Test-case 1 (test label t_40)
for _ in range(100):
    # # Description: A project with 40 tasks, non-linear dependencies, and mixed resource requirements.
    # print(f"Test_case 1, t_40:\n")
    # num_tasks = 40
    # durations = [3, 5, 2, 4, 6, 3, 2, 5, 1, 3, 7, 2, 3, 4, 5, 6, 3, 4, 2, 5, 1, 2, 6, 4, 5, 3, 2, 3, 7, 2, 5, 6, 3, 2,
    #              4, 5, 3, 1, 4, 2]
    # resource_requirements = [
    #     [1, 2], [2, 3], [1, 1], [3, 2], [2, 2], [1, 1], [0, 1], [3, 2], [2, 0], [1, 2],
    #     [2, 2], [3, 1], [1, 0], [3, 3], [2, 1], [1, 2], [2, 1], [3, 2], [1, 0], [2, 3],
    #     [0, 1], [1, 1], [3, 2], [2, 2], [1, 0], [3, 1], [1, 2], [2, 1], [1, 1], [2, 3],
    #     [3, 2], [2, 1], [1, 2], [3, 3], [2, 0], [1, 1], [0, 2], [3, 2], [2, 1], [1, 2]
    # ]
    # resource_availabilities = [10, 10]
    # predecessors = [
    #     [],  # Task 0
    #     [0],  # Task 1 depends on Task 0
    #     [0],  # Task 2 depends on Task 0
    #     [1],  # Task 3 depends on Task 1
    #     [2],  # Task 4 depends on Task 2
    #     [1, 2],  # Task 5 depends on Task 1 and Task 2
    #     [4],  # Task 6 depends on Task 4
    #     [3],  # Task 7 depends on Task 3
    #     [6],  # Task 8 depends on Task 6
    #     [7],  # Task 9 depends on Task 7
    #     [5],  # Task 10 depends on Task 5
    #     [8, 9],  # Task 11 depends on Task 8 and Task 9
    #     [10],  # Task 12 depends on Task 10
    #     [11],  # Task 13 depends on Task 11
    #     [12],  # Task 14 depends on Task 12
    #     [13],  # Task 15 depends on Task 13
    #     [14, 8],  # Task 16 depends on Task 14 and Task 8
    #     [15],  # Task 17 depends on Task 15
    #     [16],  # Task 18 depends on Task 16
    #     [17],  # Task 19 depends on Task 17
    #     [18, 11],  # Task 20 depends on Task 18 and Task 11
    #     [19],  # Task 21 depends on Task 19
    #     [20],  # Task 22 depends on Task 20
    #     [21, 10],  # Task 23 depends on Task 21 and Task 10
    #     [22],  # Task 24 depends on Task 22
    #     [23],  # Task 25 depends on Task 23
    #     [24, 12],  # Task 26 depends on Task 24 and Task 12
    #     [25],  # Task 27 depends on Task 25
    #     [26],  # Task 28 depends on Task 26
    #     [27],  # Task 29 depends on Task 27
    #     [28, 15],  # Task 30 depends on Task 28 and Task 15
    #     [29],  # Task 31 depends on Task 29
    #     [30],  # Task 32 depends on Task 30
    #     [31],  # Task 33 depends on Task 31
    #     [32, 20],  # Task 34 depends on Task 32 and Task 20
    #     [33],  # Task 35 depends on Task 33
    #     [34],  # Task 36 depends on Task 34
    #     [35],  # Task 37 depends on Task 35
    #     [36],  # Task 38 depends on Task 36
    #     [37, 22]  # Task 39 depends on Task 37 and Task 22
    # ]
    #
    # tasks = {
    #     i: {'duration': durations[i], 'predecessors': predecessors[i], 'resources': resource_requirements[i]} for i in
    #     range(num_tasks)
    # }
    #
    # project = RCPSP(
    #     num_tasks=len(tasks),
    #     durations=[task['duration'] for task in tasks.values()],
    #     resource_requirements=[task['resources'] for task in tasks.values()],
    #     resource_availabilities=resource_availabilities
    # )
    # project.tasks = tasks

    # print("Test 1\n")
    # # Description: A larger project with tight resource constraints and overlapping tasks.
    # num_tasks = 8
    # durations = [3, 2, 4, 3, 5, 6, 2, 4]
    # resource_requirements = [[2, 1], [3, 2], [4, 3], [1, 2], [3, 1], [2, 4], [1, 1], [3, 3]]
    # resource_availabilities = [6, 5]
    # predecessors = [
    #     [],  # Task 0
    #     [0],  # Task 1 depends on Task 0
    #     [0],  # Task 2 depends on Task 0
    #     [1],  # Task 3 depends on Task 1
    #     [1, 2],  # Task 4 depends on Tasks 1 and 2
    #     [3],  # Task 5 depends on Task 3
    #     [4],  # Task 6 depends on Task 4
    #     [5, 6]  # Task 7 depends on Tasks 5 and 6
    # ]
    # tasks = {
    #     0: {'duration': 3, 'predecessors': [], 'resources': [2, 1]},
    #     1: {'duration': 2, 'predecessors': [0], 'resources': [3, 2]},
    #     2: {'duration': 4, 'predecessors': [0], 'resources': [4, 3]},
    #     3: {'duration': 3, 'predecessors': [1], 'resources': [1, 2]},
    #     4: {'duration': 5, 'predecessors': [1, 2], 'resources': [3, 1]},
    #     5: {'duration': 6, 'predecessors': [3], 'resources': [2, 4]},
    #     6: {'duration': 2, 'predecessors': [4], 'resources': [1, 1]},
    #     7: {'duration': 4, 'predecessors': [5, 6], 'resources': [3, 3]}
    # }
    # project = RCPSP(
    #     num_tasks=len(tasks),
    #     durations=[task['duration'] for task in tasks.values()],
    #     resource_requirements=[task['resources'] for task in tasks.values()],
    #     resource_availabilities=resource_availabilities
    # )
    # project.tasks = tasks

    print("Test 9\n")
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
    tasks = {
        0: {'duration': 2, 'predecessors': [], 'resources': [2, 3]},
        1: {'duration': 3, 'predecessors': [0], 'resources': [3, 1]},
        2: {'duration': 4, 'predecessors': [0], 'resources': [2, 2]},
        3: {'duration': 1, 'predecessors': [1], 'resources': [1, 1]},
        4: {'duration': 5, 'predecessors': [2, 3], 'resources': [3, 2]},
        5: {'duration': 3, 'predecessors': [4], 'resources': [2, 3]}
    }
    project = RCPSP(
        num_tasks=len(tasks),
        durations=[task['duration'] for task in tasks.values()],
        resource_requirements=[task['resources'] for task in tasks.values()],
        resource_availabilities=resource_availabilities
    )
    project.tasks = tasks

    # Run the ABC algorithm
    population_size = 20
    scouts = 5
    max_trial = 10

    # Run the ABC algorithm
    best_schedule, best_makespan, task_start_times = abc(population_size, scouts, max_trial, project)
    results.append(best_makespan)

# Count the occurrences of each makespan value
counts = Counter(results)

# Calculate probabilities
total_runs = len(results)
print("Results of Testing:")
for makespan, count in sorted(counts.items()):
    probability = (count / total_runs) * 100
    print(f"Best_makespan: {makespan}, Probability: {probability:.2f}%")

