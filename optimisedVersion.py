import random
from PlotGant import plot_schedule

class RCPSP:
    def __init__(self, num_tasks, durations, resource_requirements, resource_availabilities):
        self.num_tasks = num_tasks
        self.durations = durations
        self.resource_requirements = resource_requirements
        self.resource_availabilities = resource_availabilities
        self.tasks = {}

    def get_successors(self, task_id):
        """Returns a list of successor task IDs for the given task."""
        successors = []
        for t_id, task in self.tasks.items():
            if task_id in task['predecessors']:
                successors.append(t_id)
        return successors

    def detect_circular_dependencies(self):
        """Detects circular dependencies in the task graph using DFS."""

        def dfs(task_id, visited, stack):
            visited.add(task_id)
            stack.add(task_id)

            for pred in self.tasks[task_id]['predecessors']:
                if pred not in visited:
                    if dfs(pred, visited, stack):
                        return True
                elif pred in stack:
                    return True

            stack.remove(task_id)
            return False

        visited = set()
        stack = set()
        for task_id in self.tasks:
            if task_id not in visited:
                if dfs(task_id, visited, stack):
                    return True
        return False

    def topological_sort(self):
        """Generates a topological sort of tasks using Kahn's algorithm."""
        in_degree = {task_id: 0 for task_id in self.tasks}
        for task_id, task in self.tasks.items():
            for pred in task['predecessors']:
                in_degree[task_id] += 1

        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        topo_order = []

        while queue:
            current = queue.pop(0)
            topo_order.append(current)

            # For each task, check for tasks that depend on it (i.e., its successors)
            for successor_id, successor_task in self.tasks.items():
                if current in successor_task['predecessors']:
                    in_degree[successor_id] -= 1
                    if in_degree[successor_id] == 0:
                        queue.append(successor_id)

        if len(topo_order) != len(self.tasks):
            raise ValueError("Graph contains a cycle; topological sort not possible.")

        return topo_order

    def serial_schedule_generation_scheme(self, task_order=None):
        import heapq

        schedule = {}
        resource_usage = [0] * (max(self.durations) * self.num_tasks)  # Tracks resource usage over time

        if task_order is None:
            task_order = list(self.tasks.keys())

        task_start_times = {}
        ready_queue = []  # Tasks ready to be scheduled

        # Determine the number of unsatisfied predecessors for each task
        unsatisfied_predecessors = {
            task_id: len(self.tasks[task_id]['predecessors']) for task_id in self.tasks
        }

        # Initialize the ready queue with tasks that have no predecessors
        for task_id, count in unsatisfied_predecessors.items():
            if count == 0:
                heapq.heappush(ready_queue, task_id)

        current_time = 0

        while ready_queue:
            # Process ready tasks
            task_id = heapq.heappop(ready_queue)
            task = self.tasks[task_id]

            # Compute the earliest possible start time based on predecessor completion
            start_time = max(
                (task_start_times[pred] + self.tasks[pred]['duration'] for pred in task['predecessors']),
                default=current_time
            )

            # Ensure resource constraints are met
            while True:
                can_start = True
                for t in range(start_time, start_time + task['duration']):
                    if t >= len(resource_usage) or any(
                            resource_usage[t] + req > avail
                            for req, avail in zip(task['resources'], self.resource_availabilities)
                    ):
                        can_start = False
                        break
                if can_start:
                    break
                start_time += 1

            # Assign the computed start time
            task_start_times[task_id] = start_time

            # Update resource usage during the task's execution
            for t in range(start_time, start_time + task['duration']):
                for r, req in enumerate(task['resources']):
                    resource_usage[t] += req

            schedule[task_id] = start_time

            # Update unsatisfied predecessors for dependent tasks
            for successor_id in self.get_successors(task_id):
                unsatisfied_predecessors[successor_id] -= 1
                if unsatisfied_predecessors[successor_id] == 0:
                    heapq.heappush(ready_queue, successor_id)

            # Increment current time
            current_time = max(current_time, start_time + task['duration'])

        return schedule, task_start_times

    def compute_makespan(self, task_start_times):
        return max(start + self.tasks[task_id]['duration'] for task_id, start in task_start_times.items())

    def validate_task_order(self, task_order):
        visited = set()
        for task_id in task_order:
            for pred in self.tasks[task_id]['predecessors']:
                if pred not in visited:
                    return False
            visited.add(task_id)
        return True

def optimize_parallelism(schedule, task_start_times, project):
    """
    Adjusts the start times of tasks in the schedule to allow better parallelism,
    ensuring resource and precedence constraints are respected.
    """
    # Convert schedule to a structure that includes task IDs and their start times
    sorted_tasks = [(task_id, task_start_times[task_id]) for task_id in schedule]
    sorted_tasks.sort(key=lambda x: x[1])  # Sort by start time

    new_task_start_times = task_start_times.copy()

    for task_id, start_time in sorted_tasks:
        earliest_start = 0

        # Check predecessor constraints
        for pred in project.tasks[task_id]['predecessors']:
            pred_end_time = new_task_start_times[pred] + project.tasks[pred]['duration']
            earliest_start = max(earliest_start, pred_end_time)

        # Check resource constraints
        task_duration = project.tasks[task_id]['duration']
        task_resources = project.tasks[task_id]['resources']

        for t in range(earliest_start, start_time):
            if all(
                project.resource_availabilities[r] >= sum(
                    project.tasks[other_task]['resources'][r]
                    for other_task in schedule
                    if new_task_start_times[other_task] <= t < new_task_start_times[other_task] + project.tasks[other_task]['duration']
                ) + task_resources[r]
                for r in range(len(task_resources))
            ):
                # Update the task start time if all conditions are met
                new_task_start_times[task_id] = t
                break

    # Reconstruct the new schedule
    new_schedule = [task_id for task_id in schedule]

    return new_schedule, new_task_start_times

def constrained_shuffle(task_order, project):
    valid_order = task_order[:]
    for _ in range(len(task_order) * 2):  # Attempt constrained swaps
        i, j = random.sample(range(len(valid_order)), 2)
        if project.validate_task_order(valid_order[:i] + [valid_order[j]] + valid_order[i+1:j] + [valid_order[i]] + valid_order[j+1:]):
            valid_order[i], valid_order[j] = valid_order[j], valid_order[i]
    return valid_order

def abc(population_size, scouts, max_trial, project, max_iterations=100, max_shuffle_attempts=10):
    if project.detect_circular_dependencies():
        return None, float('inf'), None
    #initialization
    food_number = population_size // 2
    food_sources = []
    base_task_order = project.topological_sort()

    for _ in range(food_number):
        task_order = base_task_order[:]
        shuffle_attempts = 0

        while not project.validate_task_order(task_order):
            task_order = constrained_shuffle(task_order, project)
            shuffle_attempts += 1
            if shuffle_attempts >= max_shuffle_attempts:
                break

        food_sources.append({
            "task_order": task_order,
            "makespan": None,
            "task_start_times": None
        })

    trials = [0] * food_number
    best_schedule = None
    best_makespan = float('inf')

    def evaluate_schedule(source):
        if source["makespan"] is None:
            schedule, task_start_times = project.serial_schedule_generation_scheme(source["task_order"])
            source["makespan"] = project.compute_makespan(schedule)
            source["task_start_times"] = task_start_times
        return source["makespan"], source["task_start_times"]

    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        # send employed bees
        for source in food_sources:
            makespan, task_start_times = evaluate_schedule(source)
            if makespan < best_makespan:
                best_schedule = source["task_order"][:]
                best_makespan = makespan
        # send onlooker bees
        for i in range(food_number):
            new_task_order = food_sources[i]["task_order"][:]
            a, b = random.sample(range(project.num_tasks), 2)
            new_task_order[a], new_task_order[b] = new_task_order[b], new_task_order[a]

            if project.validate_task_order(new_task_order):
                new_source = {
                    "task_order": new_task_order,
                    "makespan": None,
                    "task_start_times": None
                }
                new_makespan, _ = evaluate_schedule(new_source)

                if new_makespan < food_sources[i]["makespan"]:
                    food_sources[i] = new_source
                    trials[i] = 0

                    if new_makespan < best_makespan:
                        best_schedule = new_task_order
                        best_makespan = new_makespan
                else:
                    trials[i] += 1
        # send scout bees
        scout_indices = sorted(range(food_number), key=lambda x: trials[x], reverse=True)[:scouts]
        for idx in scout_indices:
            task_order = base_task_order[:]
            random.shuffle(task_order)
            shuffle_attempts = 0

            while not project.validate_task_order(task_order):
                random.shuffle(task_order)
                shuffle_attempts += 1
                if shuffle_attempts >= max_shuffle_attempts:
                    break

            food_sources[idx] = {
                "task_order": task_order,
                "makespan": None,
                "task_start_times": None
            }
            trials[idx] = 0

        if all(trial <= max_trial for trial in trials):
            break

    # check if further parallelism is possible
    # Recalculate makespan after optimizing parallelism todo vrati na staro
    best_schedule, task_start_times = optimize_parallelism(best_schedule, task_start_times, project)
    best_makespan = project.compute_makespan(task_start_times)  # Use task_start_times here

    return best_schedule, best_makespan, task_start_times


#simple helper function used for testing, checks if predecessors or resource constraints are validated in the final given solution
def validate_solution(project, best_schedule, task_start_times):
    # Check precedence constraints
    for task, task_data in project.tasks.items():
        for predecessor in task_data['predecessors']:
            if task_start_times[task] < task_start_times[predecessor] + project.durations[predecessor]:
                print(f"Precedence constraint violated: Task {task} starts before Task {predecessor} is completed.")
                return False

    # Check resource constraints
    num_resources = len(project.resource_availabilities)
    max_time = max(task_start_times[task] + project.durations[task] for task in project.tasks)
    resource_usage = [[0] * max_time for _ in range(num_resources)]

    for task, start_time in task_start_times.items():
        for resource_idx, resource_required in enumerate(project.resource_requirements[task]):
            for t in range(start_time, start_time + project.durations[task]):
                resource_usage[resource_idx][t] += resource_required
                if resource_usage[resource_idx][t] > project.resource_availabilities[resource_idx]:
                    print(f"Resource constraint violated: Resource {resource_idx} exceeds availability at time {t}.")
                    return False

    print("The solution is valid: all precedence and resource constraints are fulfilled.")
    return True

if __name__ == "__main__":
# # batch3 - COMPLEX, EDGE CASES ETC with more activities
#     print("Test 9\n")
#     # Description: A project where one task has zero duration.
#     num_tasks = 4
#     durations = [3, 0, 5, 2]
#     resource_requirements = [[1, 2], [0, 0], [2, 1], [1, 1]]
#     resource_availabilities = [5, 5]
#     predecessors = [
#         [],  # Task 0
#         [0],  # Task 1 depends on Task 0
#         [1],  # Task 2 depends on Task 1
#         [1],  # Task 3 depends on Task 1
#     ]
#     tasks = {
#         0: {'duration': 3, 'predecessors': [], 'resources': [1, 2]},
#         1: {'duration': 0, 'predecessors': [0], 'resources': [0, 0]},
#         2: {'duration': 5, 'predecessors': [1], 'resources': [2, 1]},
#         3: {'duration': 2, 'predecessors': [1], 'resources': [1, 1]},
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, start_times_of_activities = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
#
#     print("Test 10\n")
#     # Description: A project where one task requires zero resources.
#     num_tasks = 3
#     durations = [4, 3, 5]
#     resource_requirements = [[1, 1], [0, 0], [2, 3]]
#     resource_availabilities = [5, 5]
#     predecessors = [
#         [],  # Task 0
#         [0],  # Task 1 depends on Task 0
#         [1],  # Task 2 depends on Task 1
#     ]
#     tasks = {
#         0: {'duration': 4, 'predecessors': [], 'resources': [1, 1]},
#         1: {'duration': 3, 'predecessors': [0], 'resources': [0, 0]},
#         2: {'duration': 5, 'predecessors': [1], 'resources': [2, 3]},
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, start_times_of_activities = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
#
#     print("Test 11\n")
#     # Description: A project where resource availability exactly matches demand.
#     num_tasks = 4
#     durations = [3, 2, 4, 5]
#     resource_requirements = [[1, 2], [2, 3], [3, 2], [4, 1]]
#     resource_availabilities = [10, 8]  # Total demand is exactly matched.
#     predecessors = [
#         [],  # Task 0
#         [0],  # Task 1 depends on Task 0
#         [1],  # Task 2 depends on Task 1
#         [1],  # Task 3 depends on Task 1
#     ]
#     tasks = {
#         0: {'duration': 3, 'predecessors': [], 'resources': [1, 2]},
#         1: {'duration': 2, 'predecessors': [0], 'resources': [2, 3]},
#         2: {'duration': 4, 'predecessors': [1], 'resources': [3, 2]},
#         3: {'duration': 5, 'predecessors': [1], 'resources': [4, 1]},
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, start_times_of_activities = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
#
#     print("Test 1: 10 Activities\n")
#     # Description: A project with 10 activities and balanced resource availability.
#
#     num_tasks = 10
#     durations = [2, 3, 1, 4, 5, 2, 3, 4, 2, 1]
#     resource_requirements = [
#         [2, 1], [1, 2], [1, 1], [2, 3], [1, 1],
#         [2, 2], [1, 3], [2, 1], [3, 2], [1, 1]
#     ]
#     resource_availabilities = [6, 6]
#     predecessors = [
#         [],  # Task 0
#         [0],  # Task 1 depends on Task 0
#         [0],  # Task 2 depends on Task 0
#         [1, 2],  # Task 3 depends on Tasks 1 and 2
#         [3],  # Task 4 depends on Task 3
#         [2],  # Task 5 depends on Task 2
#         [4, 5],  # Task 6 depends on Tasks 4 and 5
#         [5],  # Task 7 depends on Task 5
#         [6],  # Task 8 depends on Task 6
#         [8, 7],  # Task 9 depends on Tasks 8 and 7
#     ]
#     tasks = {
#         i: {'duration': durations[i], 'predecessors': predecessors[i], 'resources': resource_requirements[i]}
#         for i in range(num_tasks)
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, start_times_of_activities = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
#
#     print("Test 2: 14 Activities\n")
#     # Description: A project with 14 activities, moderate dependencies, and manageable resources.
#
#     num_tasks = 14
#     durations = [3, 2, 4, 1, 2, 5, 3, 2, 4, 3, 1, 2, 3, 4]
#     resource_requirements = [
#         [1, 1], [2, 3], [1, 2], [3, 1], [2, 2],
#         [1, 3], [2, 1], [1, 1], [3, 2], [2, 3],
#         [1, 2], [2, 1], [1, 1], [3, 2]
#     ]
#     resource_availabilities = [7, 7]
#     predecessors = [
#         [],  # Task 0
#         [0],  # Task 1 depends on Task 0
#         [0],  # Task 2 depends on Task 0
#         [1, 2],  # Task 3 depends on Tasks 1 and 2
#         [3],  # Task 4 depends on Task 3
#         [4],  # Task 5 depends on Task 4
#         [2],  # Task 6 depends on Task 2
#         [5, 6],  # Task 7 depends on Tasks 5 and 6
#         [7],  # Task 8 depends on Task 7
#         [8],  # Task 9 depends on Task 8
#         [8],  # Task 10 depends on Task 8
#         [9, 10],  # Task 11 depends on Tasks 9 and 10
#         [11],  # Task 12 depends on Task 11
#         [12],  # Task 13 depends on Task 12
#     ]
#     tasks = {
#         i: {'duration': durations[i], 'predecessors': predecessors[i], 'resources': resource_requirements[i]}
#         for i in range(num_tasks)
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 20
#     scouts = 5
#     max_trial = 5
#     best_schedule, best_makespan, start_times_of_activities = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
#
#     print("Test 3: 17 Activities\n")
#     #Description: A project with 17 activities and interdependent task chains.
#
    # num_tasks = 17
    # durations = [2, 4, 1, 3, 2, 5, 3, 1, 4, 2, 3, 2, 4, 1, 3, 2, 4]
    # resource_requirements = [
    #     [1, 2], [2, 1], [1, 1], [2, 3], [1, 1],
    #     [3, 2], [2, 2], [1, 1], [3, 3], [2, 2],
    #     [1, 2], [2, 1], [1, 3], [2, 1], [3, 2],
    #     [1, 2], [2, 3]
    # ]
    # resource_availabilities = [8, 8]
    # predecessors = [
    #     [], [0], [1], [2], [3],
    #     [4], [2], [5, 6], [7], [8],
    #     [9], [10], [11], [12], [13],
    #     [14], [15]
    # ]
    # tasks = {
    #     i: {'duration': durations[i], 'predecessors': predecessors[i], 'resources': resource_requirements[i]}
    #     for i in range(num_tasks)
    # }
    # project = RCPSP(
    #     num_tasks=len(tasks),
    #     durations=[task['duration'] for task in tasks.values()],
    #     resource_requirements=[task['resources'] for task in tasks.values()],
    #     resource_availabilities=resource_availabilities
    # )
    # project.tasks = tasks
    #
    # # Run the ABC algorithm
    # population_size = 10
    # scouts = 3
    # max_trial = 5
    # best_schedule, best_makespan, start_times_of_activities = abc(population_size, scouts, max_trial, project)
    #
    # print(f"Best schedule: {best_schedule}")
    # print(f"Best makespan: {best_makespan}\n")
    # validate_solution(project,best_schedule,start_times_of_activities)
    # plot_schedule(best_schedule, project, start_times_of_activities)

#     print("Test 4: 20 Activities\n")
#     # Description: A project with 20 activities and high complexity.
#
#     num_tasks = 20
#     durations = [1, 3, 2, 4, 5, 2, 3, 1, 4, 3, 2, 3, 1, 4, 2, 3, 1, 2, 4, 3]
#     resource_requirements = [
#         [1, 2], [2, 1], [1, 1], [3, 2], [2, 3],
#         [1, 1], [3, 2], [2, 1], [1, 3], [2, 2],
#         [3, 2], [2, 1], [1, 3], [2, 2], [3, 1],
#         [2, 1], [1, 2], [2, 3], [3, 2], [1, 1]
#     ]
#     resource_availabilities = [10, 10]
#     predecessors = [
#         [], [0], [0], [1, 2], [3],
#         [4], [2], [5, 6], [7], [8],
#         [9], [10], [11], [12], [13],
#         [14], [15], [16], [17], [18]
#     ]
#     tasks = {
#         i: {'duration': durations[i], 'predecessors': predecessors[i], 'resources': resource_requirements[i]}
#         for i in range(num_tasks)
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, start_times_of_activities = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
#
#     # Main code
#     print("Test 9\n")
#     # Description: A project where one task has zero duration.
#     num_tasks = 4
#     durations = [3, 0, 5, 2]
#     resource_requirements = [[1, 2], [0, 0], [2, 1], [1, 1]]
#     resource_availabilities = [5, 5]
#     predecessors = [
#         [],  # Task 0
#         [0],  # Task 1 depends on Task 0
#         [1],  # Task 2 depends on Task 1
#         [1],  # Task 3 depends on Task 1
#     ]
#     tasks = {
#         0: {'duration': 3, 'predecessors': [], 'resources': [1, 2]},
#         1: {'duration': 2, 'predecessors': [0], 'resources': [0, 0]},
#         2: {'duration': 5, 'predecessors': [1], 'resources': [2, 1]},
#         3: {'duration': 2, 'predecessors': [1,2], 'resources': [1, 1]},
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, task_start_times = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
#     # Plot the schedule
#     plot_schedule(best_schedule, project, task_start_times)

    # print("Test 9\n")
    # # Description: A larger project with tight resource constraints.
    # num_tasks = 6
    # durations = [2, 3, 4, 1, 5, 3]
    # resource_requirements = [[2, 3], [3, 1], [2, 2], [1, 1], [3, 2], [2, 3]]
    # resource_availabilities = [5, 4]
    # predecessors = [
    #     [],  # Task 0
    #     [0],  # Task 1 depends on Task 0
    #     [0],  # Task 2 depends on Task 0
    #     [1],  # Task 3 depends on Task 1
    #     [2, 3],  # Task 4 depends on Tasks 2 and 3
    #     [4],  # Task 5 depends on Task 4
    # ]
    # tasks = {
    #     0: {'duration': 2, 'predecessors': [], 'resources': [2, 3]},
    #     1: {'duration': 3, 'predecessors': [0], 'resources': [3, 1]},
    #     2: {'duration': 4, 'predecessors': [0], 'resources': [2, 2]},
    #     3: {'duration': 1, 'predecessors': [1], 'resources': [1, 1]},
    #     4: {'duration': 5, 'predecessors': [2, 3], 'resources': [3, 2]},
    #     5: {'duration': 3, 'predecessors': [4], 'resources': [2, 3]}
    # }
    # project = RCPSP(
    #     num_tasks=len(tasks),
    #     durations=[task['duration'] for task in tasks.values()],
    #     resource_requirements=[task['resources'] for task in tasks.values()],
    #     resource_availabilities=resource_availabilities
    # )
    # project.tasks = tasks
    #
    # # Run the ABC algorithm
    # population_size = 50
    # scouts = 5
    # max_trial = 10
    # best_schedule, best_makespan, task_start_times = abc(population_size, scouts, max_trial, project)
    #
    # print(f"test 9 Best schedule: {best_schedule}")
    # print(f"Best makespan: {best_makespan}\n")
    # plot_schedule(best_schedule, project, task_start_times)

# #Test 8: Tight Resource Constraints with Heavy Task Duration Overlap
#     print("Test 8\n")
#     # Description: A project with tight resource constraints and heavy task duration overlap to test resource usage.
#     num_tasks = 8
#     durations = [7, 4, 5, 3, 2, 6, 5, 4]
#     resource_requirements = [[4, 3], [3, 4], [4, 2], [3, 3], [2, 2], [3, 4], [2, 3], [4, 3]]
#     resource_availabilities = [10, 7]
#     predecessors = [
#         [],  # Task 0
#         [0],  # Task 1 depends on Task 0
#         [0, 1],  # Task 2 depends on Tasks 0 and 1
#         [1],  # Task 3 depends on Task 1
#         [2],  # Task 4 depends on Task 2
#         [3],  # Task 5 depends on Task 3
#         [4],  # Task 6 depends on Task 4
#         [5],  # Task 7 depends on Task 5
#     ]
#     tasks = {
#             0: {'duration': 3, 'predecessors': [], 'resources': [4, 3]},
#             1: {'duration': 2, 'predecessors': [0], 'resources': [3, 4]},
#             2: {'duration': 4, 'predecessors': [0,1], 'resources': [4, 2]},
#             3: {'duration': 1, 'predecessors': [1], 'resources': [3, 3]},
#             4: {'duration': 5, 'predecessors': [2], 'resources': [2, 2]},
#             5: {'duration': 3, 'predecessors': [3], 'resources': [3, 4]},
#             6: {'duration': 4, 'predecessors': [4], 'resources': [2, 3]},
#             7: {'duration': 2, 'predecessors': [5], 'resources': [4, 3]}
#         }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 60
#     scouts = 30
#     max_trial = 12
#     best_schedule, best_makespan, task_start_times = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
#     print("Start times: ", task_start_times)
#
#     # Plot the schedule
#     plot_schedule(best_schedule, project, task_start_times)
#TEST 9
    # num_tasks = 8
    # durations = [7, 4, 5, 3, 2, 6, 5, 4]
    # resource_requirements = [[4, 3], [3, 4], [4, 2], [3, 3], [2, 2], [3, 4], [2, 3], [4, 3]]
    # resource_availabilities = [10, 7]
    # predecessors = [
    #     [],  # Task 0
    #     [0],  # Task 1 depends on Task 0
    #     [0, 1],  # Task 2 depends on Tasks 0 and 1
    #     [1],  # Task 3 depends on Task 1
    #     [2],  # Task 4 depends on Task 2
    #     [3],  # Task 5 depends on Task 3
    #     [4],  # Task 6 depends on Task 4
    #     [5],  # Task 7 depends on Task 5
    # ]
    # tasks = {
    #     i: {'duration': durations[i], 'predecessors': predecessors[i], 'resources': resource_requirements[i]} for i in range(num_tasks)
    # }
    # project = RCPSP(
    #     num_tasks=len(tasks),
    #     durations=[task['duration'] for task in tasks.values()],
    #     resource_requirements=[task['resources'] for task in tasks.values()],
    #     resource_availabilities=resource_availabilities
    # )
    # project.tasks = tasks
    #
    # # Run the ABC algorithm
    # population_size = 50
    # scouts = 5
    # max_trial = 15
    # best_schedule, best_makespan, task_start_times = abc(population_size, scouts, max_trial, project)
    #
    # print(f"ABC Makespan: {best_makespan}")
    # print(f"ABC Schedule: {task_start_times}")
    # print("Start times: ", task_start_times)
    # plot_schedule(best_schedule, project, task_start_times)

# #CIRCULAR TESTS - inf none expected
#     print("Test C1\n")
#     # Description: Tests a project with task dependencies but no resource conflicts.
#     num_tasks = 6
#     durations = [3, 2, 4, 5, 3, 6]
#     resource_requirements = [[1, 2], [2, 3], [2, 1], [1, 4], [3, 2], [2, 3]]
#     resource_availabilities = [5, 5]
#     predecessors = [
#         [],  # Task 0
#         [0],  # Task 1 depends on Task 0
#         [1],  # Task 2 depends on Task 1
#         [2],  # Task 3 depends on Task 2
#         [2],  # Task 4 depends on Task 2
#         [4, 5],  # Task 5 depends on Tasks 4 and 5
#     ]
#     tasks = {
#             0: {'duration': 3, 'predecessors': [], 'resources': [1, 2]},
#             1: {'duration': 2, 'predecessors': [0], 'resources': [2, 3]},
#             2: {'duration': 4, 'predecessors': [1], 'resources': [2, 1]},
#             3: {'duration': 5, 'predecessors': [2], 'resources': [1, 4]},
#             4: {'duration': 3, 'predecessors': [2], 'resources': [3, 2]},
#             5: {'duration': 6, 'predecessors': [4, 5], 'resources': [2, 3]}
#         }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, task_start_times = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}")
#
#     # Test 7: Circular Precedence Edge Case (Invalid Input)
#     print("Test C2\n")
#     # Description: This test checks if the system gracefully handles an invalid input with circular dependencies.
#     num_tasks = 3
#     durations = [3, 2, 4]
#     resource_requirements = [[1, 2], [2, 1], [1, 1]]
#     resource_availabilities = [3, 3]
#     predecessors = [
#         [2],  # Task 0 depends on Task 2
#         [0],  # Task 1 depends on Task 0
#         [1],  # Task 2 depends on Task 1 (circular dependency)
#     ]
#
#     # Test 5: Circular Dependency (Simple Loop)
#     # Description: A project where Task 1 depends on Task 2 and Task 2 depends on Task 1.
#     num_tasks = 3
#     durations = [4, 3, 5]
#     resource_requirements = [[1, 1], [2, 2], [1, 3]]
#     resource_availabilities = [5, 5]
#     predecessors = [
#         [],  # Task 0
#         [2],  # Task 1 depends on Task 2
#         [1],  # Task 2 depends on Task 1
#     ]
#     tasks = {
#         0: {'duration': 4, 'predecessors': [], 'resources': [1, 1]},
#         1: {'duration': 3, 'predecessors': [2], 'resources': [2, 2]},
#         2: {'duration': 5, 'predecessors': [1], 'resources': [1, 3]},
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, task_start_times = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
#
#     # Test 6: Circular Dependency (Complex)
#     print("Test C3\n")
#     # Description: A project with a more complex circular dependency among multiple tasks.
#     num_tasks = 4
#     durations = [3, 6, 2, 4]
#     resource_requirements = [[1, 1], [2, 3], [1, 2], [3, 1]]
#     resource_availabilities = [5, 5]
#     predecessors = [
#         [3],  # Task 0 depends on Task 3
#         [0],  # Task 1 depends on Task 0
#         [1],  # Task 2 depends on Task 1
#         [2],  # Task 3 depends on Task 2
#     ]
#     tasks = {
#         0: {'duration': 3, 'predecessors': [3], 'resources': [1, 1]},
#         1: {'duration': 6, 'predecessors': [0], 'resources': [2, 3]},
#         2: {'duration': 2, 'predecessors': [1], 'resources': [1, 2]},
#         3: {'duration': 4, 'predecessors': [2], 'resources': [3, 1]},
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, task_start_times = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
#
#     # Test 7: Self-Dependency
#     print("Test C4\n")
#     # Description: A project where a task depends on itself, creating a circular dependency.
#     num_tasks = 2
#     durations = [4, 3]
#     resource_requirements = [[1, 2], [2, 3]]
#     resource_availabilities = [5, 5]
#     predecessors = [
#         [0],  # Task 0 depends on itself
#         [0],  # Task 1 depends on Task 0
#     ]
#     tasks = {
#         0: {'duration': 4, 'predecessors': [0], 'resources': [1, 2]},
#         1: {'duration': 3, 'predecessors': [0], 'resources': [2, 3]},
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, task_start_times = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
#     # Test 8: Disconnected Graph (Valid but Contains Isolated Task)
#     print("Test C5\n")
#     # Description: A project with a disconnected graph where one task has no predecessors or successors.
#     num_tasks = 3
#     durations = [2, 5, 3]
#     resource_requirements = [[1, 1], [2, 2], [1, 3]]
#     resource_availabilities = [5, 5]
#     predecessors = [
#         [],  # Task 0
#         [0],  # Task 1 depends on Task 0
#         [2],  # Task 2 depends on itself (Circular Dependency)
#     ]
#     tasks = {
#         0: {'duration': 2, 'predecessors': [], 'resources': [1, 1]},
#         1: {'duration': 5, 'predecessors': [0], 'resources': [2, 2]},
#         2: {'duration': 3, 'predecessors': [2], 'resources': [1, 3]},
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, task_start_times = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")

# batch2 tests are in testing-bruteForce.py where the results are compared to the brufeForce algorithm
# NOTE: these tests are consisted usually of under 12 activities for bruteForce permormance reasons

# # batch3 - COMPLEX, EDGE CASES ETC with more activities
#     print("Test 9\n")
#     # Description: A project where one task has zero duration.
#     num_tasks = 4
#     durations = [3, 0, 5, 2]
#     resource_requirements = [[1, 2], [0, 0], [2, 1], [1, 1]]
#     resource_availabilities = [5, 5]
#     predecessors = [
#         [],  # Task 0
#         [0],  # Task 1 depends on Task 0
#         [1],  # Task 2 depends on Task 1
#         [1],  # Task 3 depends on Task 1
#     ]
#     tasks = {
#         0: {'duration': 3, 'predecessors': [], 'resources': [1, 2]},
#         1: {'duration': 0, 'predecessors': [0], 'resources': [0, 0]},
#         2: {'duration': 5, 'predecessors': [1], 'resources': [2, 1]},
#         3: {'duration': 2, 'predecessors': [1], 'resources': [1, 1]},
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, start_times_of_activities = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
#
#     print("Test 10\n")
#     # Description: A project where one task requires zero resources.
#     num_tasks = 3
#     durations = [4, 3, 5]
#     resource_requirements = [[1, 1], [0, 0], [2, 3]]
#     resource_availabilities = [5, 5]
#     predecessors = [
#         [],  # Task 0
#         [0],  # Task 1 depends on Task 0
#         [1],  # Task 2 depends on Task 1
#     ]
#     tasks = {
#         0: {'duration': 4, 'predecessors': [], 'resources': [1, 1]},
#         1: {'duration': 3, 'predecessors': [0], 'resources': [0, 0]},
#         2: {'duration': 5, 'predecessors': [1], 'resources': [2, 3]},
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, start_times_of_activities = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
#
#     print("Test 11\n")
#     # Description: A project where resource availability exactly matches demand.
#     num_tasks = 4
#     durations = [3, 2, 4, 5]
#     resource_requirements = [[1, 2], [2, 3], [3, 2], [4, 1]]
#     resource_availabilities = [10, 8]  # Total demand is exactly matched.
#     predecessors = [
#         [],  # Task 0
#         [0],  # Task 1 depends on Task 0
#         [1],  # Task 2 depends on Task 1
#         [1],  # Task 3 depends on Task 1
#     ]
#     tasks = {
#         0: {'duration': 3, 'predecessors': [], 'resources': [1, 2]},
#         1: {'duration': 2, 'predecessors': [0], 'resources': [2, 3]},
#         2: {'duration': 4, 'predecessors': [1], 'resources': [3, 2]},
#         3: {'duration': 5, 'predecessors': [1], 'resources': [4, 1]},
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, start_times_of_activities = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
#
#     print("Test 1: 10 Activities\n")
#     # Description: A project with 10 activities and balanced resource availability.
#
#     num_tasks = 10
#     durations = [2, 3, 1, 4, 5, 2, 3, 4, 2, 1]
#     resource_requirements = [
#         [2, 1], [1, 2], [1, 1], [2, 3], [1, 1],
#         [2, 2], [1, 3], [2, 1], [3, 2], [1, 1]
#     ]
#     resource_availabilities = [6, 6]
#     predecessors = [
#         [],  # Task 0
#         [0],  # Task 1 depends on Task 0
#         [0],  # Task 2 depends on Task 0
#         [1, 2],  # Task 3 depends on Tasks 1 and 2
#         [3],  # Task 4 depends on Task 3
#         [2],  # Task 5 depends on Task 2
#         [4, 5],  # Task 6 depends on Tasks 4 and 5
#         [5],  # Task 7 depends on Task 5
#         [6],  # Task 8 depends on Task 6
#         [8, 7],  # Task 9 depends on Tasks 8 and 7
#     ]
#     tasks = {
#         i: {'duration': durations[i], 'predecessors': predecessors[i], 'resources': resource_requirements[i]}
#         for i in range(num_tasks)
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, start_times_of_activities = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
#
#     print("Test 2: 14 Activities\n")
#     # Description: A project with 14 activities, moderate dependencies, and manageable resources.
#
#     num_tasks = 14
#     durations = [3, 2, 4, 1, 2, 5, 3, 2, 4, 3, 1, 2, 3, 4]
#     resource_requirements = [
#         [1, 1], [2, 3], [1, 2], [3, 1], [2, 2],
#         [1, 3], [2, 1], [1, 1], [3, 2], [2, 3],
#         [1, 2], [2, 1], [1, 1], [3, 2]
#     ]
#     resource_availabilities = [7, 7]
#     predecessors = [
#         [],  # Task 0
#         [0],  # Task 1 depends on Task 0
#         [0],  # Task 2 depends on Task 0
#         [1, 2],  # Task 3 depends on Tasks 1 and 2
#         [3],  # Task 4 depends on Task 3
#         [4],  # Task 5 depends on Task 4
#         [2],  # Task 6 depends on Task 2
#         [5, 6],  # Task 7 depends on Tasks 5 and 6
#         [7],  # Task 8 depends on Task 7
#         [8],  # Task 9 depends on Task 8
#         [8],  # Task 10 depends on Task 8
#         [9, 10],  # Task 11 depends on Tasks 9 and 10
#         [11],  # Task 12 depends on Task 11
#         [12],  # Task 13 depends on Task 12
#     ]
#     tasks = {
#         i: {'duration': durations[i], 'predecessors': predecessors[i], 'resources': resource_requirements[i]}
#         for i in range(num_tasks)
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 20
#     scouts = 5
#     max_trial = 5
#     best_schedule, best_makespan, start_times_of_activities = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
#
#     print("Test 3: 17 Activities\n")
#     # Description: A project with 17 activities and interdependent task chains.
#
#     num_tasks = 17
#     durations = [2, 4, 1, 3, 2, 5, 3, 1, 4, 2, 3, 2, 4, 1, 3, 2, 4]
#     resource_requirements = [
#         [1, 2], [2, 1], [1, 1], [2, 3], [1, 1],
#         [3, 2], [2, 2], [1, 1], [3, 3], [2, 2],
#         [1, 2], [2, 1], [1, 3], [2, 1], [3, 2],
#         [1, 2], [2, 3]
#     ]
#     resource_availabilities = [8, 8]
#     predecessors = [
#         [], [0], [1], [2], [3],
#         [4], [2], [5, 6], [7], [8],
#         [9], [10], [11], [12], [13],
#         [14], [15]
#     ]
#     tasks = {
#         i: {'duration': durations[i], 'predecessors': predecessors[i], 'resources': resource_requirements[i]}
#         for i in range(num_tasks)
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, start_times_of_activities = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
#
#     print("Test 4: 20 Activities\n")
#     # Description: A project with 20 activities and high complexity.
#
#     num_tasks = 20
#     durations = [1, 3, 2, 4, 5, 2, 3, 1, 4, 3, 2, 3, 1, 4, 2, 3, 1, 2, 4, 3]
#     resource_requirements = [
#         [1, 2], [2, 1], [1, 1], [3, 2], [2, 3],
#         [1, 1], [3, 2], [2, 1], [1, 3], [2, 2],
#         [3, 2], [2, 1], [1, 3], [2, 2], [3, 1],
#         [2, 1], [1, 2], [2, 3], [3, 2], [1, 1]
#     ]
#     resource_availabilities = [10, 10]
#     predecessors = [
#         [], [0], [0], [1, 2], [3],
#         [4], [2], [5, 6], [7], [8],
#         [9], [10], [11], [12], [13],
#         [14], [15], [16], [17], [18]
#     ]
#     tasks = {
#         i: {'duration': durations[i], 'predecessors': predecessors[i], 'resources': resource_requirements[i]}
#         for i in range(num_tasks)
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, start_times_of_activities = abc(population_size, scouts, max_trial, project)
#
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}\n")
# #
#     # Main code
#     print("Test 9\n")
#     # Description: A project where one task has zero duration.
#     num_tasks = 4
#     durations = [3, 0, 5, 2]
#     resource_requirements = [[1, 2], [0, 0], [2, 1], [1, 1]]
#     resource_availabilities = [5, 5]
#     predecessors = [
#         [],  # Task 0
#         [0],  # Task 1 depends on Task 0
#         [1],  # Task 2 depends on Task 1
#         [1],  # Task 3 depends on Task 1
#     ]
#     tasks = {
#         0: {'duration': 3, 'predecessors': [], 'resources': [1, 2]},
#         1: {'duration': 2, 'predecessors': [0], 'resources': [0, 0]},
#         2: {'duration': 5, 'predecessors': [1], 'resources': [2, 1]},
#         3: {'duration': 2, 'predecessors': [1, 2], 'resources': [1, 1]},
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 10
#     scouts = 3
#     max_trial = 5
#     best_schedule, best_makespan, task_start_times = abc(population_size, scouts, max_trial, project)
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}")
#     print(f"Start times: {task_start_times}\n")
#     print("Test with 40 Tasks\n")
#     # Description: A project with 40 tasks, non-linear dependencies, and mixed resource requirements.
#     num_tasks = 40
#     durations = [3, 5, 2, 4, 6, 3, 2, 5, 1, 3, 7, 2, 3, 4, 5, 6, 3, 4, 2, 5, 1, 2, 6, 4, 5, 3, 2, 3, 7, 2, 5, 6, 3, 2, 4, 5, 3, 1, 4, 2]
#     resource_requirements = [
#         [1, 2], [2, 3], [1, 1], [3, 2], [2, 2], [1, 1], [0, 1], [3, 2], [2, 0], [1, 2],
#         [2, 2], [3, 1], [1, 0], [3, 3], [2, 1], [1, 2], [2, 1], [3, 2], [1, 0], [2, 3],
#         [0, 1], [1, 1], [3, 2], [2, 2], [1, 0], [3, 1], [1, 2], [2, 1], [1, 1], [2, 3],
#         [3, 2], [2, 1], [1, 2], [3, 3], [2, 0], [1, 1], [0, 2], [3, 2], [2, 1], [1, 2]
#     ]
#     resource_availabilities = [10, 10]
#     predecessors = [
#         [],           # Task 0
#         [0],          # Task 1 depends on Task 0
#         [0],          # Task 2 depends on Task 0
#         [1],          # Task 3 depends on Task 1
#         [2],          # Task 4 depends on Task 2
#         [1, 2],       # Task 5 depends on Task 1 and Task 2
#         [4],          # Task 6 depends on Task 4
#         [3],          # Task 7 depends on Task 3
#         [6],          # Task 8 depends on Task 6
#         [7],          # Task 9 depends on Task 7
#         [5],          # Task 10 depends on Task 5
#         [8, 9],       # Task 11 depends on Task 8 and Task 9
#         [10],         # Task 12 depends on Task 10
#         [11],         # Task 13 depends on Task 11
#         [12],         # Task 14 depends on Task 12
#         [13],         # Task 15 depends on Task 13
#         [14, 8],      # Task 16 depends on Task 14 and Task 8
#         [15],         # Task 17 depends on Task 15
#         [16],         # Task 18 depends on Task 16
#         [17],         # Task 19 depends on Task 17
#         [18, 11],     # Task 20 depends on Task 18 and Task 11
#         [19],         # Task 21 depends on Task 19
#         [20],         # Task 22 depends on Task 20
#         [21, 10],     # Task 23 depends on Task 21 and Task 10
#         [22],         # Task 24 depends on Task 22
#         [23],         # Task 25 depends on Task 23
#         [24, 12],     # Task 26 depends on Task 24 and Task 12
#         [25],         # Task 27 depends on Task 25
#         [26],         # Task 28 depends on Task 26
#         [27],         # Task 29 depends on Task 27
#         [28, 15],     # Task 30 depends on Task 28 and Task 15
#         [29],         # Task 31 depends on Task 29
#         [30],         # Task 32 depends on Task 30
#         [31],         # Task 33 depends on Task 31
#         [32, 20],     # Task 34 depends on Task 32 and Task 20
#         [33],         # Task 35 depends on Task 33
#         [34],         # Task 36 depends on Task 34
#         [35],         # Task 37 depends on Task 35
#         [36],         # Task 38 depends on Task 36
#         [37, 22]      # Task 39 depends on Task 37 and Task 22
#     ]
#
#     tasks = {
#         i: {'duration': durations[i], 'predecessors': predecessors[i], 'resources': resource_requirements[i]} for i in range(num_tasks)
#     }
#
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 150
#     scouts = 15
#     max_trial = 25
#     best_schedule, best_makespan, task_start_times = abc(population_size, scouts, max_trial, project)
#     print(f"Best schedule: {best_schedule}")
#     print(f"Best makespan: {best_makespan}")
#     print(f"Start times: {task_start_times}\n")
#     #test
#     is_valid = validate_solution(project, best_schedule, task_start_times)
#     plot_schedule(best_schedule, project, task_start_times)
#
#     print("TEST 1: Basic Dependencies\n") #8 vs 9
#     num_tasks = 4
#     durations = [3, 2, 4, 1]
#     resource_requirements = [[2, 1], [1, 2], [3, 3], [1, 1]]
#     resource_availabilities = [4, 4] #4,5
#     predecessors = [
#         [],  # Task 0
#         [0],  # Task 1 depends on Task 0
#         [0],  # Task 2 depends on Task 0
#         [1, 2],  # Task 3 depends on Tasks 1 and 2
#     ]
#     tasks = {
#         i: {'duration': durations[i], 'predecessors': predecessors[i], 'resources': resource_requirements[i]} for i in range(num_tasks)
#     }
#     project = RCPSP(
#         num_tasks=len(tasks),
#         durations=[task['duration'] for task in tasks.values()],
#         resource_requirements=[task['resources'] for task in tasks.values()],
#         resource_availabilities=resource_availabilities
#     )
#     project.tasks = tasks
#
#     # Run the ABC algorithm
#     population_size = 50
#     scouts = 5
#     max_trial = 20
#     best_schedule, best_makespan, task_start_times = abc(population_size, scouts, max_trial, project)
#
#     print(f"ABC Makespan: {best_makespan}")
#     print(f"ABC Schedule: {best_schedule}")
#     print(f"ABC start times: {task_start_times}")
#     plot_schedule(best_schedule,project,task_start_times)


    # print("Test with 40 Tasks\n")
    # # Description: A project with 40 tasks, non-linear dependencies, and mixed resource requirements.
    # num_tasks = 40
    # durations = [3, 5, 2, 4, 6, 3, 2, 5, 1, 3, 7, 2, 3, 4, 5, 6, 3, 4, 2, 5, 1, 2, 6, 4, 5, 3, 2, 3, 7, 2, 5, 6, 3, 2, 4, 5, 3, 1, 4, 2]
    # resource_requirements = [
    #     [1, 2], [2, 3], [1, 1], [3, 2], [2, 2], [1, 1], [0, 1], [3, 2], [2, 0], [1, 2],
    #     [2, 2], [3, 1], [1, 0], [3, 3], [2, 1], [1, 2], [2, 1], [3, 2], [1, 0], [2, 3],
    #     [0, 1], [1, 1], [3, 2], [2, 2], [1, 0], [3, 1], [1, 2], [2, 1], [1, 1], [2, 3],
    #     [3, 2], [2, 1], [1, 2], [3, 3], [2, 0], [1, 1], [0, 2], [3, 2], [2, 1], [1, 2]
    # ]
    # resource_availabilities = [10, 10]
    # predecessors = [
    #     [],           # Task 0
    #     [0],          # Task 1 depends on Task 0
    #     [0],          # Task 2 depends on Task 0
    #     [1],          # Task 3 depends on Task 1
    #     [2],          # Task 4 depends on Task 2
    #     [1, 2],       # Task 5 depends on Task 1 and Task 2
    #     [4],          # Task 6 depends on Task 4
    #     [3],          # Task 7 depends on Task 3
    #     [6],          # Task 8 depends on Task 6
    #     [7],          # Task 9 depends on Task 7
    #     [5],          # Task 10 depends on Task 5
    #     [8, 9],       # Task 11 depends on Task 8 and Task 9
    #     [10],         # Task 12 depends on Task 10
    #     [11],         # Task 13 depends on Task 11
    #     [12],         # Task 14 depends on Task 12
    #     [13],         # Task 15 depends on Task 13
    #     [14, 8],      # Task 16 depends on Task 14 and Task 8
    #     [15],         # Task 17 depends on Task 15
    #     [16],         # Task 18 depends on Task 16
    #     [17],         # Task 19 depends on Task 17
    #     [18, 11],     # Task 20 depends on Task 18 and Task 11
    #     [19],         # Task 21 depends on Task 19
    #     [20],         # Task 22 depends on Task 20
    #     [21, 10],     # Task 23 depends on Task 21 and Task 10
    #     [22],         # Task 24 depends on Task 22
    #     [23],         # Task 25 depends on Task 23
    #     [24, 12],     # Task 26 depends on Task 24 and Task 12
    #     [25],         # Task 27 depends on Task 25
    #     [26],         # Task 28 depends on Task 26
    #     [27],         # Task 29 depends on Task 27
    #     [28, 15],     # Task 30 depends on Task 28 and Task 15
    #     [29],         # Task 31 depends on Task 29
    #     [30],         # Task 32 depends on Task 30
    #     [31],         # Task 33 depends on Task 31
    #     [32, 20],     # Task 34 depends on Task 32 and Task 20
    #     [33],         # Task 35 depends on Task 33
    #     [34],         # Task 36 depends on Task 34
    #     [35],         # Task 37 depends on Task 35
    #     [36],         # Task 38 depends on Task 36
    #     [37, 22]      # Task 39 depends on Task 37 and Task 22
    # ]
    #
    # tasks = {
    #     i: {'duration': durations[i], 'predecessors': predecessors[i], 'resources': resource_requirements[i]} for i in range(num_tasks)
    # }
    #
    # project = RCPSP(
    #     num_tasks=len(tasks),
    #     durations=[task['duration'] for task in tasks.values()],
    #     resource_requirements=[task['resources'] for task in tasks.values()],
    #     resource_availabilities=resource_availabilities
    # )
    # project.tasks = tasks
    #
    # # Run the ABC algorithm
    # population_size = 15
    # scouts = 5
    # max_trial = 5
    # best_schedule, best_makespan, task_start_times = abc(population_size, scouts, max_trial, project)
    # print(f"Best schedule: {best_schedule}")
    # print(f"Best makespan: {best_makespan}")
    # print(f"Start times: {task_start_times}\n")
    # #test
    # is_valid = validate_solution(project, best_schedule, task_start_times)
    # plot_schedule(best_schedule, project, task_start_times)


# # Test 8: Tight Resource Constraints with Heavy Task Duration Overlap
    # print("Test 8\n")
    # # Description: A project with tight resource constraints and heavy task duration overlap to test resource usage.
    # num_tasks = 8
    # durations = [7, 4, 5, 3, 2, 6, 5, 4]
    # resource_requirements = [[4, 3], [3, 4], [4, 2], [3, 3], [2, 2], [3, 4], [2, 3], [4, 3]]
    # resource_availabilities = [10, 7]
    # predecessors = [
    #     [],  # Task 0
    #     [0],  # Task 1 depends on Task 0
    #     [0, 1],  # Task 2 depends on Tasks 0 and 1
    #     [1],  # Task 3 depends on Task 1
    #     [2],  # Task 4 depends on Task 2
    #     [3],  # Task 5 depends on Task 3
    #     [4],  # Task 6 depends on Task 4
    #     [5],  # Task 7 depends on Task 5
    # ]
    # tasks = {
    #     0: {'duration': 3, 'predecessors': [], 'resources': [4, 3]},
    #     1: {'duration': 2, 'predecessors': [0], 'resources': [3, 4]},
    #     2: {'duration': 4, 'predecessors': [0, 1], 'resources': [4, 2]},
    #     3: {'duration': 1, 'predecessors': [1], 'resources': [3, 3]},
    #     4: {'duration': 5, 'predecessors': [2], 'resources': [2, 2]},
    #     5: {'duration': 3, 'predecessors': [3], 'resources': [3, 4]},
    #     6: {'duration': 4, 'predecessors': [4], 'resources': [2, 3]},
    #     7: {'duration': 2, 'predecessors': [5], 'resources': [4, 3]}
    # }
    # project = RCPSP(
    #     num_tasks=len(tasks),
    #     durations=[task['duration'] for task in tasks.values()],
    #     resource_requirements=[task['resources'] for task in tasks.values()],
    #     resource_availabilities=resource_availabilities
    # )
    # project.tasks = tasks

    # print("Trial 10\n")
    # # Description: A larger project with tight resource constraints and overlapping tasks.
    # num_tasks = 8
    # durations = [3, 2, 4, 3, 5, 6, 2, 4]
    # resource_requirements = [[2, 1], [3, 2], [4, 3], [1, 2], [3, 1], [2, 4], [1, 1], [3, 3]]
    # resource_availabilities = [6, 5]
    # predecessors = [
    #     [],         # Task 0
    #     [0],        # Task 1 depends on Task 0
    #     [0],        # Task 2 depends on Task 0
    #     [1],        # Task 3 depends on Task 1
    #     [1, 2],     # Task 4 depends on Tasks 1 and 2
    #     [3],        # Task 5 depends on Task 3
    #     [4],        # Task 6 depends on Task 4
    #     [5, 6]      # Task 7 depends on Tasks 5 and 6
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
    #
    # # Run the ABC algorithm, trial1 dole a trial2 - 60,3,12
    # population_size = 60
    # scouts = 5
    # max_trial = 10
    # best_schedule, best_makespan, task_start_times = abc(population_size, scouts, max_trial, project)
    #
    # print(f"Best schedule: {best_schedule}")
    # print(f"Best makespan: {best_makespan}\n")
    # print("Start times: ", task_start_times)

    # Plot the schedule
    #plot_schedule(best_schedule, project, task_start_times)

    print("Eksperiment 3\n")
    # Description: A project where resource availability exactly matches demand.
    num_tasks = 13
    durations = [3, 8, 2, 4, 3, 6, 5, 7, 9, 2, 10, 4, 1]
    resource_requirements = [
        [1, 2], [2, 3], [3, 2], [4, 1], [1, 1],
        [2, 2], [3, 3], [2, 4], [4, 2], [1, 3],
        [2, 2], [3, 1], [1, 1]
    ]
    resource_availabilities = [16, 10]  # Adjusted to allow parallel activities with constraints.
    predecessors = [
        [],  # Task 0: Independent
        [0],  # Task 1 depends on Task 0
        [1],  # Task 2 depends on Task 1
        [1],  # Task 3 depends on Task 1
        [0],  # Task 4 depends on Task 0
        [4],  # Task 5 depends on Task 4
        [2, 3],  # Task 6 depends on Tasks 2 and 3
        [5],  # Task 7 depends on Task 5
        [7],  # Task 8 depends on Task 7
        [6, 8],  # Task 9 depends on Tasks 6 and 8
        [],  # Task 10: Independent
        [10],  # Task 11 depends on Task 10
        [11, 9]  # Task 12 depends on Tasks 11 and 9
    ]
    tasks = {
        0: {'duration': 3, 'predecessors': [], 'resources': [1, 2]},
        1: {'duration': 8, 'predecessors': [0], 'resources': [2, 3]},
        2: {'duration': 2, 'predecessors': [1], 'resources': [3, 2]},
        3: {'duration': 4, 'predecessors': [1], 'resources': [4, 1]},
        4: {'duration': 3, 'predecessors': [0], 'resources': [1, 1]},
        5: {'duration': 6, 'predecessors': [4], 'resources': [2, 2]},
        6: {'duration': 5, 'predecessors': [2, 3], 'resources': [3, 3]},
        7: {'duration': 7, 'predecessors': [5], 'resources': [2, 4]},
        8: {'duration': 9, 'predecessors': [7], 'resources': [4, 2]},
        9: {'duration': 2, 'predecessors': [6, 8], 'resources': [1, 3]},
        10: {'duration': 10, 'predecessors': [], 'resources': [2, 2]},
        11: {'duration': 4, 'predecessors': [10], 'resources': [3, 1]},
        12: {'duration': 1, 'predecessors': [11, 9], 'resources': [1, 1]},
    }
    project = RCPSP(
        num_tasks=len(tasks),
        durations=[task['duration'] for task in tasks.values()],
        resource_requirements=[task['resources'] for task in tasks.values()],
        resource_availabilities=resource_availabilities
    )
    project.tasks = tasks

    # Run the ABC algorithm
    population_size = 10
    scouts = 2
    max_trial = 5
    best_schedule, best_makespan, start_times_of_activities = abc(population_size, scouts, max_trial, project)

    print(f"Best schedule: {best_schedule}")
    print(f"Best makespan: {best_makespan}\n")
    plot_schedule(best_schedule, project, start_times_of_activities)