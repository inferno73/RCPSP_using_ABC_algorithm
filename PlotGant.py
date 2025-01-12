import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
#plot used to mimic Gants diagram, visually represents the processed data from ABC algorithm
def plot_schedule(schedule, project, task_start_times):
    if schedule is None or task_start_times is None:
        print("Invalid parameters, plot cannot be shown.\n")
        return
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate a list of task names and their start times and durations
    task_names = []
    start_times = []
    durations = []
    colors = list(mcolors.TABLEAU_COLORS.values())  # Use a set of predefined colors

    # Create task bars for Gantt chart
    for i, task_id in enumerate(schedule):
        task = project.tasks[task_id]
        task_names.append(f"Task {task_id}")
        start_times.append(task_start_times[task_id])  # Use actual start time
        durations.append(task['duration'])

    # Create bars for each task
    for i in range(len(task_names)):
        ax.barh(task_names[i], durations[i], left=start_times[i], color=colors[i % len(colors)], edgecolor='black')

    # Set x-axis ticks to display integers
    max_time = max(start_times[i] + durations[i] for i in range(len(start_times)))
    ax.set_xticks(range(1, max_time + 1))  # Show numbers from 1 to the max time

    # Set plot labels
    ax.set_xlabel('Time')
    ax.set_title('Task Schedule')
    ax.grid(True)

    # Adjust plot appearance
    plt.tight_layout()
    plt.show()