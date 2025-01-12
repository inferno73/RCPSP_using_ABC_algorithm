from ortools.sat.python import cp_model

model = cp_model.CpModel()
tasks = {'A': 2, 'B': 4, 'C': 3, 'D': 1}
resources = [3, 2, 2, 1]

start_vars = {}
end_vars = {}

for task, duration in tasks.items():
    start_vars[task] = model.NewIntVar(0, 100, f'start_{task}')
    end_vars[task] = model.NewIntVar(0, 100, f'end_{task}')
    model.Add(end_vars[task] == start_vars[task] + duration)

# Add precedences
precedences = [('A', 'B'), ('A', 'C'), ('B', 'D')]
for pred, succ in precedences:
    model.Add(end_vars[pred] <= start_vars[succ])

# Define makespan variable (the maximum end time)
makespan = model.NewIntVar(0, 100, 'makespan')
for task in tasks:
    model.Add(makespan >= end_vars[task])

# Objective
model.Minimize(makespan)

# Solve the model
solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL:
    print(f"Optimal schedule found with makespan: {solver.Value(makespan)}")
    for task in tasks:
        print(f"{task}: Start = {solver.Value(start_vars[task])}, End = {solver.Value(end_vars[task])}")
else:
    print("No optimal solution found.")
