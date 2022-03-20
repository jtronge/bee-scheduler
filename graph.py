from matplotlib import pyplot as plt

fig, ax = plt.subplots()
ax.boxplot(
    [[1.0, 88.0, 3.8, 9.0, -100], [88.0, 0.0, 1.0, 3.0]],
    labels=('A', 'B'),
)
ax.set_ylabel('Workflow makespan (s)')
ax.set_xlabel('Algorithms')
plt.show()
