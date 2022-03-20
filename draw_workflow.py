import argparse
import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def partition_bfs(workflow):
    complete = set()
    partitions = {task_id: 0 for task_id in workflow}
    # insert any tasks without dependencies into the queue
    queue = [task_id for task_id in workflow if not workflow[task_id]['data_deps']]
    while len(queue) > 0:
        task_id = queue.pop(0)
        part = partitions[task_id]
        children = [child_id for child_id in workflow if task_id in workflow[child_id]['data_deps']]
        for child_id in children:
            cpart = part + 1
            # update the partition
            if cpart > partitions[child_id]:
                partitions[child_id] = cpart
            # skip the child if we already searched it
            if child_id in complete:
                continue
            complete.add(child_id)
            queue.append(child_id)
    # build the partitions
    real_partitions = []
    max_part_no = max(partitions[task_id] for task_id in partitions)
    for i in range(max_part_no + 1):
        part = [task_id for task_id in partitions if partitions[task_id] == i]
        real_partitions.append(part)
    return real_partitions


parser = argparse.ArgumentParser(description='display a graph of a workflow DAG')
parser.add_argument('graph_file', help='location of JSON graph file')
args = parser.parse_args()
G = nx.Graph()
with open(args.graph_file) as fp:
    workflow = json.load(fp)
# partition the workflow graph
partitions = partition_bfs(workflow)
max_count = max(len(part) for part in partitions)
# create the layout
layout = {}
width = 200
height = 20
for i, part in enumerate(partitions):
    count = len(part)
    y = (len(partitions) - i) * height
    for j, task_id in enumerate(part):
        # 0.5 is added to nudge nodes over on levels with fewer tasks
        x = (j + 0.5) * width / count
        layout[task_id] = (x, y)
print(partitions)
for task_id in workflow:
    task = workflow[task_id]
    for other_task_id in task['data_deps']:
        G.add_edge(task_id, other_task_id)
# nx.draw(G, layout)
nx.draw_networkx(
    G,
    layout,
    node_size=3200,
    node_color='#ffffff',
    linewidths=2, # width of the edge around nodes
    edgecolors='#000000', # color of the edge around nodes
    width=2, # width of the edges connecting nodes
)
plt.show()
