"""Code used to generate workflows from input traces."""
import random
import json

import loader

TOTAL_WORKFLOWS = 2**8

count = 4
tasks = loader.swf('traces/KIT-FH2-2016-1.swf')
# generate linear chain workflows
first = 0
for i in range(TOTAL_WORKFLOWS):
    length = random.randint(2, 9)
    linear_chain = [
        loader.convert_task(task) for task in tasks[first:first+length]
        if task['status'] == 1
    ]
    linear_chain = {task['id']: task for task in linear_chain}
    last_task_id = None
    for task_id in linear_chain:
        if last_task_id is not None:
            # add a data dependency between this task and the last task
            linear_chain[task_id]['data_deps'][last_task_id] = linear_chain[last_task_id]['cores']
        last_task_id = task_id
    with open('data/linear-chain-%i.json' % (i,), 'w') as fp:
        json.dump(linear_chain, fp, indent=4)
    first += length
