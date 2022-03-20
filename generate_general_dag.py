""" """
import random
import json

import loader

TOTAL_WORKFLOWS = 2**8

tasks = loader.swf('traces/KIT-FH2-2016-1.swf')
pos = 0
for i in range(TOTAL_WORKFLOWS):
    step_count = random.randint(3, 4)
    last_step = None
    workflow = {}
    for j in range(step_count):
        task_count = random.randint(3, 4)
        step = []
        for x in range(task_count):
            task = loader.convert_task(tasks[pos])
            pos += 1
            if last_step is not None:
                task['data_deps'] = {
                    other_task['id']: other_task['cores']
                    for other_task in last_step
                }
            step.append(task)
        last_step = step
        workflow.update({
            task['id']: task
            for task in step
        })
    with open('data/general-%i.json' % (i), 'w') as fp:
        json.dump(workflow, fp)
    print(workflow)
