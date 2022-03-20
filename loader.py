# see https://www.cs.huji.ac.il/labs/parallel/workload/swf.html
SWF_FIELDS = [
    'job_number',
    'submit_time',
    'wait_time',
    'run_time',
    'number_of_allocated_processors',
    'average_cpu_time_used',
    'used_memory',
    'requested_number_of_processors',
    'requested_time',
    'requested_memory',
    'status',
    'user_id',
    'group_id',
    'executable_application_number',
    'queue_number',
    'partition_number',
    'preceding_job_number',
    'think_time_from_preceding_job',
]

def swf(fname):
    """Load a Standard Workflow Format (SWF) trace (https://www.cs.huji.ac.il/labs/parallel/workload/swf.html)."""
    tasks = []
    with open(fname) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';'):
                continue
            line = line.split()
            if len(line) < len(SWF_FIELDS):
                continue
            task = {key: int(line[i]) for i, key in enumerate(SWF_FIELDS)}
            tasks.append(task)
    return tasks


def convert_task(task):
    """Convert an SWF task into task that can be scheduled."""
    return {
        'id': 'task-%i' % (task['job_number'],),
        'runtime': task['run_time'],
        'cores': task['requested_number_of_processors'],
        # map from task ID to data requirements (an integer value)
        'data_deps': {},
    }
