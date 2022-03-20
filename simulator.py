import random
import json

from matplotlib import pyplot as plt
import numpy as np
import abc


class Resource(abc.ABC):

    @abc.abstractmethod
    def execute(self, tasks, time, schedule, comm_matrix):
        """Execute the tasks on this resource."""

    @abc.abstractproperty
    def metadata(self):
        """Return a set of metadata for this given resource."""


class ResourceUsage:
    """Class for keeping track of resource usage and intervals."""
    def __init__(self, total, start_time):
        # total amount of resources
        self.total = total
        # map from event time to resource usage
        self.usage = {start_time: 0}

    def insert_interval(self, t0, t1, amount):
        """Insert an interval of resource usage."""
        event_times = sorted(self.usage)
        before_t0 = None
        before_t1 = None
        for time in event_times:
            if time >= t0 and time < t1:
                self.usage[time] += amount
            if time < t0:
                before_t0 = time
            if time < t1:
                before_t1 = time
        if t0 not in self.usage:
            self.usage[t0] = (self.usage[before_t0] + amount
                              if before_t0 is not None else amount)
        if t1 not in self.usage:
            self.usage[t1] = (self.usage[before_t1]
                              if before_t1 is not None else 0)

    def find_interval(self, total_time, amount):
        """Find the first interval where we can use amount of resources for total_time."""
        assert amount <= self.total
        event_times = sorted(self.usage)
        for i, start_time in enumerate(event_times):
            if (self.total - self.usage[start_time]) < amount:
                continue
            for end_time in event_times[i+1:]:
                if (self.total - self.usage[end_time]) < amount:
                    break
                if (end_time - start_time) >= total_time:
                    return start_time
        return max(event_times)


class SequentialResource(Resource):
    """Simple resource representation with load, speed and cost."""

    def __init__(self, id_, cores, load, speed, cost_per_core_second):
        self.id_ = id_
        self.cores = cores
        self.load = load
        # computational power
        self.speed = speed
        self.cost_per_core_second = cost_per_core_second


    def execute(self, tasks, submit_time, schedule, comm_matrix):
        """Simulate submitting and then executing all these tasks at this time."""
        wait_time = 0
        info = {}
        # current resource usage info
        usage = ResourceUsage(self.cores, submit_time + wait_time)
        # curr_time = submit_time
        for task_id in tasks:
            used_cores = tasks[task_id]['cores']
            # calculate transfer time
            transfer_time = sum(
                tasks[task_id]['data_deps'][other_task_id]
                * comm_matrix[self.id_][schedule[other_task_id]]
                for other_task_id in tasks[task_id]['data_deps']
            )
            # calculate the runtime based on the "speed" of this resource
            real_runtime = transfer_time + self.speed * tasks[task_id]['runtime']
            # calculate the earliest start time possible
            start_time = usage.find_interval(real_runtime, used_cores)
            # insert the resource usage interval
            usage.insert_interval(start_time, start_time + real_runtime, used_cores)
            info[task_id] =  {
                'resource_id': self.id_,
                'submit_time': submit_time,
                'finish_time': start_time + real_runtime,
                'start_time': start_time,
                'used_cores': used_cores,
                'cost': self.cost_per_core_second * real_runtime * used_cores,
            }
            used_cores += tasks[task_id]['cores']
        return info

    @property
    def metadata(self):
        """Return resource metadata."""
        return {
            'cores': self.cores,
            'load': self.load,
            'speed': self.speed,
            'cost_per_core_second': self.cost_per_core_second,
        }


class Workflow:
    """Workflow class."""

    def __init__(self, tasks): # task_sets):
        self.tasks = tasks

    def ready_tasks(self, complete_tasks):
        """Return a dict of ready tasks."""
        return {
            task_id: self.tasks[task_id]
            for task_id in self.tasks
            if all(dep_task_id in complete_tasks
                   for dep_task_id in self.tasks[task_id]['data_deps'])
               and task_id not in complete_tasks
        }


def simulation(workflow, resources, comm_matrix, scheduler):
    """Run a simulation loop."""
    # keep track of all allocations for future task scheduling
    full_schedule = {}
    # profile of task runs
    profile = {}
    resource_metadata = {res_id: resources[res_id].metadata for res_id in resources}
    curr_time = 0
    while len(workflow.tasks) > len(profile):
        tasks = workflow.ready_tasks([task_id for task_id in profile])
        # schedule the tasks
        allocations = scheduler.schedule(
            tasks,
            resource_metadata,
            comm_matrix,
            full_schedule,
        )
        if not allocations:
            # can't finish workflow
            return profile
        # "run" the tasks and save the profile results
        times = []
        for res_id in resources:
            scheduled_tasks = {task_id: tasks[task_id] for task_id in allocations
                               if allocations[task_id] == res_id}
            result = resources[res_id].execute(
                scheduled_tasks,
                curr_time,
                # note: the full schedule is required for calculating data transfers
                full_schedule,
                comm_matrix,
            )
            if result:
                profile.update(result)
                times.append(max(result[task_id]['finish_time'] - curr_time
                                 for task_id in result))
        curr_time = max(times)
        full_schedule.update(allocations)
    return profile


class FCFSScheduler:

    def schedule(self, tasks, resource_metadata, comm_matrix, full_schedule):
        """An FCFS based scheduler"""
        schedule = {}
        for task_id in tasks:
            # 1. filter
            avail = [
                res_id for res_id in resource_metadata
                if resource_metadata[res_id]['cores'] >= tasks[task_id]['cores']
            ]
            # 2. choose/schedule
            if not avail:
                # no resources available
                continue
            schedule[task_id] = avail[random.randint(0, len(avail) - 1)]
        return schedule


# See also https://en.wikipedia.org/wiki/Skyline_operator
class SamplerScheduler:
    """Sampler scheduler for optimizing both cost and makespan."""

    def __init__(self, count):
        self.count = count

    def schedule(self, tasks, resource_metadata, comm_matrix, full_schedule):
        """An optimal schedule."""
        # 1. get feasible resources for each task
        feasible = {
            task_id: [
                res_id for res_id in resource_metadata
                if resource_metadata[res_id]['cores'] >= tasks[task_id]['cores']
            ]
            for task_id in tasks
        }
        # now we compute scores for each task-resource pair
        def compute_score(task, resource_id, resource):
            """Compute the score for the task and resource (return (runtime, cost))."""
            transfer_time = sum(task['data_deps'][other_task_id]
                                * comm_matrix[resource_id][full_schedule[other_task_id]]
                                for other_task_id in task['data_deps'])
            runtime = resource['speed'] * task['runtime']
            cost = resource['cost_per_core_second'] * runtime
            return (resource['load'] + transfer_time + runtime, cost)
        # compute the scores for each task ID-resource pair
        f_scores = {
            task_id: {
                res_id: compute_score(tasks[task_id], res_id, resource_metadata[res_id])
                for res_id in feasible[task_id]
            }
            for task_id in tasks
        }

        def generate_random(feasible):
            """Generate a set of random schedules."""
            schedule = {}
            for task_id in feasible:
                i = random.randint(0, len(feasible[task_id]) - 1)
                schedule[task_id] = feasible[task_id][i]
            return schedule

        # generate a set of random schedules
        random_schedules = [generate_random(feasible) for i in range(self.count)]
        # determine which results are optimal
        longest_times = [
            max(f_scores[task_id][schedule[task_id]][0] for task_id in schedule)
            for schedule in random_schedules
        ]
        costs = [
            sum(f_scores[task_id][schedule[task_id]][1] for task_id in schedule)
            for schedule in random_schedules
        ]

        value_matrix = np.array([
            (
                max(f_scores[task_id][schedule[task_id]][0] for task_id in schedule),
                sum(f_scores[task_id][schedule[task_id]][1] for task_id in schedule),
            )
            for schedule in random_schedules
        ])
        product = np.matmul(value_matrix, value_matrix.transpose())
        lengths = product.diagonal()
        best_schedule = None
        best_length = None
        for i, length in enumerate(lengths):
            if best_schedule is None or best_length is None or length < best_length:
                best_schedule = i
                best_length = length
        return random_schedules[best_schedule]


class RandomScheduler:

    def schedule(self, tasks, resource_metadata, comm_matrix, full_schedule):
        """Produce a random schedule."""
        resources = list(resource_metadata)
        return {
            task_id: resources[random.randint(0, len(resources) - 1)]
            for task_id in tasks
        }


def simulation_test_loop(resources, comm_matrix, schedulers, workflows):
    """Run a simulation test loop for the given resource configs, schedulers and workflows."""
    results = {
        scheduler_name: {}
        for scheduler_name in schedulers
    }
    for scheduler_name in schedulers:
        scheduler = schedulers[scheduler_name]
        for fname in workflows:
            with open(fname) as fp:
                workflow = Workflow(json.load(fp))
            profile = simulation(workflow, resources, comm_matrix, scheduler)
            results[scheduler_name][fname] = profile
    return results


def graph_makespan(results):
    """Graph the results of this experiment."""
    fig, ax = plt.subplots()
    schedulers = list(results)
    width = 0.2
    workflow_count = [len(results[scheduler_name]) for scheduler_name in results][0]
    x = np.arange(workflow_count)
    rects = []
    n = len(schedulers)
    for i, scheduler_name in enumerate(schedulers):
        # calculate the makespan time
        height = [
            sum(max(profile[task_id]['finish_time'] for task_id in profile)
                - min(profile[task_id]['submit_time']
                      for task_id in profile)
                for profile in results[scheduler_name][wfname])
            / len(results[scheduler_name][wfname])
            for wfname in results[scheduler_name]
        ]
        # calculate the x positions
        rect = ax.bar(x - n * width / 2 + i * width + 0.5 * width, height, width, label=scheduler_name)
        rects.append(rect)
    ax.legend()
    ax.set_ylabel('Makespan time (s)')
    ax.set_xticks(
        x,
        ['WF{}'.format(i) for i in range(workflow_count)],
    )
    plt.show()


def graph_boxplot(results):
    """Graph a makespan boxplot."""
    fig, (ax_makespans, ax_costs)= plt.subplots(2)
    x_makespans = []
    x_costs = []
    for scheduler_name in results:
        makespans = []
        costs = []
        for fname in results[scheduler_name]:
            profile = results[scheduler_name][fname]
            makespan = (max(profile[task_id]['finish_time'] for task_id in profile)
                        - min(profile[task_id]['submit_time'] for task_id in profile))
            costs.append(sum(profile[task_id]['cost'] for task_id in profile))
            makespans.append(makespan)
        x_makespans.append(makespans)
        x_costs.append(costs)
    labels = [scheduler_name for scheduler_name in results]
    ax_makespans.boxplot(x_makespans, labels=labels)
    ax_makespans.set_ylabel('Makespan (s)')
    ax_costs.boxplot(x_costs, labels=labels)
    ax_costs.set_ylabel('Cost ($)')
    ax_costs.set_xlabel('Algorithm')
    plt.show()


def compute_optimal(workflows, speed, cost_per_core_second):
    """Compute optimal workflow execution/schedule."""
    profiles = {}
    for fname in workflows:
        with open(fname) as fp:
            workflow = Workflow(json.load(fp))
        profile = {}
        while len(workflow.tasks) > len(profile):
            if not profile:
                curr_time = 0
            else:
                curr_time = max(profile[task_id]['finish_time'] for task_id in profile)
            tasks = workflow.ready_tasks([task_id for task_id in profile])
            profile.update({
                task_id: {
                    'resource_id': None,
                    'submit_time': curr_time,
                    'start_time': curr_time,
                    'finish_time': curr_time + tasks[task_id]['runtime'] / speed,
                    'used_cores': tasks[task_id]['cores'],
                    'cost': (tasks[task_id]['runtime'] / speed) * tasks[task_id]['cores'] * cost_per_core_second,
                }
                for task_id in tasks
            })
        profiles[fname] = profile
    return profiles


def run(workflows, result_file):
    resources = {
        'res-0': SequentialResource('res-0', 1024, 10, 1, 0.00003),
        'res-1': SequentialResource('res-1', 65536, 30, 2, 0.0001),
        'res-2': SequentialResource('res-2', 2048, 20, 1.2, 0.00008),
        'res-3': SequentialResource('res-3', 512, 10, 1.1, 0.00001),
    }
    comm_matrix = {
        id0: {id1: 0 if id0 == id1 else random.randint(1, 10)
              for id1 in resources}
        for id0 in resources
    }
    schedulers = {
        'First Choice FCFS': FCFSScheduler(),
        'Sampler 2': SamplerScheduler(2),
        'Sampler 8': SamplerScheduler(8),
        'Sampler 16': SamplerScheduler(16),
        'Sampler 32': SamplerScheduler(32),
    }
    results = simulation_test_loop(resources, comm_matrix, schedulers, workflows)
    results['Optimal'] = compute_optimal(workflows, 2.0, 0.00001)
    with open(result_file, 'w') as fp:
        json.dump(results, fp, indent=4)
    graph_boxplot(results)


if __name__ == '__main__':
    print('linear chain workflows')
    workflows = ['data/linear-chain-%i.json' % (i,) for i in range(2**8)]
    run(workflows, 'results/linear-chain.json')
    print('general workflows')
    workflows = ['data/general-%i.json' % (i,) for i in range(2**8)]
    run(workflows, 'results/general.json')
