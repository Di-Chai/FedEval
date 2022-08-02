from multiprocessing import Pool, Manager
import os
from functools import reduce


def handle_error(error):
    print('#############################')
    print('Error:', error)
    print('#############################')


# (my_rank, n_jobs, dataList, resultHandleFunction, parameterList)
def multiple_process(
        distribute_list, partition_data_func, task_function, n_jobs,
        reduce_function, parameter_list
    ):
    """

    :rtype:
    """
    if callable(partition_data_func) and callable(task_function) and callable(reduce_function):
        print('Parent process %s.' % os.getpid())

        manager = Manager()
        ShareQueue = manager.Queue()
        Locker = manager.Lock()

        p = Pool()
        for i in range(n_jobs):
            p.apply_async(task_function, args=(ShareQueue, Locker, partition_data_func(distribute_list, i, n_jobs),
                                               [i] + parameter_list,),
                          error_callback=handle_error)
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')

        result_list = []
        while not ShareQueue.empty():
            result_list.append(ShareQueue.get_nowait())

        return reduce(reduce_function, result_list)
    else:
        print('Parameter error')


# Example
def task(share_queue, locker, data, parameter_list):

    result = sum(data)

    locker.acquire()
    share_queue.put(result)
    locker.release()


if __name__ == "__main__":
    data = [e for e in range(100)]

    n_job = 4

    partition_func = lambda data, i, n_job: [data[e] for e in range(len(data)) if e % n_job == i]

    print(multiple_process(
        distribute_list=data, partition_data_func=partition_func, task_function=task,
        n_jobs=n_job, reduce_function=lambda x, y: x+y, parameter_list=[]
    ))


