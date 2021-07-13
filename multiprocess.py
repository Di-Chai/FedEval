from multiprocessing import Pool
from time import sleep
from typing import Mapping
import os


def print_info(a, b, c):
    print(os.getpid(), a)
    sleep(10)


if __name__ == '__main__':

    p = Pool(15)
    
    p.map(print_info, [e for e in range(15)])

    print('Waiting for all subprocesses done...')
    p.close()
    p.join()

# from multiprocessing import Pool, TimeoutError
# import time
# import os

# def f(x):
#     print('process', os.getpid(), x)
#     time.sleep(5)
#     return x*x

# if __name__ == '__main__':
#     # start 4 worker processes
#     with Pool(processes=100) as pool:

#         # print "[0, 1, 4,..., 81]"
#         print(pool.map(f, range(100)))

#         # print same numbers in arbitrary order
#         # for i in pool.imap_unordered(f, range(10)):
#         #     print(i)

#         # evaluate "f(20)" asynchronously
#         res = pool.apply_async(f, (20,))      # runs in *only* one process
#         print(res.get(timeout=1))             # prints "400"

#         # evaluate "os.getpid()" asynchronously
#         res = pool.apply_async(os.getpid, ()) # runs in *only* one process
#         print(res.get(timeout=1))             # prints the PID of that process

#         # launching multiple evaluations asynchronously *may* use more processes
#         multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
#         print([res.get(timeout=1) for res in multiple_results])

#         # make a single worker sleep for 10 secs
#         res = pool.apply_async(time.sleep, (10,))
#         try:
#             print(res.get(timeout=1))
#         except TimeoutError:
#             print("We lacked patience and got a multiprocessing.TimeoutError")

#         print("For the moment, the pool remains available for more work")

#     # exiting the 'with'-block has stopped the pool
#     print("Now the pool is closed and no longer available")