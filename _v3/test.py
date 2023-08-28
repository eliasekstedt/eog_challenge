
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def test0():
    seq = np.random.normal(0, 1, 9)
    seq = tuple(seq.tolist())
    out = [1*(el>0) for el in seq]
    print(out)

def test1():
    left = tuple(np.random.randint(0, 1) for _ in range(2))
    middle = tuple(np.random.randint(0, 1) for _ in range(2))
    right = tuple(np.random.randint(0, 1) for _ in range(3))
    left = left + right
    


def test2():
    pass

def test3():
    pass

def test4():
    pass

def test5():
    pass

def test6():
    pass

def test7():
    pass

def test8():
    pass


def main():
    #test8()
    #test7()
    #test6()
    #test5()
    #test4()
    #test3()
    #test2()
    test1()
    #test0()

if __name__ == '__main__':
    main()


