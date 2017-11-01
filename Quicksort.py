# Quicksort
# By Yash Desai
# This program uses the quicksort algorithm to sort an array.
# Made with the help of the pseudo-code available on wikipedia.


# The function quicksort sorts a whole list, the helper function
# identifies the begining and end of the list (which are needeed
# for computations) without needing the user to input the extra
# parameters.
def quicksort(A):
    quicksorthelper(A,0,len(A)-1)

def quicksorthelper(A,first,last):
    if first < last:
        p = partition(A,first,last)
        quicksorthelper(A, first, p-1)
        quicksorthelper(A, p + 1, last)

# The partition function goes through the whole list (excluding
# the pivot), and pushes elements greater than the pivot to the
# right, and less than the pivot to the left. This is done by
# having the counter i sit on elements greater than
# the pivot until counter j lands upon an element less than the
# pivot. Then a swap is made between the elements of the two
# counters. Refer to the wikipedia page for a full explanation.
def partition(A,first,last):
    pivot = A[last]
    i = first - 1
    for j in range(first,last):
        if A[j] < pivot:
            i += 1
            A[i],A[j] = A[j],A[i]
    if A[last] < A[i+1]:
        A[i+1], A[last] = A[last], A[i+1]

    return i+1


B = [9,8,7,6,5,4,3,2,1]
print('B = %s'% (B))
quicksort(B)
print('B = %s'% (B))
