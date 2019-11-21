from functools import reduce

def extend_with_return(some_list, other_list):
    some_list.extend(other_list)
    return some_list

def append_with_return(some_list, other_list):
    some_list.append(other_list)
    return some_list

def dense_to_sparse(indices, length):
    sparse = []
    indices_set = set(indices)
    for index in range(length):
        if index in indices_set:
            sparse.append(1)
        else:
            sparse.append(0)
    return sparse

def sparse_to_dense(indices):
    dense = []
    for index in range(len(indices)):
        if indices[index] != 0:
            dense.append(index)
    return dense

def make_strict(some_list):
    strict_list = []
    for elem in some_list:
        if not elem in strict_list:
            strict_list.append(elem)
    return strict_list

def print_enumeration(some_list):
    reduce(lambda x, elem: print(elem[0], elem[1]), enumerate(some_list), '')
