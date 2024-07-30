import sys
import time
import timeit
import multiprocessing
import random
import numpy as np
import matplotlib.pyplot as plt

class Cell:
    def __init__(self, vertex):
        self.vertex = vertex
        self.next = None
        self.previous = None

class LinkedListStack:
    def __init__(self):
        self.stack = {}
        self.size = 0

    def top(self, header):
        """ Returns the last cell added to a list at given header.

        :param header: Indicates which list on the stack should be used -> type=mutable
        :return: The last cell in a list at given header -> type=Cell
        """

        # Traverse list until their is no next cell
        last_cell = self.stack[header]
        while last_cell.next:
            last_cell = last_cell.next

        return last_cell

    def push(self, header, vertex):
        """ Creates a new cell and adds it to the end of the list at given header.

        :param header: Indicates which list on the stack should be used -> type=mutable
        :param vertex: A unique vertex id -> type=int
        :return: None
        """

        temp_cell = Cell(vertex)
        # If header specified is not in our stack then create a new list at that header else add new cell to existing list
        if header not in self.stack.keys():
            self.stack[header] = temp_cell
        else:
            last_cell = self.top(header)
            last_cell.next = temp_cell
            temp_cell.previous = last_cell

        self.size += 1

        return None

    def pop(self, header):
        """ Removes the last cell added to a list at given header and returns it.

        :param header: Indicates which list on the stack should be used -> type=mutable
        :return: The cell that was removed -> type=Cell
        """

        last_cell = self.top(header)
        # If last cell is not the only one left remove it from list else remove the whole header
        if last_cell.previous:
            last_cell.previous.next = None
            last_cell.previous = None
        else:
            del self.stack[header]

        self.size -= 1

        return last_cell

    def find(self, vertex):
        """ Locates a cell with a specified vertex id.

        :param vertex: The vertex id to be located
        :return: If the Cell is found returns that Cell and the header that its under else returns None -> type=Cell,mutable
        """

        # Go through all headers in the stack and traverse lists until vertex id is found
        for header in self.stack:
            temp_cell = self.stack[header]
            if temp_cell.vertex == vertex:
                return temp_cell, header
            while temp_cell.next:
                temp_cell = temp_cell.next
                if temp_cell.vertex == vertex:
                    return temp_cell, header

        return None

    def max_header(self):
        """ Finds the header with the highest value.

        :return: Header with the highest value -> type=mutable
        """

        # If their are more than two headers in stack return the max value header else return the only header
        if len(self.stack) > 1:
            return max(self.stack)
        else:
            return list(self.stack)[0]

    def print(self):
        """ Helper function, prints a visual representation of the stack and its lists."""

        for header in self.stack:
            out = str(header) + ':'
            head = self.stack[header]
            out += head.vertex
            while head.next:
                head = head.next
                out += " -> " + head.vertex
            print(out)

    def isEmpty(self):
        """ Checks if a stack is empty.

        :return: True if empty, False if not -> type=bool
        """

        if self.stack:
            return False

        return True

    def remove(self, vertex):
        """ Removes a cell from any list on the stack.

        :param vertex: The vertex id to be removed
        :return: The cell that was removed -> type=Cell
        """

        temp = self.find(vertex)
        assert temp
        temp_cell, header = temp

        if not temp_cell.previous:  # If cell is the first on a list
            if not temp_cell.next:  # If cell is the last on the list
                del self.stack[header]
            else:
                self.stack[header] = temp_cell.next
                temp_cell.next.previous = None
                temp_cell.next = None
        else:
            temp_cell.previous.next = temp_cell.next
            if temp_cell.next:  # If cell is not the last on the list
                temp_cell.next.previous = temp_cell.previous
            temp_cell.previous = None
            temp_cell.next = None

        self.size -= 1

        return temp_cell


def parse(filename):
    """ Extracts graph data from text file.

    :param filename: The name of the input text file to be parsed -> type=string
    :return: Graph data -> type=dict(tup(None,list(str))) [Note: None is to be replaced by partition]
    """

    file = open(filename)
    lines = file.read().split('\n')[:-1]  # Separate lines and remove last empty line

    data = {}
    for line in lines:
        line_data = ' '.join(line.split()).split(' ')  # Removes repeated spaces in string and splits the data on empty space
        vertex_id, _, _ = line_data[:3]  # Select first non-list data (coordinates not needed for this assignment)
        data[vertex_id] = None, line_data[3:]  # Get connecting vertices as list

    return data


def init_partitions(data):
    """ Sets the initial partitions for all vertices.

    :param data: Graph data -> type=dict(tup(None,list(str)))
    :return: Partition updated graph data -> type=dict(tup(int,list(str)))
    """

    # Sets partitions randomly, making sure both partitions have an equal amount of vertices
    partitions = [1] * int(len(data) / 2) + [2] * int(len(data) / 2)
    random.shuffle(partitions)
    for idx, vertex in enumerate(data):
        data[vertex] = partitions[idx], data[vertex][1]

    # Uncomment for alternate partition
    """
    isPartition1 = True
    for vertex in data:
        if isPartition1:
            data[vertex] = 1, data[vertex][1]
        else:
            data[vertex] = 2, data[vertex][1]
        isPartition1 = not isPartition1"""

    return data


def init_buckets(data):
    """ Creates the buckets for the two partitions.

    :param data: The partitioned graph data -> type=dict(tup(int,list(str)))
    :return: The two buckets for each partition with vertices allocated as Cells -> type=tup(LinkedListStack,LinkedListStack)
    """

    bucket1 = LinkedListStack()
    bucket2 = LinkedListStack()
    for vertex in data:
        # Calculate the vertex gain
        gain = 0
        for neighbour in data[vertex][1]:
            # If a vertex is in the same partition as its neighbour, increment its gain else decrement its gain
            if data[neighbour][0] == data[vertex][0]:
                gain -= 1
            else:
                gain += 1

        # Add vertex to linked list corresponding to its partition where header=gain
        if data[vertex][0] == 1:
            bucket1.push(header=gain, vertex=vertex)
        elif data[vertex][0] == 2:
            bucket2.push(header=gain, vertex=vertex)

    return bucket1, bucket2


def swap_partitions(data, vertex):
    """ Swaps the partition of a vertex to the opposite one.

    :param data: The partitioned graph data -> type=dict(tup(int,list(str)))
    :param vertex: The vertex id to be swapped
    """

    if data[vertex][0] == 1:
        data[vertex] = 2, data[vertex][1]
    else:
        data[vertex] = 1, data[vertex][1]

    return data


def swap_max(data, from_bucket):
    """ Finds the vertex with the highest gain, swaps its partition and removes it from the bucket.

    :param data: The partitioned graph data -> type=dict(tup(int,list(str)))
    :param from_bucket: The bucket from which the highest gain vertex needs to be selected -> type=LinkedListStack
    :return: The highest gain vertex that was removed and its gain -> type=tup(Cell,int)
    """

    max_gain = from_bucket.max_header()  # Finds the highest gain header in a bucket
    max_gain_node = from_bucket.pop(max_gain)  # Pops the highest gain vertex in highest gain header
    swap_partitions(data, max_gain_node.vertex)  # Swaps the partition of the highest gain vertex

    return max_gain_node, max_gain


def count_cuts(data):
    """ Counts the number of cuts between the two partitions.

    :param data: The partitioned graph data -> type=dict(tup(int,list(str)))
    :return: The number of cuts
    """

    num_cuts = 0
    # Traverse data and increment counter when two connecting nodes are in different partitions
    for node in data:
        for neighbour in data[node][1]:
            if data[node][0] != data[neighbour][0]:
                num_cuts += 1
    assert num_cuts % 2 == 0

    return int(num_cuts / 2)  # Need to divide by 2 to remove duplicates (for any A-B there is A->B and B->A)


def make_pass(data, bucket1, bucket2):
    """ Runs a full pass for all vertices.

    :param data: The partitioned graph data -> type=dict(tup(int,list(str)))
    :param bucket1: First partition bucket -> type=tup(LinkedListStack,LinkedListStack)
    :param bucket2: Second partition bucket -> type=tup(LinkedListStack,LinkedListStack)
    :return: History of cut number and last swapped vertex -> type=list(tup(int,int))
    """

    cuts_history = []
    locked = []
    num_cuts = count_cuts(data)
    is_bucket1 = True

    # Repeat until all vertices are explored (ie both buckets are empty)
    while not bucket1.isEmpty() or not bucket2.isEmpty():

        # Get max gain node, swap and lock it
        if is_bucket1:
            max_gain_node, max_gain = swap_max(data, from_bucket=bucket1)
        else:
            max_gain_node, max_gain = swap_max(data, from_bucket=bucket2)
        is_bucket1 = not is_bucket1

        locked.append(max_gain_node.vertex)
        num_cuts -= max_gain
        cuts_history.append((num_cuts, max_gain_node.vertex))

        # Update new gains for nodes neighbours
        for max_node_neighbour in data[max_gain_node.vertex][1]:
            if max_node_neighbour in locked:  # If the node is locked ignore it
                continue
            gain = 0
            for neighbour in data[max_node_neighbour][1]:
                if data[max_node_neighbour][0] == data[neighbour][0]:
                    gain -= 1
                else:
                    gain += 1

            # Update on which list the vertices are on based on their new gain
            if data[max_node_neighbour][0] == 1:
                bucket1.remove(max_node_neighbour)
                bucket1.push(gain, max_node_neighbour)
            elif data[max_node_neighbour][0] == 2:
                bucket2.remove(max_node_neighbour)
                bucket2.push(gain, max_node_neighbour)

    return cuts_history


def backtrack(data, cuts_history):
    """ Undos swaps to find optimum graph data with minimum number of cuts.

    :param data: The partitioned graph data -> type=dict(tup(int,list(str)))
    :param cuts_history: History of cut number and last swapped vertex -> type=list(tup(int,int))
    :return: Updated graph data for global cut optimum-> type=dict(tup(int,list(str)))
    """
    best_data = None
    best_cuts = 999999
    for _ in range(len(data)):
        cuts, last_node = cuts_history.pop()
        data = swap_partitions(data, last_node)
        partitions = [data[node][0] for node in data]
        if cuts <= best_cuts and partitions.count(1) == partitions.count(2):
            best_cuts = cuts
            best_data = data.copy()

    return best_data


def generate_data_string(data):
    """ Creates a string out of the graph data.

    :param data: The partitioned graph data -> type=dict(tup(int,list(str)))
    :return: A string representation of the partitioned graph data -> type=str
    """

    res = ""
    for node in data:
        if data[node][0] == 1:
            res += '0'
        else:
            res += '1'

    return res


def fm_pass(data, partitioned):
    """ Runs the graph partitoning algorithm.

    :param data: The graph data -> type=dict(tup(None,list(str)))
    :param partitioned: Flag that indicates if data has already been partitioned -> type=bool
    :return: Updated graph data for global cut optimum-> type=dict(tup(int,list(str)))
    """

    if not partitioned:
        data = init_partitions(data)
    bucket1, bucket2 = init_buckets(data)
    cut_history = make_pass(data, bucket1, bucket2)
    data = backtrack(data, cut_history.copy())

    return data, cut_history


def apply_ils(data, mutation_size, starting_passes, max_passes=None, time_limit=None):
    """ Applies iterative local search to the current optimum graph data.

    :param data: Current optimum graph data -> type=dict(tup(int,list(str)))
    :param mutation_size: The number of bits to be mutated -> type=int
    :return: The new optimum graph data after ILS is applied -> type=dict(tup(int,list(str)))
    """
    start_time = time.time()
    fm_passes = starting_passes
    count_same_optimum = 0
    cuts = []
    while True:
        mutated_data = data.copy()  # Copy graph data so we don't overwrite it
        for mutation_idx in random.sample(data.keys(), k=mutation_size):  # Mutate data
            mutated_data = swap_partitions(mutated_data, mutation_idx)
        mutated_data, passes = converge_fm(mutated_data)  # Apply local search
        if count_cuts(mutated_data) == count_cuts(data):
            count_same_optimum += 1
        if count_cuts(mutated_data) < count_cuts(data):
            data = mutated_data.copy()
        cuts.append(count_cuts(data))
        fm_passes += passes
        if fm_passes%1000==0:
            print(fm_passes)
        #if fm_passes >= max_passes:
        if time.time() - start_time >= time_limit:
            print("Time:", time.time() - start_time)
            return data, cuts[-1], count_same_optimum


def converge_fm(data):

    data, _ = fm_pass(data, partitioned=False)
    best_cuts = count_cuts(data)
    passes = 1
    while True:
        best_data = data.copy()
        data, _ = fm_pass(data, partitioned=True)
        if count_cuts(data) >= best_cuts:
            return best_data, passes
        best_cuts = count_cuts(data)
        passes += 1


def main():

    start = time.time()
    filename = "Graph500.txt"
    #filename = "test.txt"
    #filename = "GraphGenerated.txt"
    data = parse(filename)

    multiprocessing_on = True
    algorithm = "ILS"
    max_passes = 10000

    if algorithm == "FM":
        best_data, passes = converge_fm(data)
        print(count_cuts(best_data))
        # plt.plot(range(len(results)), results)
        # plt.show()
    elif algorithm == "MLS":
        pass
    elif algorithm == "ILS":
        runs = 1
        TIMES_1 = [596.6834070682526, 596.4903457164764, 596.8875207901001, 596.6023685932159, 597.4418828487396, 601.8175735473633, 599.5397260189056, 606.7463183403015, 603.4856634140015, 599.461761713028, 598.4186115264893, 597.1144876480103, 596.8396728038788, 596.7467710971832, 598.7275643348694, 604.6177606582642, 602.6691942214966, 599.6753785610199, 597.9340145587921, 596.9101750850677, 598.0853025913239, 598.9581625461578, 598.0708887577057, 596.6155617237091, 597.1930103302002]
        runs_results = []

        if multiprocessing_on:
            for _ in range(runs):
                TIMES = TIMES_5
                if len(TIMES) >= multiprocessing.cpu_count():
                    sys.exit()
                num_of_cores = len(TIMES)
                local_optimum_data, starting_passes = converge_fm(data)
                inputs = [[local_optimum_data, 10, starting_passes, None, time_limit] for time_limit in TIMES]

                if __name__ == "__main__":  # Safeguard so multiprocessing doesn't go into an infinite loop
                    pool = multiprocessing.Pool(num_of_cores)
                    ils_data, best_cut, count_same_opt = zip(*pool.starmap(apply_ils, inputs))
                    pool.close()
                    pool.join()
                    print(list(zip(best_cut, count_same_opt)))
                    runs_results.append(list(zip(best_cut, count_same_opt)))
        else:
            for time_limit in TIMES:
                local_best_data, passes = converge_fm(data)
                ils_data, best_cut, count_same_opt = apply_ils(local_best_data,
                                                               mutation_size=10,
                                                               starting_passes=passes,
                                                               time_limit=time_limit)
                runs_results.append((best_cut, count_same_opt))
                break


        for run in runs_results:
            print(run)

        #plt.plot(range(1, max_passes), cuts)
        #plt.show()
        #print(runs_results)
    elif algorithm == "GLS":
        pass
    else:
        sys.exit()

    """
    if not multiprocessing_on:
        pass
    else:
        num_of_cores = multiprocessing.cpu_count() - 1
        if __name__ == "__main__":  # Safeguard so multiprocessing doesn't go into an infinite loop
            pool = multiprocessing.Pool(num_of_cores)
            result = pool.map(run, [data] * passes)
            pool.close()
            pool.join()
            #print(result)
        print("Cores", num_of_cores)"""

    #plt.plot(range(len(results)), results)
    #plt.show()
    #print(results)
    print("Time:", time.time() - start)


main()
