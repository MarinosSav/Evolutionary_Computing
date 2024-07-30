import random

node_num = 500
f = open("GraphGenerated.txt", "w")

nodes = {}

for node_id in range(1, node_num+1):
    potential_neighbours = list(range(1, node_num + 1))
    potential_neighbours.remove(node_id)
    num_connections = random.randint(1, node_num - 1)
    nodes[node_id] = random.sample(potential_neighbours, k=num_connections)

# Make sure vertices are bi-directional
for node_id in nodes:
    print(node_id)
    for neighbour in nodes[node_id]:
        if node_id not in nodes[neighbour]:
            temp_neighbours = nodes[neighbour]
            temp_neighbours.append(node_id)
            nodes[neighbour] = temp_neighbours

for node_id in nodes:
    out = '   ' + str(node_id) + ' (_,_)   ' + str(len(nodes[node_id])) + ' '
    for neighbour in nodes[node_id]:
        out += ' ' + str(neighbour)
    f.write(out + '\n')

f.close()
