# Dijkstra's Algorithm
# By Yash Desai

graph = {'a':{'b':10,'c':3}, 'b':{'c':1,'d':2}, 'c':{'b':4,'d':8, 'e':2}, 'd':{'e':7}, 'e':{'d':9,}}

#def dijkstra(graph,start,end):
start = 'a'
end = 'd'

unseen = dict(graph)
distance = {}
prev = []

for i in graph:
    if i == start:
        distance[i] = 0

    else:
        distance[i] = 99999

print(distance)
j = 0
while unseen:
    j += 1
    u = str(min(distance))
    del unseen[u]

    for i in dict.keys(graph[u]):
        alt = distance[u] + graph[u][i]
        if alt < distance[i]:
            distance[i] = alt
            prev.append(i)

    #return distance[end]


dijkstra(graph,'a','d')
                
