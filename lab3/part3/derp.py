output = open('graph.dat', 'w')
output.write(str(10000) + '\n')
output.write(str(10000*10000) + '\n')

for i in range(0, 10000):
  for j in range(0, 10000):
    output.write(str(i) + ' ' + str(j) + '\n')
output.close()