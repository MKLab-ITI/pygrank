param01 = [1]+[0.0, 0.0, 0.0, 0.004950495049504955, 0.2524752475247525, 0.0, 0.2524752475247525, 0.004950495049504955, 0.49267204438314627, 0.0, 0.0, 0.2524752475247525, 0.2524752475247525, 0.995049504950495, 0.0, 0.995049504950495, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.995049504950495, 0.0, 0.995049504950495, 0.004950495049504955, 0.7475247524752475, 0.004950495049504955, 0.7475247524752475, 0.0, 0.995049504950495, 0.0, 0.5, 0.2549259876482698, 0.5, 0.5, 0.5, 0.5, 0.0]
param02 = [1]+[0.0, 0.004950495049504955, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.007401235173022269, 0.5, 0.0, 0.5, 0.995049504950495, 0.995049504950495, 0.5, 0.7450740123517302, 0.5, 0.5, 0.5, 0.5, 0.2524752475247525, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.004950495049504955, 0.7475247524752475, 0.0]
param12 = [1]+[0.0, 0.5336409726464324, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
param11 = [1]+[0.0, 0.25002450740123516, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2426475369819111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.995049504950495, 0.0, 0.0, 0.5, 0.7475247524752475, 0.5, 0.5, 0.5, 0.2524752475247525, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0]

from matplotlib import pyplot as plt
plt.subplot(1,4,1)
plt.plot(list(range(len(param01))), param01, 'b', label='citeseer0 10%')
plt.title('citeseer0 10%')
plt.ylabel("$h_i$")
#plt.legend()
plt.subplot(1,4,2)
plt.plot(list(range(len(param01))), param02, 'g', label='citeseer0 20%')
#plt.legend()
plt.title('citeseer0 20%')
plt.subplot(1,4,3)
plt.plot(list(range(len(param01))), param11, 'r', label='citeseer1 10%')
plt.title('citeseer1 10%')
plt.subplot(1,4,4)
plt.plot(list(range(len(param01))), param12, 'orange', label='citeseer1 20%')
plt.title('citeseer1 20%')
#plt.xticks(list(range(len(param01))))
#plt.legend()
#plt.rcParams['text.usetex'] = True
#plt.xlabel("i")
plt.show()
