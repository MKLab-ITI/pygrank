from matplotlib import pyplot as plt
import pygrank as pg
tuned = [1]+[0.0, 0.7573524630180889, 0.0, 1.0, 0.004950495049504955, 0.4975492598764827, 0.2573767277717871, 0.0, 0.2549259876482698, 0.009851975296539583, 0.5, 0.5, 0.5, 0.2549259876482698, 0.5, 0.5, 0.5, 0.009851975296539583, 0.009851975296539583, 0.25002450740123516, 0.2524752475247525, 0.0, 0.5, 0.0, 0.009851975296539583, 0.0, 0.004950495049504955, 0.0, 0.5, 0.004950495049504955, 0.7475247524752475, 0.004950495049504955, 0.009851975296539583, 0.0, 0.995049504950495, 0.0, 0.5, 0.0, 0.004950495049504955, 0.0]
ppr99 = [0.99**i for i in range(len(tuned))]
ppr9 = [0.9**i for i in range(len(tuned))]
ppr85 = [0.85**i for i in range(len(tuned))]
ppr5 = [0.5**i for i in range(len(tuned))]
hk = pg.HeatKernel(3)
hk.convergence.start()
hk3 = [None]
for _ in range(len(tuned)):
    hk3.append(hk._coefficient(hk3[-1]))
    hk.convergence.has_converged(0)
hk3 = [v/max(hk3[1:]) for v in hk3[1:]]
hk = pg.HeatKernel(7)
hk.convergence.start()
hk7 = [None]
for _ in range(len(tuned)):
    hk7.append(hk._coefficient(hk7[-1]))
    hk.convergence.has_converged(0)
hk7 = [v/max(hk7[1:]) for v in hk7[1:]]


plt.plot(list(range(len(tuned))), ppr9, label='select (ppr0.9)')
#plt.plot(list(range(len(tuned))), ppr99, label='ppr0.99')
plt.plot(list(range(len(tuned))), hk7, label='hk7')
plt.plot(list(range(len(tuned))), tuned, label='tuned')
#plt.xticks(list(range(len(tuned))))
plt.xticks(None)
plt.legend()
plt.rcParams['text.usetex'] = True
plt.xlabel("i")
plt.ylabel("$h_i$")
plt.show()
