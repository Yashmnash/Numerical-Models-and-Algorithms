import random
import matplotlib.pyplot as plt


Yash = [0]
Difference = []


for i in range(101):

    Heads = random.randint(0,1)

    if Heads:
        Yash.append(Yash[-1] + 1)

    else:
        Yash.append(Yash[-1] - 1)
        

    if i>1000:
        Difference.append(Yash[-1] - Yash[-999])


plt.title('Yash\'s random walk')
plt.ylabel('Earnings in USD')
plt.xlabel('Flips')
plt.xlim(0,101)
plt.grid(True)

plt.plot(Yash)
plt.show()

plt.title('Yash\'s random walk (variation)')
plt.ylabel('Difference every 1000 flips')
plt.xlabel('Flips')
plt.xlim(0,1000000)
plt.ylim(-200,200)
plt.grid(True)

plt.plot(Difference)
plt.show()



    
        
