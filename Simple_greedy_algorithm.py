# Simple Greedy Algorithm using coins
# By Yash Desai

print("This program computes the minimum number of coins to "
        "make a certain value in denominations of 1,5, and 10.")

while(True): 
    value = int(input("Please enter a monetary value (greater than 0)"
                      ": "))

    tempvalue = value

    num_of_coins = 0

    while(value >= 10):
        value -= 10
        num_of_coins += 1

    while(value >= 5):
        value -= 5
        num_of_coins += 1

    while(value >= 1):
        value -= 1
        num_of_coins += 1

    print("The minimum number of coins required to represent %s is"
            " %s\n" % (tempvalue,num_of_coins))
