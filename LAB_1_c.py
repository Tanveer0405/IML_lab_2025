# 3. WAP to take 5 different co-ordinates in loop and find total error using linear regression

# y=mx+c
import matplotlib.pyplot as plt

m = float(input("Enter value of m: "))
c = float(input("Enter value of c: "))

x_vals = []
y_vals = []

for i in range(5):
    x = float(input(f"Enter x coordinate {i+1}: "))
    y = float(input(f"Enter y coordinate {i+1}: "))
    x_vals.append(x)
    y_vals.append(y)

errors = []
y_calcs = [] 
total_error = 0

for i in range(5):
    y_calc = m * x_vals[i] + c
    y_calcs.append(y_calc) 
    error =abs(y_vals[i] - y_calc)
    errors.append(error)
    total_error = total_error + error

print("\nErrors for each point:", errors)
print("Total Error:", total_error)

for i in range(5):
    plt.plot([x_vals[i], x_vals[i]], [y_vals[i], y_calcs[i]], 'k--')

plt.scatter(x_vals, y_vals)
plt.plot(x_vals, y_calcs)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
