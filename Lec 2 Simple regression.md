# 1. Simple regression model
Very classic simple regression model, which is **house price prediction** :

<img width="496" alt="Screen Shot 2022-07-07 at 8 55 56 PM" src="https://user-images.githubusercontent.com/99445916/177895111-eefbb90c-6aa4-4624-8977-d92c301e14d3.png">

By giving a certain house size(suqare feet) to predict price($) of this house.
Notation: <br />
          *x* = "input" variable feature,<br />
          *y* = "output" variable , "target" variable,<br />
          *m* = number of training examples,<br />
          *(x, y)* = single training example,<br />
          ($x^i$, $y^i$) = $i^{th}$ training example ($1^{st}$, $2^{nd}$, $3^{rd}$, ...)<br />
## 1.1 Model representation


<img width="332" alt="Screen Shot 2022-07-07 at 9 15 46 PM" src="https://user-images.githubusercontent.com/99445916/177896993-c9815d23-315b-405f-ad80-bf2fef9c4dc8.png">

*f* represents function <br />
*y-hat* represents estimated y <br />
Suppose input x(size) into function, and then predict output y(price). This is a mapping fucntion **X -> Y**.

Linear Function(model): $f_{w,b}(x) = wx + b$ , (Univariate) Linear regression with **one** variable.



# Appendix
Using NumPy to scientific conputing, and Matplotlib for plotting data(Visualization)

```python
import numpy as np
import matplotlib.pyplot as plt

# X_train is the input variable(size in 1000 sqaure feet)
# Y_train is the target(price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
```

```python
# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")
```

```python
# m is the number of training examples
m = len(x_train)
print(f"Number of training examples is: {m}")
```

```python
i = 0 # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
```

```python
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()
```

## Model function
Remember above formula about w and b? Now let's define w = 100 and b = 100.

```python
w = 100
b = 100
print(f"w: {w}")
print(f"b: {b}")
```
Then we have two functions based on x_train values, $f_{wb} = w * x[0] + b$ and $f_{wb} = w * x[1] + b$, But how to compute larger numbers of training example? **Using for loop!**

```python
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb
```

Now call the compute_model_output function and plot the output.
```python
tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
```

<img width="404" alt="Screen Shot 2022-07-08 at 5 04 10 PM" src="https://user-images.githubusercontent.com/99445916/178070094-d90580b7-6258-4b8f-9b86-cee66af45e38.png">


We see w=100 and b=100 which do not effect our model or result in a line that fits our data.(Not a good fit) Let's adjust w=200 and b=100.

```python 
w = 200
b = 100
```

<img width="389" alt="Screen Shot 2022-07-08 at 5 11 54 PM" src="https://user-images.githubusercontent.com/99445916/178070981-0bf3801b-2c04-4d81-a6a1-a340faacd173.png">

And make a prediction by seting x = 1.2
```python
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")
```

