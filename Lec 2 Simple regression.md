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
