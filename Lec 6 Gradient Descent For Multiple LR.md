<img width="802" alt="Screen Shot 2022-07-13 at 11 58 30 PM" src="https://user-images.githubusercontent.com/99445916/178895319-c9c89245-5244-4b09-96b1-7bc988dbd3ef.png">



# Cost Function
In multiple linear regression, its cost function should be very similiar with simple linear regression:
$$J(w_0, w_1,..., w_n,b) = \frac{1}{2m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)}) - y^{(i)})$$ and we define $f_{w, b}(x) = w^Tx$


# Compare these two types of gradient descent
<img width="805" alt="Screen Shot 2022-07-14 at 12 25 07 AM" src="https://user-images.githubusercontent.com/99445916/178898317-411cf3bf-0d54-41f7-804d-18c1153bc424.png">

## An alternative to gradient descent
Normal equation:
* Only for linear regression
* Solve for w, b without iterations
* But does not generalize to other learning algorithms
* And slow when number of features is large(n > 10,000)


# Appendix
```python
import copy, math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays
```
Some natations we will use in this project <br>

<img width="745" alt="Screen Shot 2022-07-15 at 10 35 28 PM" src="https://user-images.githubusercontent.com/99445916/179335820-4ca7105c-1c08-48e9-b845-7505aae3f96c.png">

```python
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
```
The dataset should seem like <br>

<img width="534" alt="Screen Shot 2022-07-15 at 10 40 03 PM" src="https://user-images.githubusercontent.com/99445916/179336076-196a5262-29df-43b8-838f-c69978227245.png">

<img width="288" alt="Screen Shot 2022-07-15 at 10 44 27 PM" src="https://user-images.githubusercontent.com/99445916/179336284-cfd41225-386a-4999-92ff-64b7ea437343.png">
* $x^{(i)}$ is vector containing example , $x^{(i)} = (x_0^{(i)}, x_1^{(i)}, ... , x_{n-1}^{(i)})$
* $x_j^{(i)}$ is element j in example, The superscript in parenthesis indicates the example number while the subscript represents an element.

print our inputing dataset
```python
# data is stored in numpy array/matrix
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)
```
 




