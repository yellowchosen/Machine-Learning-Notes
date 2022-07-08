# Cost function- definition

Recall simple linear regression model: $f_{w,b}(x) = wx + b$  
**w,b**: Parameters/Coefficients/Weights

Cost fucntion: taking estimated y($\hat{y}$) to compare with true y(target), and sum up all of errors square then find average. 

$$ J(w, b) = \frac{1}{2m}\sum\limits_{i=1}^m(\hat{y}^{(i)} - y^{(i)})^2 $$

The goal is minimizing $J(w, b)$ , BTW In statistics, we call this lease square!

<img width="797" alt="Screen Shot 2022-07-08 at 5 56 53 PM" src="https://user-images.githubusercontent.com/99445916/178075405-782e6c15-a88a-43ae-9b22-e0f75ed9921f.png">

So that the intuition of cost function is finding the suitable parameters $(w, b)$ which let cost function to be minimum!

# Appendix

```python
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
x_train = np.array([1.0, 2.0])           #(size in 1000 square feet)
y_train = np.array([300.0, 500.0])           #(price in 1000s of dollars)
```

The formula we have: $$ J(w, b) = \frac{1}{2m}\sum\limits_{i=1}^{m-1}(f_{w,b}(x^{(i)}) - y^{(i)})^2 $$ and $$ f_{w,b}(x^{(i)}) = wx^{(i)} + b $$

Then still use our for loop method to build function of computing cost:
```python
def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost
```

Finally, we learned the cost equation provides a measure of how well your predictions match your training data.
And minimizing the cost can provide optimal values of $w$ and $b$.
