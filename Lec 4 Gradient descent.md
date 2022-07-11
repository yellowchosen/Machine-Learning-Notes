# 1.0 Gradient Descent
It is a good method to find the values of $w$ and $b$ when cost function under minimum(the smallest possible cost).
## 1.1 The general logic of GD in linear regression
We can randomly set up a combination of parameters, e.g. start with $$(w, b) = (0, 0)$$ or $$(w_1, w_2, ... , w_n, b) = (0, 0, ... , 0, 0)$$ and then keep changing $w, b$ to reduce $J(w, b)$ Until we settle at or near a minimum(local minimum). But we can not determine that this local minimum be equal to global minimum. Because there will be totally different results(local minimum) when we choose different start points(parameters). And imagine it is a downhill and we would like walk to the lowest position, so that every single time we just move a tiny baby little step to direction we decide and look around new direction and repeat, till we are arrived the lowest position(local minimum)!

<img width="809" alt="Screen Shot 2022-07-09 at 3 42 30 PM" src="https://user-images.githubusercontent.com/99445916/178120433-dd1c8b9b-a406-4d83-87d8-fd6ffcac4c76.png">

## 1.2 Implementing GD algorithm
**Formula:**

$$ w = w - \alpha\frac{d}{dw}J(w, b)$$ 

* $\alpha$ is learning rate which control how big of a step take downhill and between 0 and 1. So that if $\alpha$ is very large, then a very aggressive gradient descent procedure where trying to take "huge" steps downhill. **Remind we will learn how to choose great learning rate**.
* Derivative term telling us in which direction we want to take our baby step.

Remember we have another parameter $b$, 

$$ b = b - \alpha\frac{d}{db}J(w, b)$$

Repeat these two update steps until algorithm convergence which means reaching the point at a local minimum where the parameters $w$ and $b$ no longer change much with each additional step that we take. 

**Remember Simultaneous Update**

<img width="816" alt="Screen Shot 2022-07-09 at 5 19 00 PM" src="https://user-images.githubusercontent.com/99445916/178122996-c7c66e63-b1bc-4a51-aa8c-8647fbe97448.png">

## 1.3 Intuition of GD

$w = w - \alpha\frac{d}{dw}J(w)$ , suppose red point is starting polsition, the slope of red point is the derivative function, and from plot we know it is positive. Based on GD fuormula, if the value of derivative function is positive, then $w = w - \alpha*(positive\  number)$, so that w is going to decreased.

<img width="336" alt="Screen Shot 2022-07-09 at 5 41 00 PM" src="https://user-images.githubusercontent.com/99445916/178123553-a4f0e823-a6a4-49d7-bba6-57eed77d455d.png">

On the contrary, negative slope(negative value of derivative function) will lead the increased $w$.

<img width="339" alt="Screen Shot 2022-07-09 at 5 53 57 PM" src="https://user-images.githubusercontent.com/99445916/178123837-a58eab9e-9463-4620-82aa-ab92dbd4a62e.png">

By the way, due to the starting point $w$ is random, if it is luckily at the minimum value(slope or derivative value = 0), then the GD algorithm will do nothing! $$w = w - \alpha * 0$$

## 1.4 Learning rate
* If $\alpha$ is too small, then gradient descents will work, but it will be slow("too tiny step")  
* If $\alpha$ is too large, then may take a huge step going from original point("overshoot, never reach minimum" / "fail to converge, diverge")


<img width="313" alt="Screen Shot 2022-07-09 at 9 14 50 PM" src="https://user-images.githubusercontent.com/99445916/178127617-163d454c-6065-4a38-866d-096adb35bfb7.png">


And just recap, because the slope of $J(w)$ is always changing and actually going smaller, So that more and more nearer a local minimum, gradient descent will automatically take smaller step.*(Derivative becomes smaller, Update steps become smaller)*.

Linear regression model: $$ f_{w, b}(x) = wx+b $$

Cost function: $$ J(w, b) = \frac{1}{2m}\sum_{i=1}^{m} (f_{w, b}(x^{(i)}) - y^{(i)})^2 $$

Derivative functions:

$$ \frac{d}{dw}J(w, b) = \frac{1}{m}\sum_{i=1}^{m} (f_{w, b}(x^{(i)}) - y^{(i)})x^{(i)} $$

$$ \frac{d}{db}J(w, b) = \frac{1}{m}\sum_{i=1}^{m} (f_{w, b}(x^{(i)}) - y^{(i)}) $$

## 1.5 "Batch" gradient descent
"Batch": Each step of gradient descent uses all the training examples.

# 2.0 Appendix

This is the Procedure of implementing the gradient descent under linear regression. <br>
Import all necessary packages

```python
import math, copy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
```

Suppose our dataset only have two observations(size of house) and two outcomes(price of house)
```python
# Load our data set
x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value
```

Compute Cost
```python
#Function to calculate the cost
def compute_cost(x, y, w, b):
   
    m = x.shape[0] 
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost
```

Recall the GD algorithm we already stated

```python
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db
```

Visualize our functions by inputing dataset

```python
plt_gradients(x_train,y_train, compute_cost, compute_gradient)
plt.show()
```
![image](https://user-images.githubusercontent.com/99445916/178178221-2385c87e-6ad0-4d56-a76c-57bdf73c760e.png)

![image](https://user-images.githubusercontent.com/99445916/178178059-05757f05-0a05-43ed-aec2-b22c75c666dd.png)

```python
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
    w = copy.deepcopy(w_in) # avoid modifying global w_in
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw   
        
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history #return w and J,w history for graphing
```

Above function will give the optimal w and b which will minimize the cost function values.



```python
# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
```

Remember, the successful gradient descent algorithm should have decreasing cost value by each iteration runs. And change in cost is so rapid initially, it is useful to plot the initial descent on a different scale than the final descent.  In the plots below, note the scale of cost on the axes and the iteration step.

```python
# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()
```
![image](https://user-images.githubusercontent.com/99445916/178180486-aef63080-48ea-4557-aa72-8c66c3b02591.png)

More accurately explain, show the progress of gradient descent during its execution by plotting the cost over iterations on a contour plot of the cost(w,b).
```python
fig, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_hist, ax)
```
![image](https://user-images.githubusercontent.com/99445916/178180994-2859c5da-5a9f-45f5-b6c3-056ab9c9bd2e.png)


# 2.0 Critical thinking
What will happen if increased learning rate$(\alpha)$
![image](https://user-images.githubusercontent.com/99445916/178181144-86ad54fb-9774-4e93-89ca-fbe7c9656303.png)
```python
# initialize parameters
w_init = 0
b_init = 0
# set alpha to a large value
iterations = 10
tmp_alpha = 8.0e-1
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)
```
Above, $w$ and $b$ are bouncing back and forth between positive and negative with the absolute value increasing with each iteration.<br>
Further, each iteration $\frac{dJ(w, b)}{dw}$ changes sign and cost is increasing rather than decreasing. **This is a clear sign that the learning rate is too large and the solution is diverging.** Let's visualize this with a plot.
```python
plt_divergence(p_hist, J_hist,x_train, y_train)
plt.show()
```

![image](https://user-images.githubusercontent.com/99445916/178181212-2c49034c-caac-46ae-9233-55daca6eae76.png)



Above, the left graph shows  $w$'s progression over the first few steps of gradient descent.  𝑤  oscillates from positive to negative and cost grows rapidly. Gradient Descent is operating on both $w$ and $b$ simultaneously, so one needs the 3-D plot on the right for the complete picture.
