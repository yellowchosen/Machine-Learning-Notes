# 1.Gradient Descent
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
