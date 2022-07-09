# Gradient Descent
It is a good method to find the values of $w$ and $b$ when cost function under minimum(the smallest possible cost).
## The general logic of GD in linear regression
We can randomly set up a combination of parameters, e.g. start with $$(w, b) = (0, 0)$$ or $$(w_1, w_2, ... , w_n, b) = (0, 0, ... , 0, 0)$$ and then keep changing $w, b$ to reduce $J(w, b)$ Until we settle at or near a minimum(local minimum). But we can not determine that this local minimum be equal to global minimum. Because there will be totally different results(local minimum) when we choose different start points(parameters). And imagine it is a downhill and we would like walk to the lowest position, so that every single time we just move a tiny baby little step to direction we decide and look around new direction and repeat, till we are arrived the lowest position(local minimum)!

<img width="809" alt="Screen Shot 2022-07-09 at 3 42 30 PM" src="https://user-images.githubusercontent.com/99445916/178120433-dd1c8b9b-a406-4d83-87d8-fd6ffcac4c76.png">

## Implementing GD algorithm
**Formula:**

$$ w = w - \alpha\frac{d}{dw}J(w, b)$$
* $\alpha$ is learning rate which control how big of a step take downhill and between 0 and 1. So that if $\alpha$ is very large, then a very aggressive gradient descent procedure where trying to take "huge" steps downhill. **Remind we will learn how to choose great learning rate**.
* Derivative term telling us in which direction we want to take our baby step.

Remember we have another parameter $b$, 

$$ b = b - \alpha\frac{d}{db}J(w, b)$$

Repeat these two update steps until algorithm convergence which means reaching the point at a local minimum where the parameters $w$ and $b$ no longer change much with each additional step that we take. 

**Remember Simultaneous Update**
