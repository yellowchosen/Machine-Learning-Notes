# Gradient Descent
It is a good method to find the values of $w$ and $b$ when cost function under minimum(the smallest possible cost).
## The general logic of GD in linear regression
We can randomly set up a combination of parameters, e.g. start with $$(w, b) = (0, 0)$$ or $$(w_1, w_2, ... , w_n, b) = (0, 0, ... , 0, 0)$$ and then keep changing $w, b$ to reduce $J(w, b)$ Until we settle at or near a minimum$(local minimum)$. But we can not determine that this local minimum be equal to global minimum. Because there will be totally different results(local minimum) when we choose different start points(parameters). And imagine it is a hill!
<img width="809" alt="Screen Shot 2022-07-09 at 3 42 30 PM" src="https://user-images.githubusercontent.com/99445916/178120433-dd1c8b9b-a406-4d83-87d8-fd6ffcac4c76.png">
