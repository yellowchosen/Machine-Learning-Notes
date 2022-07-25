# Feature and parameter value
First of all, let's build a  typical multivariable regression model.
$$Price = w_1x_1+w_2x_2+b$$
* $x_1: size(feet^2)$, range 300 - 2000
* $x_2: bedrooms$, range 0 - 5

So in this example, $x_1$ takes on a relatively large range of values and $x_2$ takes on a relatively small range of values.

## Particular example
House: $x_1$  = 2000, $x_2$ = 5, price = 500k , So the question is what reasonable values for the size of parameters $w_1$ and $w_2$?
<img width="659" alt="Screen Shot 2022-07-24 at 11 06 55 PM" src="https://user-images.githubusercontent.com/99445916/180690874-51c1dd0b-2d45-4e67-8ab0-18c045265fa4.png">

<img width="673" alt="Screen Shot 2022-07-24 at 11 20 32 PM" src="https://user-images.githubusercontent.com/99445916/180692789-0817b6b5-9acb-488b-8f1b-708a0f9b68f0.png">

# How to set up the scaling 
We usually have many methods to doing the features scaling
## Dividing by the maximum

<img width="658" alt="Screen Shot 2022-07-24 at 11 37 51 PM" src="https://user-images.githubusercontent.com/99445916/180694282-4750e6ec-1c7c-4163-a042-2e45a4d90652.png">

## Mean normalization
$x_1 = \frac{x_1 - \mu_1}{max - min} = \frac{x_1 - \mu_1}{2000 - 300}$
