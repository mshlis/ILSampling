## Intermediate Loss Sampling
I propose a novel approach, leveraging an intermediate loss function to differentiate through a categorical draw. There exists a long history of using policy gradient techniques where only the policy network gradients are utilized, but in the last couple of years approaches like the Gumbel Softmax has surfaced. Gumbel Softmax attempts to model categorical variables through a reparametrization trick and uses softmax to approximate the argmax operator, which in result is completely differentiable. The gumbell softmax is parametrized by a temperature hyperparameter, T. at T=0, this approximation is equivalent to a draw from the categorical distribution but the gradient is undefined. As T increases, the derivative is more defined, but the sample becomes more and more smooth. This give and take is the primary issue with this approach.   

#### construction
As a forward pass, I 

