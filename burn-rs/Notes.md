This is pretty rough as an NN framework. 
This rust framework claims to be a novel new deep learning framework that doesn't compromise of flexibility, efficiency, and portability, yet to define a model you need 3 separate things:
- a structure to define the model(order of layers)
- a structure to define the model config(specification of layers)
- a forward pass upon those layers

And fom there yeah it's easy to create a template and move around backends, but the amount of repeat work in definition is still quite painful.
The actual swapping of backends and batching data utilities seem useful, akin to torch but with some more map and filter pieces.

Their suggested sample code to define the network can be found in the src folder here.

[https://burn.dev/books/burn/overview.html]
