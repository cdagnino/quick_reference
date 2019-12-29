# PyTorch

Follows the book Deep Learning With Pytorch a lot.

### Strides and storage

FILL IN



### Serializing tensors

```python
torch.save(points, "data/ourpoints.t")
points = torch.load("data/ourpoints.t")`
```





+ `torch.save(points, "data/ourpoints.t")`

+ `torch.load("data/ourpoints.t")`



### From/to cuda

```python
points_gpu = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 4.0]],
        device='cuda')
#Type changes to torch.cuda.FloatTensor
points_gpu = points.to(device='cuda') #'cuda:0' if needed

points_cpu = points_gpu.to(device='cpu')
```



### Exercises

- 􏰀  Create a tensor a from list(range(9)). Predict then check what the size, off-
  set, and strides are.

- 􏰀  Create a tensor b = a.view(3, 3). What is the value of b[1,1]?

- 􏰀  Create a tensor c = b[1:,1:]. Predict then check what the size, offset, and
  
  strides are.

- 􏰀  Pick a mathematical operation like cosine or square root. Can you find a corre-
  
  sponding function in the torch library?
  
  ```python
  b = torch.from_numpy(np.array([2.1, 3.6]))
  a = torch.tensor([2.1, 2.2])
  torch.sin(a)
  torch.sin_(a) #in-place
  ```
  
  
  
  ## Ch 3: Real-world data representation with tensors
  
  

```python
bad_indexes = torch.le(target, 3) #less or equal than 3
#gt = greater, lt= less than
mid_data = data[torch.gt(target, 3) & torch.lt(target, 7)]
```



### Text

```python
 with open('../data/p1ch4/jane-austin/1342-0.txt', encoding='utf8') as f:
            text = f.read()
               
```

### Images

PyTorch modules that deal with image data require tensors to be laid out as C x H x W (channels, height, and width, respectively). Batches operate with N x C x H x W tensors (where N is the batch size).



For example, pre-allocate a vector and load images

```python
batch_size = 100
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)

filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name) == '.png']
for i, filename in enumerate(filenames):
    img_arr = imageio.imread(filename)
    batch[i] = torch.transpose(torch.from_numpy(img_arr), 0, 2)
```



# How to make sure things are working in cuda





**It's easy to pass the model to the GPU**


```python
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

model = MyModel(blabla, blablu)
model.to(device)
```



#### How to make sure the data is passed appropriately to the GPU?

```python
def train(args, model, device, train_loader, optimizer, epoch):
	#model.train() tells your model that you are training the model. layers like dropout, batchnorm etc. which behave different on the train and test procedures know what is going on and hence can behave accordingly.
	#Call model.eval() or model.train(mode=False) to tell that you are testing
	model.train()

```





### Example of training loop with MNist

```python
model = MyModel().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
for epoch in range(1, args.epochs + 1):
	train(args, model, device, train_loader, optimizer, epoch)
	test(args, model, device, test_loader)
scheduler.step()
if args.save_model:
	torch.save(model.state_dict(), "mnist_cnn.pt")
```





https://github.com/pytorch/examples/blob/master/mnist/main.py



## Recommendation Systems



## Todo

- Implement learn rate finder

- Check mean and std of parameters (batch normalization?)

- Check mean and std of input data

#### Matrix factorization

```python
import torch
#Variable is tensor with gradient
from torch.autograd import Variable

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
    # create user embeddings
        self.user_factors = torch.nn.Embedding(n_users, n_factors,
                                               sparse=True)
    # create item embeddings
        self.item_factors = torch.nn.Embedding(n_items, n_factors,
                                               sparse=True)

    def forward(self, user, item):
        # matrix multiplication
        return (self.user_factors(user)*self.item_factors(item)).sum(1)

    def predict(self, user, item):
        return self.forward(user, item)
        
## Training loop
#Movielens dataset with ratings scaled between [0, 1] to help with convergence.on the test set, error(RMSE) of 0.66
model = MatrixFactorization(n_users, n_items, n_factors=20)
loss_fn = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(),
                            lr=1e-6)

for user, item in zip(users, items):
    # get user, item and rating data
    rating = Variable(torch.FloatTensor([ratings[user, item]]))
    user = Variable(torch.LongTensor([int(user)]))
    item = Variable(torch.LongTensor([int(item)]))

    # predict
    prediction = model(user, item)
    loss = loss_fn(prediction, rating)

    # backpropagate
    loss.backward()

    # update weights
    optimizer.step()

```



#### Dense Network



The output of the embedding layer, which are two embedding vectors, are then concatenated into one and passed into a linear network. The output of the linear network is one dimensional - representing the rating for the user-item pair. The model is fit the same way as the matrix factorization model.



Once again, we train this model on the Movielens dataset with ratings scaled
between [0, 1] to help with convergence. Applied on the test set, we obtain a
root mean-squared error(RMSE) of 0.28, a substantial improvement!



```python
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

class DenseNet(nn.Module):

    def __init__(self, n_users, n_items, n_factors, H1, D_out):
        """
        Simple Feedforward with Embeddings
        """
        super().__init__()
   	# user and item embedding layers
        self.user_factors = torch.nn.Embedding(n_users, n_factors,
                                               sparse=True)
        self.item_factors = torch.nn.Embedding(n_items, n_factors,
                                               sparse=True)
   	# linear layers
        self.linear1 = torch.nn.Linear(n_factors*2, H1)
        self.linear2 = torch.nn.Linear(H1, D_out)

    def forward(self, users, items):
        users_embedding = self.user_factors(users)
        items_embedding = self.item_factors(items)
	# concatenate user and item embeddings to form input
        x = torch.cat([users_embedding, items_embedding], 1)
        h1_relu = F.relu(self.linear1(x))
        output_scores = self.linear2(h1_relu)
        return output_scores

    def predict(self, users, items):
        # return the score
        output_scores = self.forward(users, items)
        return output_scores


```



https://github.com/HarshdeepGupta/recommender_pytorch/blob/master/MLP.py has nice ideas for negative and positive sampling



Another good source: https://medium.com/@iliazaitsev/how-to-implement-a-recommendation-system-with-deep-learning-and-pytorch-2d40476590f9



And an example that uses some of the ideas of FastAI, but with pure PyTorch: https://github.com/devforfu/pytorch_playground/blob/master/movielens.ipynb


