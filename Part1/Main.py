import logging
import numpy

logging.basicConfig()
logger = logging.getLogger('My Logger')
logger.setLevel(logging.INFO)

logger.info("write something here")

from mlp.layers import MLP, Linear, Sigmoid, Softmax  # import required layer types
from mlp.optimisers import SGDOptimiser  # import the optimiser
from mlp.dataset import MNISTDataProvider  # import data provider
from mlp.costs import MSECost, CECost  # import the cost we want to use for optimisation
from mlp.schedulers import LearningRateFixed

rng = numpy.random.RandomState([2015, 10, 10])

# define the model structure, here just one linear layer
# and mean square error cost
#cost = MSECost()
cost = CECost()
model = MLP(cost=cost)
model.add_layer(Linear(idim=784, odim=100, rng=rng))
#model.add_layer(Sigmoid(inputDimensions=100, outputDimensions=10, rng=rng))
model.add_layer(Softmax(inputDimensions=100,outputDimensions=10, rng=rng))

# one can stack more layers here

# define the optimiser, here stochasitc gradient descent
# with fixed learning rate and max_epochs as stopping criterion
lr_scheduler = LearningRateFixed(learning_rate=0.01, max_epochs=20)
optimiser = SGDOptimiser(lr_scheduler=lr_scheduler)

logger.info('Initialising data providers...')
train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=-10, randomize=True)
valid_dp = MNISTDataProvider(dset='valid', batch_size=100, max_num_batches=-10, randomize=False)

logger.info('Training started...')
optimiser.train(model, train_dp, valid_dp)

logger.info('Testing the model on test set:')
test_dp = MNISTDataProvider(dset='eval', batch_size=100, max_num_batches=-10, randomize=False)
cost, accuracy = optimiser.validate(model, test_dp)
logger.info('MNIST test set accuracy is %.2f %% (cost is %.3f)' % (accuracy * 100., cost))
