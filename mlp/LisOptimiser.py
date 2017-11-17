# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh

import numpy
import time
import logging
from scipy import ndimage

from mlp.layers import MLP
from mlp.dataset import DataProvider
from mlp.schedulers import LearningRateScheduler

from mlp.layers import MLP, Linear, Sigmoid, Softmax #import required layer types
# from mlp.optimisers import SGDOptimiser, Optimiser #import the optimiser
from mlp.dataset import MNISTDataProvider #import data provider
from mlp.costs import MSECost,CECost #import the cost we want to use for optimisation
from mlp.schedulers import LearningRateFixed
import matplotlib.pyplot as plt
logger = logging.getLogger('Li')


class Optimiser(object):
    def pretrain_epoch(self, model, train_iter):
        raise NotImplementedError()

    def pretrain(self, model, train_iter, valid_iter=None):
        raise NotImplementedError()

    def train_epoch(self, model, train_iter):
        raise NotImplementedError()

    def train(self, model, train_iter, valid_iter=None):
        raise NotImplementedError()

    def validate(self, model, valid_iterator, l1_weight=0, l2_weight=0):
        assert isinstance(model, MLP), (
            "Expected model to be a subclass of 'mlp.layers.MLP'"
            " class but got %s " % type(model)
        )

        assert isinstance(valid_iterator, DataProvider), (
            "Expected iterator to be a subclass of 'mlp.dataset.DataProvider'"
            " class but got %s " % type(valid_iterator)
        )

        acc_list, nll_list = [], []
        for x, t in valid_iterator:
            y = model.fprop(x)
            nll_list.append(model.cost.cost(y, t))
            acc_list.append(numpy.mean(self.classification_accuracy(y, t)))

        acc = numpy.mean(acc_list)
        nll = numpy.mean(nll_list)

        prior_costs = Optimiser.compute_prior_costs(model, l1_weight, l2_weight)

        return nll + sum(prior_costs), acc

    @staticmethod
    def classification_accuracy(y, t):
        """
        Returns classification accuracy given the estimate y and targets t
        :param y: matrix -- estimate produced by the model in fprop
        :param t: matrix -- target  1-of-K coded
        :return: vector of y.shape[0] size with binary values set to 0
                 if example was miscalssified or 1 otherwise
        """
        y_idx = numpy.argmax(y, axis=1)
        t_idx = numpy.argmax(t, axis=1)
        rval = numpy.equal(y_idx, t_idx)
        return rval

    @staticmethod
    def compute_prior_costs(model, l1_weight, l2_weight):
        """
        Computes the cost contributions coming from parameter-dependent only
        regularisation penalties
        """
        assert isinstance(model, MLP), (
            "Expected model to be a subclass of 'mlp.layers.MLP'"
            " class but got %s " % type(model)
        )

        l1_cost, l2_cost = 0, 0
        for i in range(0, len(model.layers)):
            params = model.layers[i].get_params()
            for param in params:
                if l2_weight > 0:
                    l2_cost += 0.5 * l2_weight * numpy.sum(param**2)
                if l1_weight > 0:
                    l1_cost += l1_weight * numpy.sum(numpy.abs(param))

        return l1_cost, l2_cost


class SGDOptimiser(Optimiser):


    def __init__(self, lr_scheduler,
                 dp_scheduler=None,
                 l1_weight=0.0,
                 l2_weight=0.0):

        super(SGDOptimiser, self).__init__()

        assert isinstance(lr_scheduler, LearningRateScheduler), (
            "Expected lr_scheduler to be a subclass of 'mlp.schedulers.LearningRateScheduler'"
            " class but got %s " % type(lr_scheduler)
        )

        self.lr_scheduler = lr_scheduler
        self.dp_scheduler = dp_scheduler
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def pretrain(self, model, train_iterator, valid_iterator=None,denoising=False):
        if denoising:
            noise = numpy.random.uniform()

            train_iterator.x = ndimage.gaussian_filter(train_iterator.x,sigma=noise)

        rng = numpy.random.RandomState([2015,10,10])
        converged = False
        # cost_name = model.cost.get_name()
        cost = MSECost()
        inputs = train_iterator
        converged = False
        # cost_name = model.cost.get_name()
        tr_stats = []

        for i in range (0,len(model.layers)):
            converged = False
            self.lr_scheduler.epoch = 0
            tmpmodel = MLP(cost=cost)
            cost_name = tmpmodel.cost.get_name()
            tmphidlayerin = Sigmoid(idim=model.layers[i].idim, odim=model.layers[i].odim,rng=rng)
            tmphidlayerout = Sigmoid(idim=model.layers[i].odim, odim=model.layers[i].idim,rng=rng)
            tmpmodel.add_layer(tmphidlayerin)
            tmpmodel.add_layer(tmphidlayerout)
            print i
            while not converged:
                inputs.reset()
                tstart = time.clock()
                tr_nll, tr_acc = self.pretrain_epoch(tmpmodel, inputs, learning_rate=self.lr_scheduler.get_rate())
                tstop = time.clock()
                tr_stats.append((tr_nll, tr_acc))
                logger.info('Epoch %i: Training cost (%s) is %.3f. Accuracy is %.2f%%'
                            % (self.lr_scheduler.epoch + 1, cost_name, tr_nll, tr_acc * 100.))
                train_speed = train_iterator.num_examples_presented() / (tstop - tstart)
                tot_time = tstop - tstart
                self.lr_scheduler.get_next_rate(None)
                #pps = presentations per second
                logger.info("Epoch %i: Took %.0f seconds. Training speed %.0f pps. "
                            % (self.lr_scheduler.epoch, tot_time, train_speed))
                converged = (self.lr_scheduler.get_rate() == 0)
            wb = tmpmodel.layers[0].get_params()
            w = wb[0]
            b = wb[1]
            c = [w,b]
            wb2 = model.layers[i].get_params()
            w2=wb2[0]

            if w.shape==w2.shape:
                print 'good'
            else:
                print 'bad'
            model.layers[i].set_params(c)
            #model.activation = fprop(x)
            outputs = tmpmodel.layers[0].fprop(inputs.x)

            inputs.x = outputs
            #imgout = tmpmodel.fprop(inputs.x)
            #imgout = imgout[100,:].reshape(28,28)
            #inx = inputs.x
            #imgin = inx[100,:].reshape(28,28)
        #return imgout, imgin

    def pretrain_epoch(self, model, inputs, learning_rate):
        acc_list, nll_list = [], []
        for x, t in inputs:
            if self.dp_scheduler is not None:
                y = model.fprop_dropout(x, self.dp_scheduler)
            else:
                y = model.fprop(x)
            cost = model.cost.cost(y, x)
            cost_grad = model.cost.grad(y, x)
            model.bprop(cost_grad, self.dp_scheduler)
            effective_learning_rate = learning_rate / x.shape[0]

            for i in xrange(0, len(model.layers)):
                params = model.layers[i].get_params()
                grads = model.layers[i].pgrads(inputs=model.activations[i],
                                               deltas=model.deltas[i + 1],
                                               l1_weight=self.l1_weight,
                                               l2_weight=self.l2_weight)
                uparams = []
                for param, grad in zip(params, grads):
                    param = param - effective_learning_rate * grad
                    uparams.append(param)
                model.layers[i].set_params(uparams)

            nll_list.append(cost)
            acc_list.append(numpy.mean(self.classification_accuracy(y, x)))

        #compute the prior penalties contribution (parameter dependent only)
        prior_costs = Optimiser.compute_prior_costs(model, self.l1_weight, self.l2_weight)
        training_cost = numpy.mean(nll_list) + sum(prior_costs)

        return training_cost, numpy.mean(acc_list)
    def pretrain_discriminative(self,model, train_iterator, valid_iterator=None):
        converged=False
        cost =CECost()
        cost_name = model.cost.get_name()
        rng = numpy.random.RandomState([2015,10,10])
        tr_stats, valid_stats = [], []
        cross_pretrain_model_fprop = MLP(cost=cost)
        cross_pretrain_model_bprop = MLP(cost=cost)
        pretrain_layers_fprop,pretrain_layers_bprop = [],[]
        for i in range(0,len(model.layers)-1):
            print i,'i'
            self.lr_scheduler.epoch = 0
            converged=False
            self.lr_scheduler.epoch=0
            tmphidlayer = Sigmoid(idim=model.layers[i].idim, odim=model.layers[i].odim,rng=rng)
            tmpsoftmax = Softmax(idim=model.layers[i].odim, odim=10, rng=rng)
            pretrain_layers_bprop=[]
            pretrain_layers_fprop=cross_pretrain_model_fprop.layers
            print pretrain_layers_fprop
            if len(pretrain_layers_fprop)!=0:
                pretrain_layers_fprop.pop()
            pretrain_layers_fprop.append(tmphidlayer)
            pretrain_layers_fprop.append(tmpsoftmax)
            pretrain_layers_bprop.append(tmphidlayer)
            pretrain_layers_bprop.append(tmpsoftmax)
            cross_pretrain_model_fprop.set_layers(layers=pretrain_layers_fprop)
            print len(cross_pretrain_model_fprop.layers),'fprop len'
            cross_pretrain_model_bprop.set_layers(layers=pretrain_layers_bprop)
            print len(cross_pretrain_model_bprop.layers),'bprop len'
            while not converged:
                tstart = time.clock()
                tr_nll, tr_acc = self.pretrain_epoch_discriminative(fprop_model=cross_pretrain_model_fprop,bprop_model=cross_pretrain_model_bprop,inputs=train_iterator,learning_rate=self.lr_scheduler.get_rate())
                tstop = time.clock()
                tr_stats.append((tr_nll, tr_acc))
                print tr_acc,'tr_acc'
                logger.info('Epoch %i: Training cost (%s) is %.3f. Accuracy is %.2f%%'
                            % (self.lr_scheduler.epoch + 1, cost_name, tr_nll, tr_acc * 100.))
                train_speed = train_iterator.num_examples_presented() / (tstop - tstart)
                tot_time = tstop - tstart
                self.lr_scheduler.get_next_rate(None)
                #pps = presentations per second
                logger.info("Epoch %i: Took %.0f seconds. Training speed %.0f pps. "
                            % (self.lr_scheduler.epoch, tot_time, train_speed))
                converged = (self.lr_scheduler.get_rate() == 0)
                cross_pretrain_model_fprop.layers.pop()
                cross_pretrain_model_fprop.layers.pop()
                cross_pretrain_model_fprop.layers.append(cross_pretrain_model_bprop.layers[0])
                cross_pretrain_model_fprop.layers.append(cross_pretrain_model_bprop.layers[1])
                print cross_pretrain_model_fprop.layers,'converged'

        print cross_pretrain_model_fprop.layers,'cross'
        #fine-tuning
        converged=False
        depth = len(model.layers)
        #cross_pretrain_model_fprop.add_layer(Softmax(idim=model.layers[depth-1].idim,odim=10,rng=rng))
        model.layers=cross_pretrain_model_fprop.layers
        while not converged:
            tstart = time.clock()
            tr_nll, tr_acc = self.pretrain_epoch_discriminative(fprop_model=model,bprop_model=model,inputs=train_iterator,learning_rate=self.lr_scheduler.get_rate())
            tstop = time.clock()

            tr_stats.append((tr_nll, tr_acc))
            logger.info('Epoch %i: Training cost (%s) is %.3f. Accuracy is %.2f%%'
                        % (self.lr_scheduler.epoch + 1, cost_name, tr_nll, tr_acc * 100.))
            train_speed = train_iterator.num_examples_presented() / (tstop - tstart)
            tot_time = tstop - tstart
            self.lr_scheduler.get_next_rate(None)
            #pps = presentations per second
            logger.info("Epoch %i: Took %.0f seconds. Training speed %.0f pps. "
                        % (self.lr_scheduler.epoch, tot_time, train_speed))
            converged = (self.lr_scheduler.get_rate() == 0)

    def pretrain_epoch_discriminative(self, fprop_model, bprop_model,inputs, learning_rate):
        acc_list, nll_list = [], []
        for x,t in inputs:
            if self.dp_scheduler is not None:
                y = fprop_model.fprop_dropout(x, self.dp_scheduler)
            else:
                y = fprop_model.fprop(x)
            cost = fprop_model.cost.cost(y, t)
            cost_grad = fprop_model.cost.grad(y, t)
            bprop_model.activations = fprop_model.activations
            bprop_model.deltas = fprop_model.deltas
            bprop_model.bprop(cost_grad, self.dp_scheduler)
            effective_learning_rate = learning_rate / x.shape[0]

            for i in xrange(0, len(bprop_model.layers)):
                params = bprop_model.layers[i].get_params()
                grads = bprop_model.layers[i].pgrads(inputs=bprop_model.activations[i],
                                               deltas=bprop_model.deltas[i + 1],
                                               l1_weight=self.l1_weight,
                                               l2_weight=self.l2_weight)
                uparams = []
                for param, grad in zip(params, grads):
                    param = param - effective_learning_rate * grad
                    uparams.append(param)
                bprop_model.layers[i].set_params(uparams)

            nll_list.append(cost)
            acc_list.append(numpy.mean(self.classification_accuracy(y, t)))

        #compute the prior penalties contribution (parameter dependent only)
        prior_costs = Optimiser.compute_prior_costs(bprop_model, self.l1_weight, self.l2_weight)
        training_cost = numpy.mean(nll_list) + sum(prior_costs)

        return training_cost, numpy.mean(acc_list)

    def train_epoch(self, model, train_iterator, learning_rate):

        assert isinstance(model, MLP), (
            "Expected model to be a subclass of 'mlp.layers.MLP'"
            " class but got %s " % type(model)
        )
        assert isinstance(train_iterator, DataProvider), (
            "Expected iterator to be a subclass of 'mlp.dataset.DataProvider'"
            " class but got %s " % type(train_iterator)
        )

        acc_list, nll_list = [], []
        for x, t in train_iterator:
            # t = x ##

            # get the prediction
            if self.dp_scheduler is not None:
                y = model.fprop_dropout(x, self.dp_scheduler)
            else:
                y = model.fprop(x)

            # compute the cost and grad of the cost w.r.t y
            cost = model.cost.cost(y, t)
            cost_grad = model.cost.grad(y, t)

            # do backward pass through the model
            model.bprop(cost_grad, self.dp_scheduler)

            #update the model, here we iterate over layers
            #and then over each parameter in the layer
            effective_learning_rate = learning_rate / x.shape[0]

            for i in xrange(0, len(model.layers)):
                params = model.layers[i].get_params()
                grads = model.layers[i].pgrads(inputs=model.activations[i],
                                               deltas=model.deltas[i + 1],
                                               l1_weight=self.l1_weight,
                                               l2_weight=self.l2_weight)
                uparams = []
                for param, grad in zip(params, grads):
                    param = param - effective_learning_rate * grad
                    uparams.append(param)
                model.layers[i].set_params(uparams)

            nll_list.append(cost)
            acc_list.append(numpy.mean(self.classification_accuracy(y, t)))

        #compute the prior penalties contribution (parameter dependent only)
        prior_costs = Optimiser.compute_prior_costs(model, self.l1_weight, self.l2_weight)
        training_cost = numpy.mean(nll_list) + sum(prior_costs)
        if self.dp_scheduler is not None:
            self.dp_scheduler.get_next_rate()
        return training_cost, numpy.mean(acc_list)

    def train(self, model, train_iterator, valid_iterator=None):

        converged = False
        cost_name = model.cost.get_name()
        tr_stats, valid_stats = [], []

        # do the initial validation
        train_iterator.reset()
        # tr_nll, tr_acc = self.validate(model, train_iterator, self.l1_weight, self.l2_weight)
        # logger.info('Epoch %i: Training cost (%s) for initial model is %.3f. Accuracy is %.2f%%'
        #            % (self.lr_scheduler.epoch, cost_name, tr_nll, tr_acc * 100.))
        # tr_stats.append((tr_nll, tr_acc))

        if valid_iterator is not None:
            valid_iterator.reset()
            valid_nll, valid_acc = self.validate(model, valid_iterator, self.l1_weight, self.l2_weight)
            logger.info('Epoch %i: Validation cost (%s) for initial model is %.3f. Accuracy is %.2f%%'
                        % (self.lr_scheduler.epoch, cost_name, valid_nll, valid_acc * 100.))
            valid_stats.append((valid_nll, valid_acc))

        while not converged:
            train_iterator.reset()

            tstart = time.clock()
            tr_nll, tr_acc = self.train_epoch(model=model,
                                              train_iterator=train_iterator,
                                              learning_rate=self.lr_scheduler.get_rate())
            tstop = time.clock()
            tr_stats.append((tr_nll, tr_acc))

            logger.info('Epoch %i: Training cost (%s) is %.3f. Accuracy is %.2f%%'
                        % (self.lr_scheduler.epoch + 1, cost_name, tr_nll, tr_acc * 100.))

            vstart = time.clock()
            if valid_iterator is not None:
                valid_iterator.reset()
                valid_nll, valid_acc = self.validate(model, valid_iterator,
                                                     self.l1_weight, self.l2_weight)
                logger.info('Epoch %i: Validation cost (%s) is %.3f. Accuracy is %.2f%%'
                            % (self.lr_scheduler.epoch + 1, cost_name, valid_nll, valid_acc * 100.))
                self.lr_scheduler.get_next_rate(valid_acc)
                valid_stats.append((valid_nll, valid_acc))
            else:
                self.lr_scheduler.get_next_rate(None)
            vstop = time.clock()

            train_speed = train_iterator.num_examples_presented() / (tstop - tstart)
            # valid_speed = valid_iterator.num_examples_presented() / (vstop - vstart)
            valid_speed = 0
            tot_time = vstop - tstart
            #pps = presentations per second
            logger.info("Epoch %i: Took %.0f seconds. Training speed %.0f pps. "
                        "Validation speed %.0f pps."
                        % (self.lr_scheduler.epoch, tot_time, train_speed, valid_speed))

            # we stop training when learning rate, as returned by lr scheduler, is 0
            # this is implementation dependent and depending on lr schedule could happen,
            # for example, when max_epochs has been reached or if the progress between
            # two consecutive epochs is too small, etc.
            converged = (self.lr_scheduler.get_rate() == 0)

        return tr_stats, valid_stats
