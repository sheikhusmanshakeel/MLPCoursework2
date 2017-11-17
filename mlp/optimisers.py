# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh

import numpy
import time
import logging

from mlp.layers import MLP
from mlp.dataset import DataProvider
from mlp.schedulers import LearningRateScheduler


# logger = logging.getLogger(__name__)
logging.basicConfig()
logger = logging.getLogger('My Logger')
logger.setLevel(logging.INFO)

class Optimiser(object):
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
        for i in xrange(0, len(model.layers)):
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
            "Expected learningRateExponential to be a subclass of 'mlp.schedulers.LearningRateScheduler'"
            " class but got %s " % type(lr_scheduler)
        )

        self.lr_scheduler = lr_scheduler
        self.dp_scheduler = dp_scheduler
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.tr_stats = []
        self.valid_stats = []

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
        # tr_stats, valid_stats = [], []

        # do the initial validation
        train_iterator.reset()
        tr_nll, tr_acc = self.validate(model, train_iterator, self.l1_weight, self.l2_weight)
        logger.info('Epoch %i: Training cost (%s) for initial model is %.3f. Accuracy is %.2f%%'
                    % (self.lr_scheduler.epoch, cost_name, tr_nll, tr_acc * 100.))
        self.tr_stats.append((tr_nll, tr_acc))

        if valid_iterator is not None:
            valid_iterator.reset()
            valid_nll, valid_acc = self.validate(model, valid_iterator, self.l1_weight, self.l2_weight)
            logger.info('Epoch %i: Validation cost (%s) for initial model is %.3f. Accuracy is %.2f%%'
                        % (self.lr_scheduler.epoch, cost_name, valid_nll, valid_acc * 100.))
            self.valid_stats.append((valid_nll, valid_acc))

        while not converged:
            train_iterator.reset()

            tstart = time.clock()
            tr_nll, tr_acc = self.train_epoch(model=model,
                                              train_iterator=train_iterator,
                                              learning_rate=self.lr_scheduler.get_rate())
            tstop = time.clock()
            self.tr_stats.append((tr_nll, tr_acc))

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
                self.valid_stats.append((valid_nll, valid_acc))
            else:
                self.lr_scheduler.get_next_rate(None)
            vstop = time.clock()

            train_speed = train_iterator.num_examples_presented() / (tstop - tstart)
            valid_speed = valid_iterator.num_examples_presented() / (vstop - vstart)
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

        return self.tr_stats, self.valid_stats

    def pretrain_epoch(self, model, train_iterator, learning_rate, inputs, isNoisy, activationsPresent,activations=None):

        acc_list, nll_list = [], []
        for x in inputs:
            if isNoisy:
                # make noisy x
                # use noisy x to do the fprop
                inputData = x
                noise = numpy.random.uniform()
                for imagePosition in range(0, len(x)):
                    image = x[imagePosition].reshape(28, 28)
                    noisyImage = scipy.ndimage.gaussian_filter(image, sigma=noise)
                    noisyImage = noisyImage.flatten()
                    inputData[imagePosition] = noisyImage
            else:
                inputData = x
            if self.dp_scheduler is not None:
                y = model.fprop_dropout(inputData, self.dp_scheduler)
            else:
                y = model.fprop(inputData)
            if activationsPresent:
                # print('use this activations')
                x = activations
            cost = model.cost.cost(y, x)
            cost_grad = model.cost.grad(y, x)

            model.bprop(cost_grad, self.dp_scheduler, isPretrain= True)
            # update the model, here we iterate over layers
            # and then over each parameter in the layer
            effective_learning_rate = learning_rate / x.shape[0]
            for i in xrange(len(model.layers) - 2, len(model.layers)):
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
        # compute the prior penalties contribution (parameter dependent only)
        prior_costs = Optimiser.compute_prior_costs(model, self.l1_weight, self.l2_weight)
        training_cost = numpy.mean(nll_list) + sum(prior_costs)
        return training_cost, numpy.mean(acc_list)

    def pretrain(self, model, train_iterator, activationsPresent, inputs, isNoisy, activations=None,valid_iterator=None):
        converged = False
        cost_name = model.cost.get_name()
        tr_stats, valid_stats = [], []
        # do the initial validation
        train_iterator.reset()
        logger.info('Pre-Train called')
        while not converged:
            # print('Pre-train epoch called for epoch {0}'.format(self.learningRateExponential.epoch))
            train_iterator.reset()
            tstart = time.clock()
            tr_nll, tr_acc = self.pretrain_epoch(model=model, train_iterator=train_iterator,
                                                 learning_rate=self.lr_scheduler.get_rate(), inputs=inputs,
                                                 isNoisy=isNoisy,
                                                 activationsPresent=activationsPresent, activations=activations
                                                 )
            tstop = time.clock()
            tr_stats.append((tr_nll, tr_acc))
            logger.info('Epoch %i: Training cost (%s) is %.3f. Accuracy is %.2f%%'% (self.lr_scheduler.epoch, cost_name, tr_nll, tr_acc * 100.))
            train_speed = train_iterator.num_examples_presented() / (tstop - tstart)
            tot_time = tstart
            logger.info("Epoch %i: Took %.0f seconds. Training speed %.0f pps. "% (self.lr_scheduler.epoch, tot_time, train_speed))

            # we stop training when learning rate, as returned by lr scheduler, is 0
            # this is implementation dependent and depending on lr schedule could happen,
            # for example, when max_epochs has been reached or if the progress between
            # two consecutive epochs is too small, etc.
            newRate = self.lr_scheduler.get_next_rate()
            converged = (newRate == 0)
        # logger.info('Pretrain Epoch Finished')
        return tr_stats, valid_stats

    def pretrain_discriminative_epoch(self, model, train_iterator, learning_rate, inputs, isNoisy = False, activationsPresent = False,activations=None):

        acc_list, nll_list = [], []
        for dl in inputs:

            x = dl[0]
            t = dl[1]

            if isNoisy:
                # make noisy x
                # use noisy x to do the fprop
                inputData = x
                noise = numpy.random.uniform()
                for imagePosition in range(0, len(x)):
                    image = x[imagePosition].reshape(28, 28)
                    noisyImage = scipy.ndimage.gaussian_filter(image, sigma=noise)
                    noisyImage = noisyImage.flatten()
                    inputData[imagePosition] = noisyImage
            else:
                inputData = x


            if self.dp_scheduler is not None:
                y = model.fprop_dropout(inputData, self.dp_scheduler)
            else:
                y = model.fprop(inputData)
            # If activations are present, then use those activations from the outside
            # This will never be called the first time round, but will always be called after
            if activationsPresent:
                x = activations

            # Since this is training using cross entropy, we shud compare with targets rather than x
            cost = model.cost.cost(y, t)
            cost_grad = model.cost.grad(y, t)

            model.bprop(cost_grad, self.dp_scheduler, isPretrain= True)

            # update the model, here we iterate over layers
            # and then over each parameter in the layer
            # but since it is pretraining, we only iterate over the last two layers
            effective_learning_rate = learning_rate / x.shape[0]
            for i in xrange(len(model.layers) - 2, len(model.layers)):
                params = model.layers[i].get_params()
                grads = model.layers[i].pgrads(inputs=model.activations[i],deltas=model.deltas[i + 1],l1_weight=self.l1_weight,l2_weight=self.l2_weight)
                uparams = []
                for param, grad in zip(params, grads):
                    param = param - effective_learning_rate * grad
                    uparams.append(param)
                model.layers[i].set_params(uparams)

            nll_list.append(cost)
            acc_list.append(numpy.mean(self.classification_accuracy(y, t)))
        # compute the prior penalties contribution (parameter dependent only)
        prior_costs = Optimiser.compute_prior_costs(model, self.l1_weight, self.l2_weight)
        training_cost = numpy.mean(nll_list) + sum(prior_costs)
        return training_cost, numpy.mean(acc_list)

    def pretrain_descriminative(self, model, train_iterator, activationsPresent, inputs, isNoisy, activations=None,valid_iterator=None):
        converged = False
        cost_name = model.cost.get_name()
        tr_stats, valid_stats = [], []
        # do the initial validation
        train_iterator.reset()
        logger.info('Pre-Train called')
        while not converged:
            # print('Pre-train epoch called for epoch {0}'.format(self.learningRateExponential.epoch))
            train_iterator.reset()
            tstart = time.clock()
            tr_nll, tr_acc = self.pretrain_discriminative_epoch(model=model, train_iterator=train_iterator,
                                                 learning_rate=self.lr_scheduler.get_rate(), inputs=inputs,
                                                 isNoisy=isNoisy,
                                                 activationsPresent=activationsPresent, activations=activations
                                                 )
            tstop = time.clock()
            tr_stats.append((tr_nll, tr_acc))
            logger.info('Epoch %i: Training cost (%s) is %.3f. Accuracy is %.2f%%'% (self.lr_scheduler.epoch, cost_name, tr_nll, tr_acc * 100.))
            train_speed = train_iterator.num_examples_presented() / (tstop - tstart)
            tot_time = tstart
            logger.info("Epoch %i: Took %.0f seconds. Training speed %.0f pps. "% (self.lr_scheduler.epoch, tot_time, train_speed))

            # we stop training when learning rate, as returned by lr scheduler, is 0
            # this is implementation dependent and depending on lr schedule could happen,
            # for example, when max_epochs has been reached or if the progress between
            # two consecutive epochs is too small, etc.
            newRate = self.lr_scheduler.get_next_rate()
            converged = (newRate == 0)
        # logger.info('Pretrain Epoch Finished')
        return tr_stats, valid_stats

    def GetTrainingStats(self):
        return self.tr_stats

    def GetValidationStats(self):
        return self.valid_stats

