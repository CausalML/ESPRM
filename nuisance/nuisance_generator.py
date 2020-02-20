from nuisance.estimate_mean_y_network import train_mean_y_network
from nuisance.estimate_propensity_network import train_propensity_network


class AbstractNuisanceGenerator(object):
    def __init__(self, scenario):
        self.mean_y_network = None
        self.propensity_network = None
        self.scenario = scenario

    def setup(self, x, a, y, x_dev, a_dev, y_dev):
        raise NotImplementedError()

    def get_mean_y_network(self):
        return self.mean_y_network

    def get_propensity_network(self):
        return self.propensity_network


class StandardNuisanceGenerator(AbstractNuisanceGenerator):
    def __init__(self, scenario, y_method, p_method, y_args=None, p_args=None):
        AbstractNuisanceGenerator.__init__(self, scenario)
        self.y_method = y_method
        self.p_method = p_method
        if y_args:
            self.y_args = y_args
        else:
            self.y_args = {}
        if p_args:
            self.p_args = p_args
        else:
            self.p_args = {}

    def setup(self, x, a, y, x_dev, a_dev, y_dev):
        self.mean_y_network = train_mean_y_network(
            x=x, a=a, y=y, method=self.y_method,
            x_dev=x_dev, a_dev=a_dev, y_dev=y_dev, **self.y_args)
        self.propensity_network = train_propensity_network(
            x=x, a=a, method=self.p_method,
            x_dev=x_dev, a_dev=a_dev, **self.p_args)


class OracleNuisanceGenerator(AbstractNuisanceGenerator):
    def __init__(self, scenario):
        AbstractNuisanceGenerator.__init__(self, scenario)

    def setup(self, x, a, y, x_dev, a_dev, y_dev):
        self.mean_y_network = self.scenario.get_y_network()
        self.propensity_network = self.scenario.get_propensity_network()


class SampleDataNuisanceGenerator(AbstractNuisanceGenerator):
    def __init__(self, scenario, num_sample, y_method, p_method,
                 y_args=None, p_args=None):
            AbstractNuisanceGenerator.__init__(self, scenario)
            self.num_sample = num_sample
            self.y_method = y_method
            self.p_method = p_method
            if y_args:
                self.y_args = y_args
            else:
                self.y_args = {}
            if p_args:
                self.p_args = p_args
            else:
                self.p_args = {}

    def setup(self, x, a, y, x_dev, a_dev, y_dev):
        x_new, a_new, y_new, _ = self.scenario.sample_data(self.num_sample)
        self.mean_y_network = train_mean_y_network(
            x=x_new, a=a_new, y=y_new, method=self.y_method,
            x_dev=x_dev, a_dev=a_dev, y_dev=y_dev, **self.y_args)
        self.propensity_network = train_propensity_network(
            x=x_new, a=a_new, method=self.p_method,
            x_dev=x_dev, a_dev=a_dev, **self.p_args)

