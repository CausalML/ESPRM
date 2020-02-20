import numpy as np
import torch


x_names = ["temps", "rsqstat", "zus", "College_education",
           "One_to_5_years_of_exp_in_the_job",  "Technician",
           "Skilled_clerical_worker", "Skilled_blue_colar",
           "Q1", "Q2", "Q3", "Q4"]
x_types = {
    "temps": "int",
    "rsqstat": "cat",
    "zus": "cat",
    "College_education": "int",
    "One_to_5_years_of_exp_in_the_job": "int",
    "Technician": "int",
    "Skilled_clerical_worker": "int",
    "Skilled_blue_colar": "int",
    "Q1": "int",
    "Q2": "int",
    "Q3": "int",
    "Q4": "int",
    "ipw": "float",
}

x_cat_levels = {
    "zus": ["AT", "IN", "NT", "NZ", "ZU"],
    "rsqstat": ["RS1", "RS2", "RS3"],
}


class JobsScenario(object):
    def __init__(self, jobs_df, num_dev, num_tune):
        object.__init__(self)
        x_cols = []
        for x_name, x_type in sorted(x_types.items()):
            if x_type == "int" or x_type == "float":
                x_cols.append(jobs_df[x_name])
            elif x_type == "cat":
                for level in x_cat_levels[x_name]:
                    col = (jobs_df[x_name] == level) * 1
                    x_cols.append(col)

        self.x = torch.DoubleTensor(np.stack(x_cols, axis=1))
        self.a = torch.LongTensor(jobs_df["t"]) - 1
        self.y = torch.DoubleTensor(jobs_df["y"]).view(-1, 1)
        self.ipw = torch.DoubleTensor(jobs_df["ipw"])
        if torch.cuda.is_available():
            self.x = self.x.cuda()
            self.a = self.a.cuda()
            self.y = self.y.cuda()
            self.ipw = self.ipw.cuda()

        self.dev_idx = [i for i in range(num_dev) if self.a[i] != 2]
        self.tune_idx = [i for i in range(num_dev, num_tune+num_dev)
                         if self.a[i] != 2]
        self.train_idx = [i for i in range(num_tune+num_dev, len(self.x))
                          if self.a[i] != 2]

    def get_dev(self):
        return (self.x[self.dev_idx],
                self.a[self.dev_idx],
                self.y[self.dev_idx])

    def get_tune(self):
        return (self.x[self.tune_idx],
                self.a[self.tune_idx],
                self.y[self.tune_idx])

    def get_train(self):
        return (self.x[self.train_idx],
                self.a[self.train_idx],
                self.y[self.train_idx])

    def get_all_data_for_testing(self):
        return self.x, self.a, self.y

    def get_ipw(self):
        return self.ipw
