import numpy as np

from utils.metrics.Metrics import Metrics


class PPL(Metrics):
    def __init__(self, data_loader, rnn, sess):
        super().__init__()
        self.name = 'ppl-oracle'
        self.data_loader = data_loader
        self.sess = sess
        self.rnn = rnn

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_score(self):
        return self.ppl_loss()

    def ppl_loss(self):
        ppl = []
        self.data_loader.reset_pointer()
        for it in range(self.data_loader.num_batch):
            batch = self.data_loader.next_batch()
            # fixme bad taste
            try:
                ppl_one = self.rnn.get_ppl(self.sess, batch)
            except Exception as e:
                ppl_one = self.sess.run(self.rnn.ppl_one, {self.rnn.x: batch})
            ppl.append(ppl_one)
        return 1.0/(np.sum(ppl)/(self.data_loader.num_batch*self.data_loader.seq_length*self.data_loader.batch_size))
