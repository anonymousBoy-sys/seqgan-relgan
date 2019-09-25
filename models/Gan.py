from abc import abstractmethod, ABCMeta

from utils.utils import init_sess
from utils.text_process import code_to_text, get_tokenlized
import os
import numpy as np
import tensorflow as tf

class Gan(metaclass=ABCMeta):

    def __init__(self):
        self.oracle = None
        self.generator = None
        self.discriminator = None
        self.discriminator_new = None
        self.gen_data_loader = None
        self.dis_data_loader = None
        self.dis_train_data_loader = None
        self.dis_valid_data_loader = None
        self.oracle_data_loader = None
        self.fake_data_loader = None
        self.valid_data_loader = None
        self.test_data_loader = None
        self.sess = init_sess()
        self.metrics = list()
        self.epoch = 0
        self.log = None
        self.reward = None
        # temp file
        self.oracle_file = None
        self.generator_file = None
        self.text_file = None
        self.test_file = None
        self.valid_file = None
        # pathes
        self.output_path = None
        self.save_path = None
        self.summary_path = None
        # dict
        self.wi_dict = None
        self.iw_dict = None
        self.sequence_length = None
        self.vocab_size = None

    def set_config(self, config):
        self.__dict__.update(config.dict)

    def set_oracle(self, oracle):
        self.oracle = oracle

    def set_generator(self, generator):
        self.generator = generator

    def set_discriminator(self, discriminator, discriminator_new=None):
        self.discriminator = discriminator
        self.discriminator_new = discriminator_new

    def set_data_loader(self, gen_loader, dis_loader, dis_train_loader, dis_valid_loader, oracle_loader, fake_loader, valid_loader, test_loader):
        self.gen_data_loader = gen_loader
        self.dis_data_loader = dis_loader
        self.dis_train_data_loader = dis_train_loader
        self.dis_valid_data_loader = dis_valid_loader
        self.oracle_data_loader = oracle_loader
        self.fake_data_loader = fake_loader
        self.valid_data_loader = valid_loader
        self.test_data_loader = test_loader

    def set_sess(self, sess):
        self.sess = sess

    def add_metric(self, metric):
        self.metrics.append(metric)

    def add_epoch(self):
        self.epoch += 1

    def reset_epoch(self):
        # in use
        self.epoch = 0
        return

    def evaluate_scores(self):
        from time import time
        log = "epoch:" + str(self.epoch) + '\t'
        scores = list()
        scores.append(self.epoch)
        for metric in self.metrics:
            tic = time()
            score = metric.get_score()
            log += metric.get_name() + ":" + str(score) + '\t'
            toc = time()
            print(f"time elapsed of {metric.get_name()}: {toc - tic:.1f}s")
            scores.append(score)
        print(log)
        return scores

    def evaluate(self):
        if self.oracle_data_loader is not None:
            self.oracle_data_loader.create_batches(self.generator_file)
        with open(self.log, 'a') as log:
            if self.epoch == 0 or self.epoch == 1:
                head = ["epoch"]
                for metric in self.metrics:
                    head.append(metric.get_name())
                log.write(','.join(head) + '\n')
            scores = self.evaluate_scores()
            log.write(','.join([str(s) for s in scores]) + '\n')
        return scores
    
    def evaluate_real(self):
        self.generate_samples()
        self.get_real_test_file()
        self.evaluate()

    def get_real_test_file(self):
        with open(self.generator_file, 'r') as file:
            codes = get_tokenlized(self.generator_file)
        output = code_to_text(codes=codes, dictionary=self.iw_dict)
        with open(self.text_file, 'w', encoding='utf-8') as outfile:
            outfile.write(output)
        output_file = os.path.join(self.output_path, f"epoch_{self.epoch}.txt")
        with open(output_file, 'w', encoding='utf-8') as of:
            of.write(output)

    def generate_samples(self, oracle=False):
        if oracle:
            generator = self.oracle
            output_file = self.oracle_file
        else:
            generator = self.generator
            output_file = self.generator_file
        # Generate Samples
        generated_samples = []
        for _ in range(int(self.generated_num / self.batch_size)):
            generated_samples.extend(generator.generate(self.sess))
        codes = list()

        with open(output_file, 'w') as fout:
            for sent in generated_samples:
                buffer = ' '.join([str(x) for x in sent]) + '\n'
                fout.write(buffer)

    
    def generate_specified_score_sample(self):
        #Generate a negative sample of the specified score,
        #Here is a negative sample with a score of <0.3, 0.3-0.5, 0.5-0.9, >0.9
        #only apply to seqgan
        self.generate_samples()
        self.fake_data_loader.create_batches(self.generator_file)
        for _ in range(10):
            x_batch = self.fake_data_loader.next_batch()
            y_batch = [[1, 0] for _ in range(self.batch_size)]
            feed = {
                self.discriminator_new.input_x: x_batch,
                self.discriminator_new.input_y: y_batch,
            }
            neg_samples_less_03, neg_samples_03to05, neg_samples_05to09, neg_samples_greater_09= self.sess.run(
                [self.discriminator_new.neg_samples_less_03, self.discriminator_new.neg_samples_03to05, self.discriminator_new.neg_samples_05to09, self.discriminator_new.neg_samples_greater_09], feed)
            self.output_low_samples(neg_samples_less_03, 'neg_samples_less_03')
            self.output_low_samples(neg_samples_03to05, 'neg_samples_03to05')
            self.output_low_samples(neg_samples_05to09, 'neg_samples_05to09')
            self.output_low_samples(neg_samples_greater_09, 'neg_samples_greater_09')


    def output_low_samples(self, sentences, type):
        name = None
        if type == "neg_samples_greater_09":
            name = os.path.join(self.output_path, f"neg_samples_greater_09.txt")
        elif type == "neg_samples_05to09":
            name = os.path.join(self.output_path, f"neg_samples_05to09.txt")
        elif type == "neg_samples_03to05":
            name = os.path.join(self.output_path, f"neg_samples_03to05.txt")
        elif type == "neg_samples_less_03":
            name = os.path.join(self.output_path, f"neg_samples_less_03.txt")

        with open(name, 'a+') as fout:
            for sent in sentences:
                if np.sum(sent) == 0:
                    pass
                else:
                    buffer = ' '.join([str(x) for x in sent]) + '\n'
                    fout.write(buffer)
        outputs = code_to_text(codes=sentences, dictionary=self.iw_dict)
        output_files = os.path.join(self.output_path, f"{type}_textfile.txt")
        with open(output_files, 'w', encoding='utf-8') as of:
            of.write(outputs)

    def pre_train_epoch(self):
        # Pre-train the generator using MLE for one epoch
        supervised_g_losses = []
        self.gen_data_loader.reset_pointer()

        for it in range(self.gen_data_loader.num_batch):
            batch = self.gen_data_loader.next_batch()
            g_loss = self.generator.pretrain_step(self.sess, batch)
            supervised_g_losses.append(g_loss)

        return np.mean(supervised_g_losses)


    def init_real_metric(self):

        from utils.metrics.Nll import Nll
        from utils.metrics.PPL import PPL
        from utils.metrics.DocEmbSim import DocEmbSim
        from utils.others.Bleu import Bleu
        from utils.metrics.SelfBleu import SelfBleu

        if self.valid_ppl:
            valid_ppl = PPL(self.valid_data_loader, self.generator, self.sess)
            valid_ppl.set_name('valid_ppl')
            self.add_metric(valid_ppl)
        if self.nll_gen:
            nll_gen = Nll(self.gen_data_loader, self.generator, self.sess)
            nll_gen.set_name('nll_gen')
            self.add_metric(nll_gen)
        if self.doc_embsim:
            doc_embsim = DocEmbSim(
                self.oracle_file, self.generator_file, self.vocab_size)
            doc_embsim.set_name('doc_embsim')
            self.add_metric(doc_embsim)
        if self.bleu:
            FLAGS = tf.app.flags.FLAGS
            dataset = FLAGS.data
            if dataset == "image_coco":
                real_text = 'data/testdata/test_image_coco.txt'
            elif dataset == "emnlp_news":
                real_text = 'data/testdata/test_emnlp_news.txt'
            else:
                raise ValueError
            for i in range(3, 4):
                bleu = Bleu(
                    test_text=self.text_file,
                    real_text=real_text, gram=i)
                bleu.set_name(f"Bleu{i}")
                self.add_metric(bleu)
        if self.selfbleu:
            for i in range(2, 6):
                selfbleu = SelfBleu(test_text=self.text_file, gram=i)
                selfbleu.set_name(f"Selfbleu{i}")
                self.add_metric(selfbleu)

    def save_summary(self):
        # summary writer
        self.sum_writer = tf.summary.FileWriter(
            self.summary_path, self.sess.graph)
        return self.sum_writer

    def total_distance_for_seqgan(self):
        # Total average distance calculation on the test set
        self.test_data_loader.create_batches(self.test_file)
        self.fake_data_loader.create_batches(self.generator_file)
        po_sum_all = []
        true_all = []
        po_distance_all = []
        for _ in range(self.test_data_loader.num_batch):
            y_batch_s = [[0, 1] for _ in range(self.batch_size)]
            x_batch_s = self.test_data_loader.next_batch()
            feed_s = {
                self.discriminator_new.input_x: x_batch_s,
                self.discriminator_new.input_y: y_batch_s,
            }
            real_sig_distance_sum, positive_sum, po_true = self.sess.run(
                [self.discriminator_new.real_sig_distance_sum, self.discriminator_new.positive_sum, self.discriminator_new.po_true],
                feed_s)
            true_all.append(np.sum(po_true))
            po_sum_all.append(positive_sum)
            po_distance_all.append(real_sig_distance_sum)
        po_aver = np.sum(po_sum_all) / (self.test_data_loader.num_batch * self.batch_size)
        po_distance_avr = np.sum(po_distance_all) / (self.test_data_loader.num_batch * self.batch_size)

        neg_sum_all = []
        neg_distance_all = []
        neg_all = []
        for _ in range(self.test_data_loader.num_batch):
            y_batch = [[1, 0] for _ in range(self.batch_size)]
            x_batch = self.fake_data_loader.next_batch()
            feed = {
                self.discriminator_new.input_x: x_batch,
                self.discriminator_new.input_y: y_batch,
            }
            fake_sig_distance_sum, negtive_sum, neg_true = self.sess.run(
                [self.discriminator_new.fake_sig_distance_sum, self.discriminator_new.negtive_sum, self.discriminator_new.neg_true],
                feed)
            neg_all.append(np.sum(neg_true))
            neg_sum_all.append(negtive_sum)
            neg_distance_all.append(fake_sig_distance_sum)

        neg_aver = np.sum(neg_sum_all) / (self.test_data_loader.num_batch * self.batch_size)
        neg_distance_avr = np.sum(neg_distance_all) / (self.test_data_loader.num_batch * self.batch_size)

        distance_relative = po_aver - neg_aver
        distance_absolute = po_distance_avr - neg_distance_avr
        accuracy = (np.sum(true_all) + np.sum(neg_all)) / (2 * self.test_data_loader.num_batch * self.batch_size)
        print("accuracy:" + str(accuracy))
        print("relative distance：" + str(distance_relative))
        print("absolute distance：" + str(distance_absolute))

    def total_distance_for_relgan(self):
        # Total average distance calculation on the test set
        self.test_data_loader.create_batches(self.test_file)
        self.fake_data_loader.create_batches(self.generator_file)
        po_sum_all = []
        neg_sum_all = []
        true_all = []
        neg_all = []
        po_absolute_all = []
        neg_absolute_all = []
        for _ in range(self.test_data_loader.num_batch):
            real_sig_distance, fake_sig_distance, real_sig, fake_sig, fake_sig_less_index, real_sig_greater_index = self.sess.run(
                [self.discriminator_new.real_sig_distance, self.discriminator_new.fake_sig_distance,
                 self.discriminator_new.real_sig, self.discriminator_new.fake_sig, self.discriminator_new.fake_sig_less_index,
                 self.discriminator_new.real_sig_greater_index],
                feed_dict={self.generator.x_real: self.test_data_loader.next_batch(),
                           self.generator.x_fake: self.fake_data_loader.next_batch()})

            true_all.append(np.sum(real_sig_greater_index))
            neg_all.append(np.sum(fake_sig_less_index))
            po_sum_all.append(np.sum(real_sig))
            neg_sum_all.append(np.sum(fake_sig))
            po_absolute_all.append(real_sig_distance * self.batch_size)
            neg_absolute_all.append(fake_sig_distance * self.batch_size)

        po_avr = np.sum(po_sum_all) / (self.test_data_loader.num_batch * self.batch_size)
        neg_avr = np.sum(neg_sum_all) / (self.test_data_loader.num_batch * self.batch_size)
        po_absolute_avr = np.sum(po_absolute_all) / (self.test_data_loader.num_batch * self.batch_size)
        neg_absolute_avr = np.sum(neg_absolute_all) / (self.test_data_loader.num_batch * self.batch_size)

        distance_relative = po_avr - neg_avr
        distance_absolute = po_absolute_avr - neg_absolute_avr

        accuracy = (np.sum(true_all) + np.sum(neg_all)) / (2 * self.test_data_loader.num_batch * self.batch_size)
        print("accuracy:" + str(accuracy))
        print("relative distance：" + str(distance_relative))
        print("absolute distance：" + str(distance_absolute))

    def check_valid(self):
        # TODO
        pass

    @abstractmethod
    def train_oracle(self):
        pass

    def train_cfg(self):
        pass

    def train_real(self):
        pass


class Gen(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def get_nll(self):
        pass

    @abstractmethod
    def pretrain_step(self):
        pass


class Dis(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def predict(self):
        pass
