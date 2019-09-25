import json
from time import time

from models.Gan import Gan
from models.seqgan.SeqganDataLoader import DataLoader, DisDataloader,BalanceDisDataloader
from models.seqgan.SeqganDiscriminator import Discriminator
from models.seqgan.SeqganDiscriminatorNew import DiscriminatorNew
from models.seqgan.SeqganGenerator import Generator
from models.seqgan.SeqganReward import Reward
from utils.metrics.Cfg import Cfg
from utils.metrics.EmbSim import EmbSim
from utils.metrics.Nll import Nll
from utils.oracle.OracleCfg import OracleCfg
from utils.oracle.OracleLstm import OracleLstm
from utils.text_process import *
from utils.utils import *
from colorama import Fore


class Seqgan(Gan):
    def __init__(self, oracle=None):
        super().__init__()

    def init_metric(self):
        nll = Nll(data_loader=self.oracle_data_loader, rnn=self.oracle, sess=self.sess)
        self.add_metric(nll)

        # inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        # inll.set_name('nll-test')
        # self.add_metric(inll)

        # from utils.metrics.DocEmbSim import DocEmbSim
        # docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
        # self.add_metric(docsim)

    def train_discriminator(self):
        self.generate_samples()
        self.dis_data_loader.load_train_data(self.oracle_file, self.generator_file)
        for _ in range(3):
            self.dis_data_loader.next_batch()
            x_batch, y_batch = self.dis_data_loader.next_batch()
            feed = {
                self.discriminator.input_x: x_batch,
                self.discriminator.input_y: y_batch,
            }
            loss,_ = self.sess.run([self.discriminator.d_loss, self.discriminator.train_op], feed)

    def train_new_discriminator(self, epoch, stage=None):
        if stage == 'pre':
            #valid dataset
            self.dis_valid_data_loader.next_batch()
            x_batch, y_batch = self.dis_valid_data_loader.next_batch()
            feed = {
                self.discriminator_new.input_x: x_batch,
                self.discriminator_new.input_y: y_batch,
            }
            merge_summary_valid = self.sess.run(self.discriminator_new.merge_summary_valid, feed)
            # tensoboard
            self.writer.add_summary(merge_summary_valid, epoch)

            #train dataset
            for i in range(3):
                self.dis_train_data_loader.next_batch()
                x_batch, y_batch = self.dis_train_data_loader.next_batch()
                feed = {
                    self.discriminator_new.input_x: x_batch,
                    self.discriminator_new.input_y: y_batch,
                }
                if i == 2:
                    merge_summary_train, _ = self.sess.run(
                        [self.discriminator_new.merge_summary_train_ad, self.discriminator_new.train_op], feed)
                    # tensoboard
                    self.writer.add_summary(merge_summary_train, epoch)
                else:
                    self.sess.run(self.discriminator_new.train_op, feed)
        else:
            # valid
            self.dis_valid_data_loader.next_batch()
            x_batch, y_batch = self.dis_valid_data_loader.next_batch()
            feed = {
                self.discriminator_new.input_x: x_batch,
                self.discriminator_new.input_y: y_batch,
            }
            merge_summary_valid_ad = self.sess.run(self.discriminator_new.merge_summary_valid_ad, feed)
            # tensoboard
            self.writer.add_summary(merge_summary_valid_ad, epoch)
            # train
            for i in range(3):
                self.dis_train_data_loader.next_batch()
                x_batch, y_batch = self.dis_train_data_loader.next_batch()
                feed = {
                    self.discriminator_new.input_x: x_batch,
                    self.discriminator_new.input_y: y_batch,
                }
                if i==2:
                    merge_summary_train_ad, _ = self.sess.run([self.discriminator_new.merge_summary_train_ad, self.discriminator_new.train_op], feed)
                    # tensoboard
                    self.writer.add_summary(merge_summary_train_ad, epoch)
                else:
                    self.sess.run(self.discriminator_new.train_op,feed)

    def init_oracle_trainng(self, oracle=None):
        if oracle is None:
            oracle = OracleLstm(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                                hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                                start_token=self.start_token)
        self.set_oracle(oracle)

        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      l2_reg_lambda=self.l2_reg_lambda, splited_steps=self.splited_steps)
        self.set_discriminator(discriminator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)

    def train_oracle(self):
        self.init_oracle_trainng()
        self.init_metric()
        self.sess.run(tf.global_variables_initializer())

        # self.pre_epoch_num = 80
        # self.adversarial_epoch_num = 100
        # self.log = open('experiment-log-seqgan.csv', 'w')
        generate_samples(self.sess, self.oracle, self.batch_size, self.generated_num, self.oracle_file)
        generate_samples(self.sess, self.generator, self.batch_size, self.generated_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)

        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % self.ntest_pre == 0:
                generate_samples(self.sess, self.generator, self.batch_size, self.generated_num, self.generator_file)
                self.evaluate()

        print('start pre-train discriminator:')
        # self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        # self.reset_epoch()
        print('adversarial training:')
        self.reward = Reward(self.generator, .8)
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards
                }
                _ = self.sess.run(self.generator.g_updates, feed_dict=feed)
            end = time()
            self.add_epoch()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            if epoch % self.ntest == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generated_num, self.generator_file)
                self.evaluate()

            self.reward.update_params()
            for _ in range(15):
                self.train_discriminator()

    def init_cfg_training(self, grammar=None):
        oracle = OracleCfg(sequence_length=self.sequence_length, cfg_grammar=grammar)
        self.set_oracle(oracle)
        self.oracle.generate_oracle()
        self.vocab_size = self.oracle.vocab_size + 1

        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)
        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        return oracle.wi_dict, oracle.iw_dict

    def init_cfg_metric(self, grammar=None):
        cfg = Cfg(test_file=self.test_file, cfg_grammar=grammar)
        self.add_metric(cfg)

    def train_cfg(self):
        cfg_grammar = """
          S -> S PLUS x | S SUB x |  S PROD x | S DIV x | x | '(' S ')'
          PLUS -> '+'
          SUB -> '-'
          PROD -> '*'
          DIV -> '/'
          x -> 'x' | 'y'
        """

        wi_dict_loc, iw_dict_loc = self.init_cfg_training(cfg_grammar)
        with open(iw_dict_loc, 'r') as file:
            iw_dict = json.load(file)

        def get_cfg_test_file(dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))

        self.init_cfg_metric(grammar=cfg_grammar)
        self.sess.run(tf.global_variables_initializer())

        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 100
        self.log = open('experiment-log-seqgan-cfg.csv', 'w')
        generate_samples(self.sess, self.generator, self.batch_size, self.generated_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)
        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples(self.sess, self.generator, self.batch_size, self.generated_num, self.generator_file)
                get_cfg_test_file()
                self.evaluate()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num * 3):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        self.reset_epoch()
        print('adversarial training:')
        self.reward = Reward(self.generator, .8)
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards
                }
                loss, _ = self.sess.run([self.generator.g_loss, self.generator.g_updates], feed_dict=feed)
                print(loss)
            end = time()
            self.add_epoch()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generated_num, self.generator_file)
                get_cfg_test_file()
                self.evaluate()

            self.reward.update_params()
            for _ in range(15):
                self.train_discriminator()
        return

    def init_real_trainng(self, data_loc=None):
        from utils.text_process import text_precess, text_to_code
        from utils.text_process import get_tokenlized, get_word_list, get_dict
        if data_loc is None:
            data_loc = 'data/image_coco.txt'
        #self.sequence_length, self.vocab_size = text_precess(data_loc)
        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      l2_reg_lambda=self.l2_reg_lambda, splited_steps=self.splited_steps)

        discriminator_new = DiscriminatorNew(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      l2_reg_lambda=self.l2_reg_lambda, splited_steps=self.splited_steps)
        self.set_discriminator(discriminator,discriminator_new=discriminator_new)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = None
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_valid_dataloader = BalanceDisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_train_dataloader = BalanceDisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)
        valid_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        test_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        fake_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, dis_train_loader=dis_train_dataloader, dis_valid_loader=dis_valid_dataloader,oracle_loader=oracle_dataloader, fake_loader=fake_dataloader, valid_loader=valid_dataloader, test_loader=test_dataloader)
        tokens = get_tokenlized(data_loc)
        with open(self.oracle_file, 'w') as outfile:
            outfile.write(text_to_code(tokens, self.wi_dict, self.sequence_length))

    def train_real(self, data_loc=None):
        self.init_real_trainng(data_loc)
        self.init_real_metric()

        self.gen_data_loader.create_batches(self.oracle_file)
        self.valid_data_loader.create_batches(self.valid_file)
        self.sess.run(tf.global_variables_initializer())

        #++ Saver
        saver_variables = tf.global_variables()
        saver = tf.train.Saver(saver_variables)
        #++ ====================

        # summary writer
        self.writer = self.save_summary()

        if self.restore:
            restore_from = tf.train.latest_checkpoint(self.save_path)
            saver.restore(self.sess, restore_from)
            print(f"{Fore.BLUE}Restore from : {restore_from}{Fore.RESET}")
            self.epoch = self.pre_epoch_num
        else:
            print('start pre-train generator:')
            for epoch in range(self.pre_epoch_num):
                start = time()
                loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
                end = time()
                print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
                self.add_epoch()
                if epoch % self.ntest_pre == 0:
                    self.evaluate_real()

            print('start pre-train discriminator:')
            for epoch in range(self.pre_epoch_num):
                print('epoch:' + str(epoch))
                self.train_discriminator()

            print('start pre-train new discriminator')
            self.generate_samples()
            self.dis_valid_data_loader.load_train_data(self.valid_file, self.generator_file)
            self.dis_train_data_loader.load_train_data(self.oracle_file, self.generator_file)
            for epoch in range(self.pre_new_dis_num):
                print('evaluate epoch:' + str(epoch))
                self.train_new_discriminator(epoch, 'pre')
            self.total_distance_for_seqgan()

            # save pre_train
            saver.save(self.sess, os.path.join(self.save_path, 'pre_train'))

        # stop after pretrain
        if self.pretrain:
            self.evaluate_real()
            exit()
            
        print('adversarial training:')
        self.reward = Reward(self.generator, .8)
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards
                }
                loss, _ = self.sess.run([self.generator.g_loss, self.generator.g_updates], feed_dict=feed)
            end = time()
            self.add_epoch()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            if epoch % self.ntest == 0 or epoch == self.adversarial_epoch_num - 1:
                self.evaluate_real()

            self.reward.update_params()
            for _ in range(15):
                self.train_discriminator()

        print('new discriminator evaluate')
        self.dis_valid_data_loader.load_train_data(self.valid_file, self.generator_file)
        self.dis_train_data_loader.load_train_data(self.oracle_file, self.generator_file)
        for epoch in range(self.pre_new_dis_num):
            print('evaluate epoch:' + str(epoch))
            self.train_new_discriminator(epoch)

        # save all
        saver.save(self.sess, os.path.join(self.save_path, 'all_train'))
