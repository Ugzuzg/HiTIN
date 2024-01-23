# Display cpv code which has the highest probability,  in reply to inputted custom description (in development) 

#!/usr/bin/env python
# coding:utf-8
from torch.utils.data import DataLoader
from data_modules.collator import Collator
from data_modules.dataset import ClassificationDataset
import helper.logger as logger
from models.model import HiAGM
import torch
from helper.configure import Configure
import os
from data_modules.data_loader import data_loaders
from data_modules.vocab import Vocab
from train_modules.criterions import ClassificationLoss
from train_modules. trainer import Trainer
from helper.utils import load_checkpoint, save_checkpoint
from helper.arg_parser import get_args

import time
import random
import numpy as np
import pprint
import warnings

from transformers import AutoTokenizer
from helper.lr_schedulers import get_linear_schedule_with_warmup
from helper.adamw import AdamW

warnings.filterwarnings("ignore")


def set_optimizer(config, model):
    """
    :param config: helper.configure, Configure Object
    :param model: computational graph
    :return: torch.optim
    """
    params = model.optimize_params_dict()
    if config.train.optimizer.type == 'Adam':
        return torch.optim.Adam(lr=config.learning_rate,  # using args
            # lr=config.train.optimizer.learning_rate,
                                params=params,
                                weight_decay=args.l2rate)
    else:
        raise TypeError("Recommend the Adam optimizer")


def play(config, args):
    """
    :param config: helper.configure, Configure Object
    """
    # loading corpus and generate vocabulary
    corpus_vocab = Vocab(config,
                         min_freq=5,
                         max_size=70000)
    if config.text_encoder.type == "bert" or config.text_encoder.type == "roberta":
        tokenizer = AutoTokenizer.from_pretrained(config.text_encoder.bert_model_dir)
    else:
        tokenizer = None

    # get data
    # train_loader, dev_loader, test_loader = data_loaders(config, corpus_vocab, tokenizer=tokenizer)

    # build up model
    hiagm = HiAGM(config, corpus_vocab, model_type=config.model.type, model_mode='TRAIN')

    hiagm.to(config.train.device_setting.device)

    # Code for counting parameters
    # from thop import clever_format
    # print(hiagm)
    # def count_parameters(model):
    #     total = sum(p.numel() for p in model.parameters())
    #     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     return total, trainable
    #
    # total_params, trainable_params = count_parameters(hiagm)
    # total_params, trainable_params = clever_format([total_params, trainable_params], "%.4f")
    # print("Total num of parameters: {}. Trainable parameters: {}".format(total_params, trainable_params))
    # sys.exit()

    # Define training objective & optimizer
    criterion = ClassificationLoss(os.path.join(config.data.data_dir, config.data.hierarchy),
                                   corpus_vocab.v2i['label'],
                                   # recursive_penalty=config.train.loss.recursive_regularization.penalty,
                                   recursive_penalty=args.hierar_penalty,  # using args
                                   recursive_constraint=config.train.loss.recursive_regularization.flag)
    # get epoch trainer
    trainer = Trainer(model=hiagm,
                      criterion=criterion,
                      optimizer=None,
                      scheduler=None,
                      vocab=corpus_vocab,
                      config=config)

    # set origin log
    best_epoch = [-1, -1]
    best_performance = [0.0, 0.0]
    '''
        ckpt_dir
            begin-time_dataset_model
                best_micro/macro-model_type-training_params_(tin_params)
                                            
    '''
    # model_checkpoint = config.train.checkpoint.dir
    model_checkpoint = os.path.join(args.ckpt_dir, args.begin_time + config.train.checkpoint.dir)  # using args
    model_name = config.model.type
    if config.structure_encoder.type == "TIN":
        model_name += '_' + str(args.tree_depth) + '_' + str(args.hidden_dim) + '_' + args.tree_pooling_type + '_' + str(args.final_dropout) + '_' + str(args.hierar_penalty)
    wait = 0

    # loading previous checkpoint
    dir_list = os.listdir(model_checkpoint)
    dir_list.sort(key=lambda fn: os.path.getatime(os.path.join(model_checkpoint, fn)))
    print(dir_list)
    latest_model_file = ''
    for model_file in dir_list[::-1]:  # best or latest ckpt
        if model_file.startswith('best'):
            continue
        else:
            latest_model_file = model_file
            break
    if os.path.isfile(os.path.join(model_checkpoint, latest_model_file)):
        logger.info('Loading Previous Checkpoint...')
        logger.info('Loading from {}'.format(os.path.join(model_checkpoint, latest_model_file)))
        best_performance, config = load_checkpoint(model_file=os.path.join(model_checkpoint, latest_model_file),
                                                    model=hiagm,
                                                    config=config,
                                                    optimizer=optimizer)
        logger.info('Previous Best Performance---- Micro-F1: {}%, Macro-F1: {}%'.format(
            best_performance[0], best_performance[1]))

    trainer.model.eval()

    lines = ['{"token": ["Marché de travaux d\'entretien en plomberie des parties communes et des logements habites des cites de 13 habitat", "something else"]}']
    eval_dataset = ClassificationDataset(config, corpus_vocab, on_memory=True, corpus_lines=lines, tokenizer=tokenizer, mode="EVAL")
    data_loader = DataLoader(eval_dataset, batch_size=1, collate_fn=Collator(config, corpus_vocab))
    for batch in data_loader:
        logits = trainer.model(batch)
        predict_results = torch.sigmoid(logits).cpu().tolist()
        print(len(predict_results[0]), corpus_vocab.i2v['label'])
        print(corpus_vocab.i2v['label'][np.argmax(predict_results[0])])

    return


if __name__ == "__main__":
    args = get_args()
    pprint.pprint(vars(args))
    configs = Configure(config_json_file=args.config_file)
    configs.update(vars(args))

    if configs.train.device_setting.device == 'cuda':
        os.system('CUDA_VISIBLE_DEVICES=' + str(configs.train.device_setting.visible_device_list))
    else:
        os.system("CUDA_VISIBLE_DEVICES=''")

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_start_method('spawn')

    logger.Logger(configs)

    # if not os.path.isdir(configs.train.checkpoint.dir):
    #     os.mkdir(configs.train.checkpoint.dir)

    # train(config)
    play(configs, args)
