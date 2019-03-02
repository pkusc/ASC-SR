import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from trainertest import testTrainer
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    print("testset ", args.testset)
    if args.testtime:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        print("prepared to train...")
        t = testTrainer(args, loader, model, loss, checkpoint)
        t.train()
    if args.data_test == 'Demo':
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        print("prepared to demo...")
        t = Trainer(args, loader, model, loss, checkpoint)
        t.demo()
    else:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        print("prepared to train...")
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()
    checkpoint.done()

