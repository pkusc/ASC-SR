import os
import math
from decimal import Decimal

import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm
from data.common import get_patch
import torchvision
class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, self.args.scale[idx_scale])
            loss = self.loss(sr, hr)
            if loss.data.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.data.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()

        temp_acc = 0
        for idx_scale, scale in enumerate(self.scale):
            eval_acc = 0
            self.loader_test.dataset.set_scale(idx_scale)
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):
                filename = filename[0]
                no_eval = isinstance(hr[0], int)

                lr_1 = torch.squeeze(lr).permute(2, 1, 0)
                hr_1 = torch.squeeze(hr).permute(2, 1, 0)

                lr_s3, hr_s3 = get_patch(lr_1, hr_1, 400, scale)
                lr_s3 = torch.unsqueeze(lr_s3.permute(2, 1, 0), 0)
                hr_s3 = torch.unsqueeze(hr_s3.permute(2, 1, 0), 0)


                if no_eval:
                    with torch.no_grad():
                        lr = self.prepare([lr])[0]
                        lr_s3 = self.prepare([lr_s3])[0]
                        #lr_s9 = self.prepare([lr_s9])[0]
                else:
                    with torch.no_grad():
                       # lr, hr = self.prepare([lr, hr])
                        lr_s3, hr_s3 = self.prepare([lr_s3, hr_s3])
                        #lr_s9, hr_s9 = self.prepare([lr_s9, hr_s9])

                with torch.no_grad():
                    #sr = self.model(lr, idx_scale)
                    #sr = utility.quantize(sr, self.args.rgb_range)
                    sr_s3 = self.model(lr_s3, idx_scale)
                    sr_s3 = utility.quantize(sr_s3, self.args.rgb_range)
                #sr_s3 = self.model(lr_s3, idx_scale)
                #sr_s3 = utility.quantize(sr_s3, self.args.rgb_range)




                #sr = self.model(lr, idx_scale)
                #sr = utility.quantize(sr, self.args.rgb_range)


                #save_list = [sr]
                save_list = [sr_s3]
                if not no_eval:
                    eval_acc += utility.calc_psnr(
                        #sr, hr, scale, self.args.rgb_range,
                        sr_s3, hr_s3, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )


                #sr = self.model(lr, idx_scale)
                #sr = utility.quantize(sr, self.args.rgb_range)


                #save_list = [sr]
                        #sr, hr, scale, self.args.rgb_range,
                    #save_list.extend([lr, hr])
                save_list.extend([lr_s3, hr_s3])

                if self.args.save_results:
                    self.ckp.save_results(filename, save_list, scale)



            self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
            best = self.ckp.log.max(0)
            
            print("mini 3:\n")
            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} from epoch {})'.format(
                    self.args.data_test, scale, self.ckp.log[-1, idx_scale],
                    best[0][idx_scale], best[1][idx_scale] + 1
                )
            )
        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def demo(self):
        tqdm_test = tqdm(self.loader_test, ncols=80)
        for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):
            with torch.no_grad():
                p = 40
                size_lr = lr.size()
                sr = torch.zeros([1, 3, size_lr[2]*4, size_lr[3]*4])
                if True:
                    srow = 0
                    scol = 0
                    srrow = p
                    srcol = p
                    while srow + p < size_lr[2]:
                        scol = 0
                        srcol = p
                        while scol + p < size_lr[3]:
                            lr_ = lr[:, :, srow: srow + p, scol: scol + p]
                            lr_, hr = self.prepare([lr_, hr])
                            sr_ = self.model(lr_, 0)
                            sr_ = utility.quantize(sr_, self.args.rgb_range)
                            sr[0, :, srrow: srrow + p*2, srcol: srcol + p*2] = sr_[0, :, p:p*3, p:p*3]
                            if srow == 0:
                                sr[0, :, 0: p, srcol: srcol + p*2] = sr_[0, :, 0: p, p:p*3]
                            if scol == 0:
                                sr[0, :, srrow: srrow+p*2, 0:p]=sr_[0, :, p: p*3, 0:p]
                            if srow ==0 and scol == 0:
                                sr[0, :, 0:p, 0:p] = sr_[0, :, 0:p, 0:p]
                            scol += int(p/2)
                            srcol += p*2
                        sz_col = min(p, size_lr[3])
                        lr_ = lr[:, :, srow: srow + p, size_lr[3]-sz_col:]
                        lr_, hr = self.prepare([lr_, hr])
                        sr_ = self.model(lr_, 0)
                        sr_ = utility.quantize(sr_, self.args.rgb_range)
                        sr[0, :, srrow: srrow + p*2, srcol:] = sr_[0, :, p:p*3, sz_col*4-size_lr[3]*4+srcol:]
                        if srow == 0:
                            sr[0, :, 0: p, srcol:] = sr_[0, :, 0: p, sz_col*4-size_lr[3]*4+srcol:]
                        srow += int(p/2)
                        srrow += p*2
                    scol = 0
                    srcol = p
                    sz_row = min(p, size_lr[2])
                    while scol + p < size_lr[3]:
                        lr_ = lr[:, :, size_lr[2]-sz_row:, scol: scol + p]
                        lr_, hr = self.prepare([lr_, hr])
                        sr_ = self.model(lr_, 0)
                        sr_ = utility.quantize(sr_, self.args.rgb_range)
                        sr[0, :, srrow: , srcol: srcol + p*2] = sr_[0, :, sz_row*4-size_lr[2]*4+srrow: , p:p*3]
                        if scol == 0:
                            sr[0, :, srrow:, 0:p]=sr_[0, :, sz_row*4-size_lr[2]*4+srrow:, 0:p]
                        scol += int(p/2)
                        srcol += p*2
                    sz_col = min(p, size_lr[3])
                    lr_ = lr[:, :, size_lr[2]-sz_row:, size_lr[3]-sz_col:]
                    lr_, hr = self.prepare([lr_, hr])
                    sr_ = self.model(lr_, 0)
                    sr_ = utility.quantize(sr_, self.args.rgb_range)
                    sr[0, :, srrow: , srcol:] = sr_[0, :, sz_row*4-size_lr[2]*4+srrow:, sz_col*4-size_lr[3]*4+srcol:]
            save_list = [sr]
            self.ckp.save_results(filename, save_list, 4)        

    def prepare(self, l, volatile=False):
        def _prepare(idx, tensor):
            if not self.args.cpu: tensor = tensor.cuda()
            if self.args.precision == 'half': tensor = tensor.half()
            # Only test lr can be volatile
            if volatile == False or idx != 0:
                ret = Variable(tensor)
            if volatile and idx == 0:
                with torch.no_grad():
                    ret = Variable(tensor)
            return ret
            #return Variable(tensor, volatile=(volatile and idx==0))
           
        return [_prepare(i, _l) for i, _l in enumerate(l)]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

