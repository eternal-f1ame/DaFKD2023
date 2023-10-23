import logging
import traceback
import copy
import torch
from torch import nn
from torch.autograd import Variable
from model.cv.generator import GeneratorCifar, weights_init
from model.nlp.discriminator import DiscriminatorForSequenceClassification
from model.base.base_trainer import BaseTrainer
from typing import Dict, Optional, Mapping, List
from model.nlp.utils import ClassifierOutput

class CVTrainer:
    def __init__(self, model,args=None):
        self.model_g = GeneratorCifar(args.noise_dimension)
        self.model_d = model
        self.id = 0
        self.args = args
        self.model_g.apply(weights_init)

    def set_id(self, trainer_id):
        self.id = trainer_id

    def get_model_params(self):
        return (self.model_d.sharedlayer.cpu().state_dict(),self.model_d.classify.cpu().state_dict(),self.model_g.cpu().state_dict())

    def get_model_params_all(self):
        return self.model_d.cpu().state_dict()

    def set_model_params_all(self,model_parameters):
        self.model_d.load_state_dict(model_parameters)

    def set_model_params(self,model_parameters):
        model_shared_parameters,model_c_parameters,model_g_parameters = model_parameters
        self.model_g.load_state_dict(model_g_parameters)
        self.model_d.sharedlayer.load_state_dict(model_shared_parameters)
        self.model_d.classify.load_state_dict(model_c_parameters)


    def get_distillation_share_data(self, global_noise, device):
        model_g = self.model_g
        model_g.to(device)
        distillation_share_data = []
        with torch.no_grad():
            for batch_noise in global_noise:
                z = Variable(batch_noise).to(device)
                batch_share_data = model_g(z)
                fake_label = torch.tensor([0]*self.args.batch_size)
                distillation_share_data.append((batch_share_data,fake_label))
        return distillation_share_data

    def get_probability_density(self, distillation_share_data, device):
        pd_list = []
        with torch.no_grad():
            for (batch_share_data,_) in distillation_share_data:
                model_d = self.model_d
                model_d.to(device)

                output = model_d(batch_share_data.to(device))[1]
                output = output.flatten()
                pd_list.append(output)
        return pd_list

    def get_pd_log(self,client_indexes , device, ds_data, model_dict, pd_dict,num):
        tmp_model = copy.deepcopy(self.model_d)
        tmp_model.to(device)
        ds_data = ds_data.to(device)
        pd_log = None
        with torch.no_grad():
            for idx in client_indexes:
                tmp_model.sharedlayer.load_state_dict(model_dict[idx][0])
                tmp_model.classify.load_state_dict(model_dict[idx][1])

                tmp_log = tmp_model(ds_data)[0].to('cpu')

                for i in range(len(tmp_log)):
                    tmp_log[i] = tmp_log[i] * pd_dict[idx][num][i]
                if pd_log == None:
                    pd_log = tmp_log
                else:
                    pd_log += tmp_log
        return pd_log

    def weighted_ensemble_distillation(self, args, device, global_ds_dataset, client_indexes, model_dict, pd_dataset_dict,
                                       round_idx, test_global,pd_dataset_dict_t,avg_model_params):
        logging.info("################weighted_ensemble_distillation################")
        try:
            T = args.temperature
            epoch_loss = []
            teacher_acc = []
            student_acc = []
            model_d = self.model_d
            model_d.sharedlayer.load_state_dict(avg_model_params[0])
            model_d.classify.load_state_dict(avg_model_params[1])

            model_d.to(device)
            if args.client_optimizer == "sgd":
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.es_lr)
            else:
                d_optimizer = torch.optim.Adam(self.model_d.parameters(),lr=args.es_lr)
                d1_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model_d.sharedlayer.parameters()),
                                             lr=args.es_lr, weight_decay=args.wd, amsgrad=True)
                d2_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model_d.classify.parameters()),
                                             lr=args.es_lr, weight_decay=args.wd, amsgrad=True)
            criterion = nn.KLDivLoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer,
                                                           mode="min",
                                                           factor=0.2,
                                                           patience=2)

            for epoch in range(args.ed_epoch):
                for i in  range(len(global_ds_dataset)):
                    ds_data, _ = global_ds_dataset[i]

                    pd_dict = dict()
                    pd_dict = pd_dataset_dict

                    teacher_log = self.get_pd_log(client_indexes, device, ds_data, model_dict, pd_dict,i).to(device)

                    ds_data = ds_data.to(device)
                    d_optimizer.zero_grad()



                    avg_log = model_d(ds_data)[0]
                    loss = (T ** 2) * criterion(
                            torch.nn.functional.log_softmax(avg_log / T, dim=1),
                            torch.nn.functional.softmax(teacher_log / T, dim=1)
                        )
                    loss.backward()
                    d_optimizer.step()

                    epoch_loss.append(loss.item())
                scheduler.step(sum(epoch_loss) / len(epoch_loss))

                logging.info('Epoch: {}\tED_Loss: {:.6f}'.format(epoch, sum(epoch_loss) / len(epoch_loss)))
                logging.info('ED_Loss: {:.6f}'.format(loss.item()))
                
            for i in range(len(test_global)):
                test_data,label = test_global[i]

                pd_dict = dict()
                pd_dict = pd_dataset_dict_t

                teacher_log = self.get_pd_log(client_indexes, device, test_data, model_dict, pd_dict,i).to(device)
                _, teacher_predicted = torch.max(teacher_log, -1)
                test_data = test_data.to(device)

                avg_log = model_d(test_data)[0]
                _, student_predicted = torch.max(avg_log, -1)
                s_num = 0
                t_num = 0
                for s, t, l in zip(student_predicted, teacher_predicted, label):
                    if s == l:
                        s_num += 1
                    if t == l:
                        t_num += 1
                s_acc = s_num / len(label)
                student_acc.append(s_acc)
                t_acc = t_num / len(label)
                teacher_acc.append(t_acc)

            if round_idx % 1 == 0:
                logging.info('student_acc: {},teacher_acc: {}'.format(sum(student_acc) / len(student_acc),sum(teacher_acc) / len(teacher_acc)))
           
            return self.get_model_params()
            
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error("Error!")
            return avg_model_params

    def train(self, train_data, device, args, round_idx):
        logging.debug("-------gan model actually train------")
        try:
            model_g = self.model_g
            model_d = self.model_d

            model_g.to(device)
            model_d.to(device)
            model_g.train()
            model_d.train()

            noise_dimension = args.noise_dimension

            # optimizer
            criterion1 = nn.CrossEntropyLoss().to(device)
            criterion2 = nn.BCELoss().to(device)  # binary cross entropy

            betas = (0.5, 0.99)
            if args.client_optimizer == "sgd":
                g_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model_g.parameters()), lr=args.lr)
                d_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model_d.parameters()), lr=args.lr)
            else:
                g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model_g.parameters()), lr=args.lr,
                                               betas=betas, weight_decay=args.wd, amsgrad=True)
                d_optimizer = torch.optim.Adam(self.model_d.parameters(), lr=args.lr)
                                               
                d1_optimizer = torch.optim.Adam(self.model_d.sharedlayer.parameters(), lr=args.lr)
                d2_optimizer = torch.optim.Adam(self.model_d.classify.parameters(), lr=args.lr)
                d3_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model_d.discriminator.parameters()), lr=args.lr,
                                               betas=betas, weight_decay=args.wd, amsgrad=True)

                scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer,
                                                           mode="min",
                                                           factor=0.2,
                                                           patience=2)

            g_epoch_loss = []
            d_epoch_loss = []
            img_num_list = []
            epoch_loss = []

            for epoch in range(args.epochs):
                for(img, labels) in train_data:
                    batch_loss = []
                    g_batch_loss = []
                    d_batch_loss = []
                    
                    num_img = img.size(0)
                    if epoch == 0:
                        img_num_list.append(num_img)
                    
                    img, labels = img.to(device), labels.to(device)
                    model_d.zero_grad()

                    log_probs,real_out = model_d(img)
                    loss1 = criterion1(log_probs, labels)
                    batch_loss.append(loss1.item())

                    real_img = Variable(img).to(device)
                    real_label = Variable(torch.ones(num_img)).to(device)
                    real_out = real_out.flatten()
                    d_loss_real = criterion2(real_out, real_label)
                    real_scores = real_out 
                    noise = Variable(torch.randn(num_img, noise_dimension,1,1)).to(device)
                    fake_img = model_g(noise).detach()
                    fake_label = Variable(torch.zeros(num_img)).to(device)
                    fake_out = model_d(fake_img)[1]
                    fake_out = fake_out.flatten()
                    d_loss_fake = criterion2(fake_out, fake_label)
                    fake_scores = fake_out  # closer to 0 means better

                    d_loss_gp = self.gradient_penalty(model_d, device, real_img, fake_img, num_img)
                    # d_loss_gp = 0
                    if round_idx < 3 and epoch < 5:
                        d_loss = d_loss_real + d_loss_fake + d_loss_gp
                        d_batch_loss.append(d_loss.item())

                    else:
                        d_loss = 0
                        d_batch_loss.append(0)


                    loss_all = loss1+d_loss                    

                    loss_all.backward()

                    d_optimizer.step()

                    noise = Variable(torch.randn(num_img, noise_dimension,1,1)).to(device)
                    fake_img = model_g(noise)
                    output = model_d(fake_img)[1]
                    output = output.flatten()
                    g_loss = criterion2(output, real_label)
                    if round_idx < 5 and epoch < 5:
                        g_optimizer.zero_grad()
                        g_loss.backward()
                        g_optimizer.step()
                    g_batch_loss.append(g_loss.item())


                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                g_epoch_loss.append(sum(g_batch_loss) / len(g_batch_loss))
                d_epoch_loss.append(sum(d_batch_loss) / len(d_batch_loss))
                scheduler1.step(sum(epoch_loss) / len(epoch_loss))

                if epoch == args.epochs-1:
                    logging.info(
                    'Client Index = {}\tEpoch: {}\timg_nums: {}\tEpoch_Loss: {:.6f}\tGenerator_Loss: {:.6f}\tDiscriminator_Loss: {:.6f}'
                    '\tDiscriminator_Real_Output: {:.6f}\tDiscriminator_Fake_Output: {:.6f}'.format(
                        self.id, epoch, sum(img_num_list),sum(batch_loss) / len(batch_loss),
                       sum(g_epoch_loss) / len(g_epoch_loss), sum(d_epoch_loss) / len(d_epoch_loss),
                        real_scores.data.mean(), fake_scores.data.mean()))

        except Exception as e:
            logging.error(traceback.format_exc())

                
    def test(self, test_data, device, args):
        model_d = self.model_d
        model_d.to(device)
        model_d.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)

                # print(target)
                target = target.to(device)
                pred = model_d(x)[0]
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

    def gradient_penalty(self, discriminator, device, real_img, fake_img, batch_size, GP_LAMBDA=0.3):
        """
        :param discriminator:
        :param device:
        :param real_img:[batch_size,1,28,28]
        :param fake_img:[batch_size,1,28,28]
        :param batch_size:
        :param GP_LAMBDA:
        :return:
        """

        fake_img = fake_img
        real_img = real_img

        alpha = torch.rand(batch_size, 1).to(device)
        alpha = alpha.expand_as(real_img.reshape(batch_size, -1))
        alpha = alpha.reshape(real_img.shape)

        interpolates = alpha * real_img + ((1 - alpha) * fake_img)
        interpolates.requires_grad_()

        disc_interpolates = discriminator(interpolates)[1]

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones_like(disc_interpolates),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * GP_LAMBDA

        return gp

class NLPTrainer(BaseTrainer):
    """GAN trainer for sequence classification tasks"""

    def __init__(
        self,
        config: Dict,
        discriminator: DiscriminatorForSequenceClassification,
        generator: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        ce_ignore_index: Optional[int] = -100,
        device=None,
        save_path="../weights/best_model.pth",
    ):
        self.config = config
        self.model = discriminator
        self.hidden_size = self.model.encoder.config.hidden_size
        self.generator = generator
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = device
        self._define_optimizer()
        self._define_scheduler()
        self.model.to(self.device)
        self.generator.to(self.device)
        self.compute_metrics = (
            compute_multi_label_metrics(config.get("multi-label-trh", 0.5))
            if config.get("multi-label", False)
            else compute_clf_metrics
        )
        self.best_valid_score = 0
        self.model_path = save_path

    def train_mode_on(self):
        self.model.train()
        self.generator.train()

    @staticmethod
    def _train_logging(
        log_env: Optional[Mapping] = None,
        info: Optional[Mapping] = None,
        dloss: torch.FloatTensor = None,
        gloss: torch.FloatTensor = None,
        **kwargs,
    ):
        if log_env is not None:
            log_env["train/discriminator_loss"].log(dloss.item())
            log_env["train/generator_loss"].log(gloss.item())

        return

    def _valid_logging(
        self,
        log_env: Optional[Mapping] = None,
        info: Optional[Mapping] = None,
        output: Optional[ClassifierOutput] = None,
        **kwargs,
    ):
        if info["overall_f1"] >= self.best_valid_score:
            self.best_valid_score = info["overall_f1"]
            torch.save(self.model.state_dict(), self.model_path)
            print("Best model saved!")
        if log_env is not None:
            log_env["valid/discriminator_loss"].log(info["loss"])
            log_env["valid/discriminator_accuracy"].log(info["overall_accuracy"])
            log_env["valid/f1"].log(info["overall_f1"])
            log_env["valid/precision"].log(info["overall_precision"])
            log_env["valid/recall"].log(info["overall_recall"])
            log_env["valid/detailed_metrics"].log(info)

    def on_train_start(self):
        train_info = {"total_train_loss": 0, "total_generator_loss": 0}
        return train_info

    def on_train_end(self, info: Dict, verbose: Optional[bool] = True):
        info["total_train_loss"] /= len(self.train_dataloader)
        info["total_generator_loss"] /= len(self.train_dataloader)
        if verbose:
            print(f"\tTrain loss discriminator: {info['total_train_loss']:.3f}")
            print(f"\tTrain loss generator: {info['total_generator_loss']:.3f}")

    def on_epoch_end(self, **kwargs):
        kwargs["train_info"]["total_train_loss"] += kwargs["epoch_info"]["loss"]
        kwargs["train_info"]["total_generator_loss"] += kwargs["epoch_info"]["generator_loss"]

    def _define_optimizer(self):
        # discriminator optimizer
        if self.config["frozen_backbone"]:
            self.model.freeze_backbone()
        m_vars = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Trainable layers {len(m_vars)}")
        self.optimizer = AdamW(m_vars, lr=self.config["lr_discriminator"])
        # generator optimizer
        g_vars = [v for v in self.generator.parameters()]
        self.generator_optimizer = torch.optim.AdamW(g_vars, lr=self.config["lr_generator"])

    def _define_scheduler(self):
        _default_scheduler = get_constant_schedule_with_warmup
        if self.config["apply_scheduler"]:
            # discriminator scheduler
            train_size = self.config["num_train_examples"]
            batch_size = self.config["batch_size"]
            epochs = self.config["num_train_epochs"]
            self.config["num_train_steps"] = int(train_size / batch_size * epochs)
            self.config["num_warmup_steps_d"] = int(
                self.config["num_train_steps"] * self.config["warmup_proportion_d"]
            )
            self.scheduler = _default_scheduler(
                self.optimizer, num_warmup_steps=self.config["num_warmup_steps_d"]
            )
            # generator scheduler
            self.config["num_warmup_steps_g"] = int(
                self.config["num_train_steps"] * self.config["warmup_proportion_g"]
            )
            self.generator_scheduler = _default_scheduler(
                self.generator_optimizer, num_warmup_steps=self.config["num_warmup_steps_g"]
            )


class NLPTrainerSequenceClassification(NLPTrainer):
    """GAN trainer for sequence classification tasks"""

    def training_step(self, batch: Mapping[str, torch.Tensor], log_env: Optional[Dict] = None):
        batch = self._prepare_inputs(batch)
        batch_size = batch["input_ids"].shape[0]
        seq_len = batch["input_ids"].shape[1]

        output = self.model(
            input_ids=batch["input_ids"],
            input_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids", None),
            labels=batch["labels"],
            labeled_mask=batch["labeled_mask"],
        )
        noise = get_noise(
            (batch_size, self.config["noise_size"]),
            dist=self.config.get("noise_type", "uniform"),
            range=self.config.get("noise_range", (-2, 2)),
            device=self.device,
        )
        gen_states = self.generator(noise)

        fake_output = self.model(external_states=gen_states, input_mask=batch["attention_mask"])

        # Generator loss estimation
        cheat_loss = -1 * torch.mean(torch.log(1 - fake_output.fake_probs + self.config["epsilon"]))
        feat_sim_loss = torch.mean(
            torch.pow(
                torch.mean(output.hidden_states, dim=0) - torch.mean(fake_output.hidden_states, dim=0), 2
            )
        )
        generator_loss = cheat_loss + feat_sim_loss

        # Discriminator loss estimation
        unsup_fake_loss = fake_output.fake_loss
        unsup_real_loss = -torch.mean(torch.log(1 - output.fake_probs + self.config["epsilon"]))
        discriminator_loss = output.loss + unsup_fake_loss + unsup_real_loss

        self._train_logging(log_env, dloss=discriminator_loss, gloss=generator_loss)

        self.generator_optimizer.zero_grad()
        self.optimizer.zero_grad()
        generator_loss.backward(retain_graph=True)
        discriminator_loss.backward()
        self.generator_optimizer.step()
        self.optimizer.step()

        if self.config["apply_scheduler"]:
            self.scheduler.step()
            self.generator_scheduler.step()

        epoch_info = dict(loss=discriminator_loss.item(), generator_loss=generator_loss.item())
        return epoch_info

    @torch.no_grad()
    def predict(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        label_names: List[str],
        label_col: str = "labels",
    ) -> Dict:
        total_loss = 0
        predictions = []
        model.eval()
        for step, batch in enumerate(data_loader):
            batch = self._prepare_inputs(batch)
            output = model(
                input_ids=batch["input_ids"],
                input_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids", None),
                labels=batch[label_col],
            )
            predictions.append(output.logits.cpu().detach().numpy())
            total_loss += output.loss
        result = self.compute_metrics(
            predictions=np.vstack(predictions),
            labels=np.vstack(data_loader.dataset[label_col]),
            label_names=label_names,
        )
        result["loss"] = total_loss / len(data_loader)
        return result

