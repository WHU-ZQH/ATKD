import torch
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother
import wandb
from enum import Enum
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import transformers

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class KLMethod(Enum):
    Forward = 1
    Reverse = 2
    JSD = 3
    DKD = 4
    SeqKD=5
    TVD=6
    ATKD=7


KL_METHOD_MAP = {
    "forward": KLMethod.Forward,
    "reverse": KLMethod.Reverse,
    "jsd": KLMethod.JSD,
    "dkd": KLMethod.DKD,
    "tvd": KLMethod.TVD,
    "atkd": KLMethod.ATKD,
    "seqkd": KLMethod.SeqKD,
}

eval_cnt = 0

class DistillTrainer(Trainer):
    def __init__(self, teacher_model, copy_model, assistant_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs["args"]
        self.teacher_model = teacher_model
        self.generator = Generator(
            self.model, self.teacher_model, self.tokenizer, args.max_propose_num
        )
        self.train_step_cnt = 0

        self.mode = args.mode
        self.online_eval_interval = args.online_eval_interval
        self.online_update_interval = args.online_update_interval
        self.beta=args.beta
        self.alpha=args.alpha
        self.topk=args.topk
        self.nckd_percent=args.nckd_percent
        self.setting=args.setting
        self.sample_steps = []
        self.copy_model=copy_model
        self.assistant_model=assistant_model

        self.sample_source = args.sample_source
        self.kl_method = KL_METHOD_MAP[args.kl_method]

    def training_step(self, model, inputs):
        max_new_tokens = 256
        self.train_step_cnt += 1
        student_temperature = 1.0
        teacher_temperature = 1.0
        student_token_ratio=0.5
        student_request_ratio=0.5

        if self.kl_method == KLMethod.JSD:
            fwd_loss_ratio = 0.9

        if self.sample_source == "student":
            try:
                self.copy_model.load_state_dict(model.module.state_dict())
            except:
                self.copy_model.load_state_dict(model.state_dict())
            generated_ids, _ = self.get_generated_ids(
                self.copy_model,
                self.tokenizer,
                inputs["prompt_input_ids"],
                inputs["prompt_attention_mask"],
                max_new_tokens,
                False,
            )
            generated_ids = generated_ids.clone().detach()
            prompt_len = inputs["prompt_input_ids"].shape[-1]
            attention_mask = generated_ids != self.tokenizer.pad_token_id
            output_mask = generated_ids[..., :] == self.tokenizer.pad_token_id
            output_mask[..., :prompt_len-1] = True
        elif self.sample_source in ["mix_request_teacher", "mix_request_gt"]:
            if random.random() < student_request_ratio:
                try:
                    self.copy_model.load_state_dict(model.module.state_dict())
                except:
                    self.copy_model.load_state_dict(model.state_dict())
                generated_ids, _ = self.get_generated_ids(
                    self.copy_model,
                    self.tokenizer,
                    inputs["prompt_input_ids"],
                    inputs["prompt_attention_mask"],
                    max_new_tokens,
                    False,
                )
                generated_ids = generated_ids.clone().detach()
                prompt_len = inputs["prompt_input_ids"].shape[-1]
                attention_mask = generated_ids != self.tokenizer.pad_token_id
                output_mask = generated_ids[..., :] == self.tokenizer.pad_token_id
                output_mask[..., :prompt_len-1] = True
            else:
                generated_ids = inputs["input_ids"]
                labels=inputs["labels"]
                attention_mask = inputs["attention_mask"]
                output_mask = (inputs["labels"][..., :] == IGNORE_TOKEN_ID)|(inputs["labels"][..., :] == self.tokenizer.pad_token_id)
        elif self.sample_source == "mix_token":
            max_new_tokens = 128
            try:
                self.copy_model.load_state_dict(model.module.state_dict())
            except:
                self.copy_model.load_state_dict(model.state_dict())
            generated_ids = self.get_mix_generated_ids(
                self.copy_model,
                self.teacher_model,
                self.tokenizer,
                inputs["prompt_input_ids"],
                inputs["prompt_attention_mask"],
                max_new_tokens,
                student_token_ratio
            )
            generated_ids = generated_ids.clone().detach()
            prompt_len = inputs["prompt_input_ids"].shape[-1]
            attention_mask = generated_ids != self.tokenizer.pad_token_id
            output_mask = generated_ids[..., :] == self.tokenizer.pad_token_id
            output_mask[..., :prompt_len-1] = True
        else:
            generated_ids = inputs["input_ids"]
            labels=inputs["labels"]
            attention_mask = inputs["attention_mask"]
            output_mask = (inputs["labels"][..., :] == IGNORE_TOKEN_ID)|(inputs["labels"][..., :] == self.tokenizer.pad_token_id)
            
        # get student/teacher logits
        student_logits = self.get_logits(model, generated_ids, attention_mask)
        student_logits = student_logits.float()

        # seq-level KD
        if self.kl_method == KLMethod.SeqKD:
            labels=torch.where(labels[...,:]==self.tokenizer.pad_token_id, torch.full_like(labels, -100), labels)
            loss = self.cross_entropy(
                student_logits / student_temperature,
                labels
            )

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            loss.backward()
            return loss.detach()

        # other KD methods
        with torch.no_grad():
            teacher_logits = self.get_logits(
                self.teacher_model, generated_ids, attention_mask)
            teacher_logits = teacher_logits.float()

        # calculate loss
        if self.kl_method == KLMethod.Forward:
            student_logits, teacher_logits=student_logits[...,:-1,:].contiguous(), teacher_logits[...,:-1,:].contiguous()
            output_mask=output_mask[...,1:]
            loss = self.get_kl(
                student_logits / student_temperature,
                teacher_logits / teacher_temperature,
                output_mask
            )
        elif self.kl_method == KLMethod.Reverse:
            student_logits, teacher_logits=student_logits[...,:-1,:].contiguous(), teacher_logits[...,:-1,:].contiguous()
            output_mask=output_mask[...,1:]
            loss = self.get_kl(
                teacher_logits / teacher_temperature,
                student_logits / student_temperature,
                output_mask
            )
        elif self.kl_method == KLMethod.TVD:
            student_logits, teacher_logits=student_logits[...,:-1,:].contiguous(), teacher_logits[...,:-1,:].contiguous()
            output_mask=output_mask[...,1:]
            loss = self.get_tvd(
                teacher_logits / teacher_temperature,
                student_logits / student_temperature,
                output_mask
            )
        elif self.kl_method == KLMethod.ATKD:
            loss = self.atkd(
                student_logits, 
                teacher_logits, generated_ids, 
                student_temperature, 
                teacher_temperature, 
                output_mask
            )
        elif self.kl_method == KLMethod.DKD:
            loss=self.dkd_loss(
                student_logits,
                teacher_logits,
                generated_ids, 
                student_temperature, 
                teacher_temperature, 
                output_mask
            )
        elif self.kl_method == KLMethod.JSD:
            student_logits, teacher_logits=student_logits[...,:-1,:].contiguous(), teacher_logits[...,:-1,:].contiguous()
            output_mask=output_mask[...,1:]
            reverse_loss = self.get_kl(
                teacher_logits / teacher_temperature,
                student_logits / student_temperature,
                output_mask
            )
            fwd_loss = self.get_kl(
                student_logits / student_temperature,
                teacher_logits / teacher_temperature,
                output_mask
            )
            loss = fwd_loss_ratio * fwd_loss + \
                (1 - fwd_loss_ratio) * reverse_loss

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        return loss.detach()

    def log(self, logs):
        # Remove the 'loss' entry with value 0 before calling the superclass method
        if 'loss' in logs and logs['loss'] == -1:
            del logs['loss']

        # Call the original `log` method of the `Trainer` class
        super().log(logs)

    def get_train_dataloader(self):
        # Create custom DataLoader with shuffle set to False
        shuffle = False if self.mode == "online" else True
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "shuffle": shuffle,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

    ###################### Helper Functions #############################

    def soft_cross_entropy(self, predicts, targets, padding_mask):
        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        entropy = -targets_prob * predict_log_prob
        expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
        entropy.masked_fill_(expand_mask, 0)
        mean_entropy = entropy.sum() / (~padding_mask).sum()
        return mean_entropy  

    def get_kl(self, predicts, targets, padding_mask, reduce=True):
        kl_loss = torch.nn.KLDivLoss(reduction="none", log_target=True)
        predict_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.log_softmax(targets, dim=-1)
        output = kl_loss(predict_prob, targets_prob)
        if reduce:
            expand_mask = padding_mask.unsqueeze(-1).expand_as(output)
            output.masked_fill_(expand_mask, 0)
            mean_output = output.sum() / (~padding_mask).sum()
            return mean_output
        else:
            return output
    
    def get_tvd(self, s_logits, t_logits, padding_mask):
        s_logits = torch.nn.functional.softmax(s_logits, dim=-1)
        t_logits = torch.nn.functional.softmax(t_logits, dim=-1)
        sel_mask = padding_mask[:, :, None].expand_as(s_logits)
        vocab_size = s_logits.size(-1)
        s_logits_slct = torch.masked_select(s_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()
        loss_tvd = (0.5 * torch.abs(s_logits_slct-t_logits_slct)).sum(dim=-1).mean()
        return loss_tvd


    def get_logits(self, model, input_ids, attention_mask):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

    def dkd_loss(self, logits_student, logits_teacher, target, alpha, beta, student_temperature, teacher_temperature, padding_mask):
        logits_student, logits_teacher=logits_student[...,:-1,:].contiguous(), logits_teacher[...,:-1,:].contiguous()
        target, padding_mask=target[...,1:], padding_mask[...,1:]
        kl_loss = torch.nn.KLDivLoss(reduction="none")
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        pred_student = torch.nn.functional.softmax(logits_student/student_temperature, dim=-1)
        pred_teacher = torch.nn.functional.softmax(logits_teacher/teacher_temperature, dim=-1)
        pred_student, pnt_stu = self.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher, pnt_tea = self.cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = kl_loss(log_pred_student, pred_teacher)

        expand_mask = padding_mask.unsqueeze(-1).expand_as(tckd_loss)
        tckd_loss.masked_fill_(expand_mask, 0)
        mean_tckd_loss = tckd_loss.sum() / (~padding_mask).sum()

        logits_student=torch.masked_select(logits_student, ~gt_mask).view(logits_student.shape[0], logits_student.shape[1], -1)
        logits_teacher=torch.masked_select(logits_teacher, ~gt_mask).view(logits_teacher.shape[0], logits_teacher.shape[1], -1)

        pred_teacher_part2 = torch.nn.functional.softmax(
            logits_teacher / teacher_temperature , dim=-1
        )
        log_pred_student_part2 = torch.nn.functional.log_softmax(
            logits_student / student_temperature , dim=-1
        )
        nckd_loss = kl_loss(log_pred_student_part2, pred_teacher_part2)

        # the following is equal to the vanilla forward KD
        # nckd_loss=pnt_tea*nckd_loss
        # return mean_tckd_loss+mean_nckd_loss

        expand_mask = padding_mask.unsqueeze(-1).expand_as(nckd_loss)
        nckd_loss.masked_fill_(expand_mask, 0)
        mean_nckd_loss = nckd_loss.sum() / (~padding_mask).sum()

        return self.alpha * mean_tckd_loss + self.beta * mean_nckd_loss


    def _get_gt_mask(self, logits, target):
        mask = torch.zeros_like(logits).scatter_(2, target.unsqueeze(2), 1).bool()
        return mask

    def _get_other_mask(self, logits, target):
        mask = torch.ones_like(logits).scatter_(2, target.unsqueeze(2), 0).bool()
        return mask

    def cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=2, keepdims=True)
        t2 = (t * mask2).sum(2, keepdims=True)
        rt = torch.cat([t1, t2], dim=2)
        return rt, t2


    @torch.inference_mode()
    def get_generated_ids(
        self,
        model,
        tokenizer,
        input_ids,
        attention_mask,
        max_new_tokens,
        require_logits,
    ):
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_scores=require_logits,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if require_logits:
            logits = torch.cat(
                [score.unsqueeze(1) for score in outputs["scores"]], dim=1
            )
        else:
            logits = None
        return outputs["sequences"], logits
    
    @torch.inference_mode()
    def get_mix_generated_ids(
        self,
        student_model,
        teacher_model,
        tokenizer,
        input_ids,
        attention_mask,
        max_new_tokens,
        mix_ratio
    ):
        org_input_ids = input_ids.clone()
        def sample_token_from_logits(logits):
            tau = 0.001 # argmax
            distribution = torch.softmax(logits / tau, dim=-1)
            next_token_id = torch.multinomial(distribution, num_samples=1)
            return next_token_id
    
        def generate_one(model, input_ids, attention_mask, past_key_values):
            if past_key_values is None:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=True,
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            past_key_values = outputs.past_key_values
            next_token = sample_token_from_logits(outputs.logits[:, -1, :])
            return next_token, past_key_values

        bsz, prompt_len = input_ids.shape
        # always generate the first token for teacher/student to get the kv cache
        student_first_token, student_key_values = generate_one(
            student_model, input_ids, attention_mask, None)
        teacher_first_token, teacher_key_values = generate_one(
            teacher_model, input_ids, attention_mask, None)
        
        torch.manual_seed(1)
        input_ids = student_first_token if random.random() < mix_ratio else teacher_first_token
        attention_mask = torch.cat([attention_mask, torch.ones(
                bsz, 1, dtype=torch.long, device='cuda')], dim=1)

        for i in range(max_new_tokens - 1):
            sample_model, past_key_values = (student_model, student_key_values) if random.random(
            ) < mix_ratio else (teacher_model, teacher_key_values)
            next_token, _ = generate_one(sample_model, input_ids, 
                                        attention_mask, past_key_values)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones(
                bsz, 1, dtype=torch.long, device='cuda')], dim=1)

        # mask eos
        eos_positions = (input_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row, col in zip(*eos_positions):
            mask[row, col+1:] = True
        input_ids[mask] = tokenizer.pad_token_id
        return torch.cat((org_input_ids, input_ids), dim=-1).cuda()
    

    def atkd(self, logits_student, logits_teacher, target, student_temperature, teacher_temperature, padding_mask):
        if "reverse" in self.setting:
            logits_student_copy=logits_teacher
            logits_teacher_copy=logits_student
            logits_student=logits_student_copy
            logits_teacher=logits_teacher_copy

        logits_student, logits_teacher=logits_student[...,:-1,:].contiguous(), logits_teacher[...,:-1,:].contiguous()
        target, padding_mask=target[...,1:], padding_mask[...,1:]
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        pred_student = torch.nn.functional.softmax(logits_student/student_temperature, dim=-1)
        pred_teacher = torch.nn.functional.softmax(logits_teacher/teacher_temperature, dim=-1)
        pred_student, pnt_stu = self.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher, pnt_tea = self.cat_mask(pred_teacher, gt_mask, other_mask)
        if "reverse" in self.setting:
            filter_value=torch.masked_select(pnt_stu.squeeze(), ~padding_mask).median()
            easy_mask=pnt_stu<filter_value
            hard_mask=~easy_mask
        else:
            filter_value=torch.masked_select(pnt_tea.squeeze(), ~padding_mask).median()
            easy_mask=pnt_tea<filter_value
            hard_mask=~easy_mask

        easy_pred_student, hard_pred_student=torch.masked_select(pred_student, easy_mask.expand_as(pred_student)).view(-1, pred_student.size(-1)), torch.masked_select(pred_student, hard_mask.expand_as(pred_student)).view(-1, pred_student.size(-1))
        easy_pred_teacher, hard_pred_teacher=torch.masked_select(pred_teacher, easy_mask.expand_as(pred_teacher)).view(-1, pred_teacher.size(-1)), torch.masked_select(pred_teacher, hard_mask.expand_as(pred_teacher)).view(-1, pred_teacher.size(-1))
        easy_padding_mask, hard_padding_mask=torch.masked_select(padding_mask, easy_mask.squeeze()), torch.masked_select(padding_mask, hard_mask.squeeze())
        easy_gt_mask, hard_gt_mask=torch.masked_select(gt_mask, easy_mask.expand_as(gt_mask)).view(-1, gt_mask.size(-1)), torch.masked_select(gt_mask, hard_mask.expand_as(gt_mask)).view(-1, gt_mask.size(-1))
        easy_logits_student, hard_logits_student=torch.masked_select(logits_student, easy_mask.expand_as(logits_student)).view(-1, logits_student.size(-1)), torch.masked_select(logits_student, hard_mask.expand_as(logits_student)).view(-1, logits_student.size(-1))
        easy_logits_teacher, hard_logits_teacher=torch.masked_select(logits_teacher, easy_mask.expand_as(logits_teacher)).view(-1, logits_teacher.size(-1)), torch.masked_select(logits_teacher, hard_mask.expand_as(logits_teacher)).view(-1, logits_teacher.size(-1))
        easy_pnt_tea, hard_pnt_tea =torch.masked_select(pnt_tea, easy_mask), torch.masked_select(pnt_tea, hard_mask)

        easy_nckd_mask, hard_nckd_mask = ~easy_gt_mask, ~hard_gt_mask

        
        easy_tckd, easy_nckd=self.get_tckd_nckd(easy_pred_student, easy_pred_teacher, easy_padding_mask, easy_pnt_tea, easy_nckd_mask, easy_logits_student, easy_logits_teacher, student_temperature, teacher_temperature)
        hard_tckd, hard_nckd=self.get_tckd_nckd(hard_pred_student, hard_pred_teacher, hard_padding_mask, hard_pnt_tea, hard_nckd_mask, hard_logits_student, hard_logits_teacher, student_temperature, teacher_temperature)

        alpha=self.alpha
        easy_loss=easy_nckd
        hard_loss=hard_tckd+hard_nckd
        return alpha*easy_loss+(1-alpha)*hard_loss

    def get_tckd_nckd(self, pred_student, pred_teacher, padding_mask, pnt_tea, nckd_mask, logits_student, logits_teacher, student_temperature, teacher_temperature, use_pnt=False):
        kl_loss = torch.nn.KLDivLoss(reduction="none")
        tckd_loss = kl_loss(torch.log(pred_student), pred_teacher)

        expand_mask = padding_mask.unsqueeze(-1).expand_as(tckd_loss)
        tckd_loss.masked_fill_(expand_mask, 0)
        mean_tckd_loss = tckd_loss.sum() / (~padding_mask).sum()

        logits_student=torch.masked_select(logits_student, nckd_mask).view(logits_student.size(0),-1)
        logits_teacher=torch.masked_select(logits_teacher, nckd_mask).view(logits_teacher.size(0),-1)

        pred_teacher_part2 = torch.nn.functional.softmax(
            logits_teacher / teacher_temperature , dim=-1
        )
        log_pred_student_part2 = torch.nn.functional.log_softmax(
            logits_student / student_temperature , dim=-1
        )
        nckd_loss = kl_loss(log_pred_student_part2, pred_teacher_part2)

        pnt_tea=pnt_tea.unsqueeze(-1).expand_as(nckd_loss)
        # the following is equal to the vanilla forward KD
        if use_pnt:
            nckd_loss=pnt_tea*nckd_loss

        expand_mask = padding_mask.unsqueeze(-1).expand_as(nckd_loss)
        nckd_loss.masked_fill_(expand_mask, 0)
        mean_nckd_loss = nckd_loss.sum() / (~padding_mask).sum()

        return mean_tckd_loss, mean_nckd_loss
