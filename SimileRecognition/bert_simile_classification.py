import torch
import sys
import csv
import os
import logging
import argparse
import random
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification, BertAdam, BertForSequenceClassification
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, RandomSampler, DistributedSampler, DataLoader, SequentialSampler
from tqdm import trange, tqdm
from sklearn.metrics import classification_report, f1_score

# bert_model = './bert_models/bert-base-chinese'
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, sent_label = None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.sent_label = sent_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, sent_label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sent_label_id = sent_label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(np.unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class SimileProcessor(DataProcessor):
    """Processor for the simile data set."""

    def __init__(self):
        self.language = 'zh'

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_test_examples(self._read_tsv(os.path.join(data_dir, 'sentences_with_xiang.txt')), 'dev')

    def get_labels(self):
        return ['EOS', 'CLS', 'SEP', "O", "ss", "sb", "si", "se", "ts", "tb", "ti", "te", "aps", "apb", "api", "ape", "vps",
                "vpb", "vpe", "vpi"]

    def _create_examples(self, lines, set_type):
        examples = []
        words = []
        labels = []
        sent_label = None
        j = 0
        guid = 0
        for (i, line) in enumerate(lines):
            if len(line) > 0:
                line_parts = line[0].strip().split()
                if len(line_parts) == 1:
                    sent_label = line_parts[0]
                elif len(line_parts) > 1:
                    guid = '%s-%s' % (set_type, j)
                    text_a = line_parts[0]
                    label = line_parts[1]
                    words.append(text_a)
                    labels.append(label)
            else:
                examples.append(InputExample(guid=guid, text_a=words, label=labels, sent_label=sent_label))
                words = []
                labels = []
                sent_label = None
                j += 1
        return examples

    def _create_test_examples(self, lines, set_type):
        examples = []
        j = 0
        for (i, line) in enumerate(lines):
            if len(line)>0:
                guid = '%s-%s' % (set_type, i)
                words = line[0].strip().replace(' ','')
                examples.append(InputExample(guid=guid, text_a=words))
                j += 1
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    sent_label_map = {'0': 0, '1': 1}
    features = []
    for (ex_index, example) in enumerate(examples):
        # tokens_a = example.text_a
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ['[CLS]'] + tokens_a + ['[SEP]']

        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,
                                      segment_ids=segment_ids, label_id=None, sent_label_id=None))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def sent_eval(pred_list, true_list, label_map):
    """Sentence level evaluate."""

    correct_num = 0
    pre_simile_num = 0
    true_simile_num = 0
    for i, pred in enumerate(pred_list):
        pre_simile_flag = 0
        correct_flag = 0
        pred = [t for t in pred if t != label_map['CLS'] and t != label_map['SEP'] and t != label_map['EOS']]
        for t in pred:
            if t != label_map['O']:
                pre_simile_num += 1
                pre_simile_flag = 1
                break
        true_label = [t.item() for t in true_list[i] if t != label_map['CLS'] and t != label_map['SEP'] and t != label_map['EOS']]
        for t in true_label:
            if t != label_map['O']:
                true_simile_num += 1
                break
        for j, t in enumerate(pred):
            correct_flag -= 1 if t != true_label[j] else 0

        if correct_flag >= 0 and pre_simile_flag == 1:
            correct_num += 1

    p = correct_num / pre_simile_num
    r = correct_num / true_simile_num
    f = (2 * p * r) / (p + r)

    return p, r, f


def ouput_preds(examples, preds):
    """Output the preds labels."""

    sent_preds = []
    for i, example in enumerate(examples):
        sentence = example.text_a
        label = example.sent_label
        pred_labels = preds[i]
        sent_preds.append([sentence, label, str(pred_labels)])
    return sent_preds


def main():
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument('--data_dir',
                        default=None,
                        type=str,
                        required=True,
                        help='The input data dir. Should contain the .tsv files (or other data files) for the task.')

    parser.add_argument('--bert_model', default=None, type=str, required=True,
                        help='Bert pre-trained model selected in the list: '
                             'bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, '
                             'bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese.')
    parser.add_argument('--task_name',
                        default=None,
                        type=str,
                        required=True,
                        help='The name of the task to train.')
    parser.add_argument('--output_dir',
                        default=None,
                        type=str,
                        required=True,
                        help='The output directory where the model predictions and checkpoints will be written.')

    # other parameters
    parser.add_argument('--cache_dir',
                        default='',
                        type=str,
                        help='Where do you want to store the pre_trained models downloaded from s3')
    parser.add_argument('--max_seq_length',
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_classify",
                        action='store_true')
    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()
    processors = {
        'simile': SimileProcessor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_classify:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    # tokenizer = torch.load('./bert_models/tokenizer.pkl')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    label_list = processor.get_labels()
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=2)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    global_step = 0
    train_loss = {}
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info('***** Running training *****')
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        all_sent_label_ids = torch.tensor([f.sent_label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sent_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc='Epoch'):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, sent_label_ids = batch

                # define a new function to compute loss values
                sc_logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(sc_logits.view(-1, 2), sent_label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss = + loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            logger.info('Epoch: %d, Loss: %f' %(_, tr_loss))
            train_loss[_] = tr_loss

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        output_config_file = os.path.join(args.output_dir, "config.json")

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=2)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir)

        # output loss
        output_loss_file = os.path.join(args.output_dir, "train_loss.txt")
        with open(output_loss_file, 'w', encoding='utf-8') as f:
            for epoch, val in train_loss.items():
                f.write('Epoch: '+str(epoch)+' '+'Loss: '+str(val))
        f.close()
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=2)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)[:5]
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        all_sent_label_ids = torch.tensor([f.sent_label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sent_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        sc_eval_loss = 0
        nb_eval_steps = 0
        sc_preds = []

        for input_ids, input_mask, segment_ids, sent_label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            sent_label_ids = sent_label_ids.to(device)

            with torch.no_grad():
                sc_logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

            # create eval loss and other metric required by the task
            loss_fct = CrossEntropyLoss()
            sc_eval = loss_fct(sc_logits.view(-1, 2), sent_label_ids.view(-1))

            sc_eval_loss += sc_eval.mean().item()
            nb_eval_steps += 1
            if len(sc_preds) == 0:
                sc_preds.append(sc_logits.detach().cpu().numpy())
            else:
                sc_preds[0] = np.append(sc_preds[0], sc_logits.detach().cpu().numpy(), axis=0)

        sc_eval_loss = sc_eval_loss / nb_eval_steps
        sc_preds = np.argmax(sc_preds[0], axis=1)

        # output label
        sent_preds = ouput_preds(eval_examples, sc_preds)

        # sc evaluate
        sc_f1 = f1_score(y_pred=sc_preds.tolist(), y_true=all_sent_label_ids.view(-1).tolist())
        sc_results = classification_report(y_pred=sc_preds, y_true=all_sent_label_ids.view(-1).tolist(), target_names=['literal', 'simile'])

        res = {'sc_f1 ': sc_f1, 'sc_eval_loss': sc_eval_loss}
        for key in sorted(res.keys()):
            logger.info("  %s = %s", key, str(res[key]))

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w", encoding='utf-8') as writer:
            logger.info("***** Eval results *****")
            for sent in sent_preds:
                writer.write("sentence: "+" ".join(sent[0]) + '\n')
                writer.write("true_label: "+" ".join(sent[1]) + '\n')
                writer.write("pred_label: "+" ".join(sent[2]) + '\n')
            writer.write(sc_results)
        writer.close()

    if args.do_classify and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)[:10]
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        nb_eval_steps = 0
        sc_preds = []

        for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                sc_logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

            nb_eval_steps += 1
            if len(sc_preds) == 0:
                sc_preds.append(sc_logits.detach().cpu().numpy())
            else:
                sc_preds[0] = np.append(sc_preds[0], sc_logits.detach().cpu().numpy(), axis=0)

        sc_preds = np.argmax(sc_preds[0], axis=1)

        # output label
        sent_preds = ouput_preds(eval_examples, sc_preds)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w", encoding='utf-8') as writer:
            logger.info("***** Eval results *****")
            for sent in sent_preds:
                writer.write("sentence: " + " ".join(sent[0]) + '\n')
                writer.write("pred_label: " + " ".join(sent[2]) + '\n')
        writer.close()

if __name__ == '__main__':
    main()
