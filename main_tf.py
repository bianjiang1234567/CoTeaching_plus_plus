import os
import datetime
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from data.utils import boolean_string
from data.mnist_tf import MNIST
from data.cifar_tf import CIFAR10, CIFAR100
from tensorflow.contrib.keras.api.keras.datasets import mnist, cifar10, cifar100
from model_tf import CoTeachingModel, compute_pure_ratio

parser = ArgumentParser()
parser.add_argument('--gpu_idx', type=str, default="0")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results/')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.5)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--num_gradual', type=int, default=10, help='how many epochs for linear drop rate, can be 5, 10, '
                                                                '15. This param is equal to Tk for R(T) in the paper.')
parser.add_argument('--exponent', type=float, default=1, help='exponent of the forget rate, can be 0.5, 1, 2. This '
                                                              'parameter is equal to c in Tc for R(T) in the paper.')
parser.add_argument('--top_bn', type=boolean_string, default=False)
parser.add_argument('--drop_rate', type=float, default=0.25)
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='mnist')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)#取2个minibatch
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--fr_type', type = str, help='forget rate type', default='type_1')

parser.add_argument('--mode_type', type = str, help='[coteaching coteaching_plus]', default='coteaching_plus')

args = parser.parse_args()

batch_size = args.batch_size
forget_rate = args.noise_rate

if args.dataset == "mnist":
    input_shape = [28, 28, 1]
    num_classes = 10
    init_epoch= 0
    args.epoch_decay_start = 80
    #train_dataset = mnist.load_data(path='/media/ubuntu/My Passport/datasets/mnist.npz')
    #test_dataset = mnist.load_data(path='/media/ubuntu/My Passport/datasets/mnist.npz')
    train_dataset = MNIST("./data/mnist", download=True, train=True, noise_type=args.noise_type,
                          noise_rate=args.noise_rate)
    test_dataset = MNIST("./data/mnist", download=True, train=False, noise_type=args.noise_type,
                         noise_rate=args.noise_rate)

elif args.dataset == "cifar10":
    input_shape = [32, 32, 3]
    num_classes = 10
    init_epoch = 20
    args.epoch_decay_start = 80
    #train_dataset = cifar10.load_data()
    #test_dataset = cifar10.load_data()
    train_dataset = CIFAR10("./data/cifar10", download=True, train=True, noise_type=args.noise_type,
                            noise_rate=args.noise_rate)
    test_dataset = CIFAR10("./data/cifar10", download=True, train=False, noise_type=args.noise_type,
                          noise_rate=args.noise_rate)

elif args.dataset == "cifar100":
    input_shape = [32, 32, 3]
    num_classes = 100
    init_epoch = 5#5
    args.epoch_decay_start = 100
    #train_dataset = cifar100.load_data()
    #test_dataset = cifar100.load_data()
    train_dataset = CIFAR100("./data/cifar100", download=True, train=True, noise_type=args.noise_type,
                             noise_rate=args.noise_rate)
    test_dataset = CIFAR100("./data/cifar100", download=True, train=False, noise_type=args.noise_type,
                           noise_rate=args.noise_rate)

else:
    raise ValueError("Unsupported dataset...")

# create log files
save_dir = args.result_dir + '/' + args.dataset + '/' + args.mode_type + '/' #'/coteaching-plus-tf/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model_str = args.dataset + '_' + args.mode_type + '_' + args.noise_type + '_' + args.fr_type + '_' + str(args.noise_rate)
txtfile = os.path.join(save_dir, model_str + ".txt")
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile + ".bak-%s" % nowTime))

noise_or_not = train_dataset.noise_or_not

# Adjust learning rate for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
learning_rate = args.lr
lr_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    lr_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

# ToDO modify forgetrate and find length of disagree id
# define drop rate schedule
if args.fr_type=='type_1':
    forget_rate_schedule = np.ones(args.n_epoch) * forget_rate
    forget_rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)
else:
    forget_rate_schedule = np.ones(args.n_epoch) * forget_rate
    forget_rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)
    forget_rate_schedule[args.num_gradual:] = np.linspace(forget_rate, 2*forget_rate, args.n_epoch-args.num_gradual)

# create training and testing batches
train_batches = train_dataset.batch_patcher(batch_size=batch_size, drop_last=True)
test_batches = test_dataset.batch_patcher(batch_size=batch_size, drop_last=True)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.visible_device_list = args.gpu_idx
config.gpu_options.allow_growth = True
with tf.Graph().as_default(), tf.Session(config=config) as sess:
    coteach = CoTeachingModel(input_shape=input_shape, n_outputs=num_classes, mode_type=args.mode_type, dataset_name= args.dataset, batch_size=batch_size,
                              drop_rate=args.drop_rate, top_bn=args.top_bn)

    #sess.run([tf.global_variables_initializer()], feed_dict={coteach.beta1: beta1} )

    for epoch in range(1, args.n_epoch):
        lr = lr_plan[epoch]  # load learning rate for current epoch
        beta1 = beta1_plan[epoch]
        #print("beta1=", beta1)
        if epoch == 1:
            sess.run([tf.global_variables_initializer()], feed_dict={coteach.beta1: beta1})

        pure_ratio_1_list, pure_ratio_2_list = [], []
        train_total, train_correct1 = 0, 0
        train_total2, train_correct2 = 0, 0

        for i, (images, labels, indexes) in enumerate(train_batches):
            *_, acc1, acc2, loss1, loss2, ind1, ind2, predicts1, predicts2, disagree_id = sess.run([coteach.train_op1, coteach.train_op2,
                                                                 coteach.extra_update_ops1, coteach.extra_update_ops2,
                                                                 coteach.acc1, coteach.acc2,
                                                                 coteach.loss1, coteach.loss2,
                                                                 coteach.ind1_update, coteach.ind2_update,
                                                                 coteach.predicts1, coteach.predicts2,
                                                                 coteach.disagree_id_tensor],
                                                                feed_dict={
                                                                    coteach.images: images, coteach.labels: labels,
                                                                    coteach.lr: lr, coteach.beta1: beta1,
                                                                    coteach.training: True,
                                                                    coteach.forget_rate: forget_rate_schedule[epoch],
                                                                    coteach.epoch: epoch, coteach.init_epoch: init_epoch,
                                                                    coteach.step: epoch*i
                                                                })
            num_remember = (1.0 - forget_rate_schedule[epoch]) * labels.shape[0]
            pure_ratio_1, pure_ratio_2 = compute_pure_ratio(ind1, ind2, indexes, noise_or_not)
            pure_ratio_1_list.append(100 * pure_ratio_1)
            pure_ratio_2_list.append(100 * pure_ratio_2)
            train_total += 1
            train_correct1 += acc1
            train_correct2 += acc2
            #print("disagree_id={}".format(disagree_id))
            #print("disagree_id_length={}".format(len(disagree_id)))
            #print("num_remember={}".format(int(num_remember)))
            #print("ind1={}".format(ind1))
            #print("ind1_length={}".format(len(ind1)))
            if (i + 1) % args.print_freq == 0:
                print("Epoch [%d/%d], Iter [%d/%d], training acc1: %.4f%%, training acc2: %.4f%%, "
                      "train loss1: %.4f, train loss2: %.4f, pure_ratio_1: %.4f%%, pure_ratio_2: %.4f%%" %
                      (epoch, args.n_epoch, i + 1, len(train_batches), acc1 * 100.0, acc2 * 100.0, loss1, loss2,
                       sum(pure_ratio_1_list) / len(pure_ratio_1_list),
                       sum(pure_ratio_2_list) / len(pure_ratio_2_list)), flush=True)
                #print("num_remember={}".format(int(num_remember)))
                #print("ind1={}".format(ind1))
                #print("ind1_length={}".format(len(ind1)))
                #print("predicts1={}".format(predicts1))
                #print("predicts1_length={}".format(len(predicts1)))
                #print("predicts2={}".format(predicts2))
                #print("predicts2_length={}".format(len(predicts2)))
                #print("disagree_id={}".format(disagree_id))
                #print("disagree_id_length={}".format(len(disagree_id)))

        train_acc1 = train_correct1 / train_total * 100.0
        train_acc2 = train_correct2 / train_total * 100.0

        # testing
        print('Evaluating %s...' % model_str)
        total, correct1, correct2 = 0, 0, 0
        for i, (images, labels, _) in enumerate(test_batches):
            pred1, pred2 = sess.run([coteach.predicts1, coteach.predicts2], feed_dict={coteach.images: images,
                                                                                       coteach.labels: labels,
                                                                                       coteach.training: False
                                                                                       })
            labels = np.reshape(labels, newshape=(labels.shape[0], ))
            total += labels.shape[0]
            correct1 += np.sum(pred1 == labels)
            correct2 += np.sum(pred2 == labels)
        acc1 = 100 * float(correct1) / float(total)
        acc2 = 100 * float(correct2) / float(total)

        mean_pure_ratio1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list) / len(pure_ratio_2_list)

        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f%% Model2 %.4f%%, Pure Ratio 1 %.4f%%, '
              'Pure Ratio 2 %.4f%%\n' % (epoch, args.n_epoch, len(test_dataset), acc1, acc2, mean_pure_ratio1,
                                         mean_pure_ratio2), flush=True)

        with open(txtfile, "a") as f:
            res = str(int(epoch)) + ": train acc1 " + str(train_acc1) + " train acc2 " + str(train_acc2)
            res += " test acc1 " + str(acc1) + " test acc2 " + str(acc2) + " pure ratio1 " + str(mean_pure_ratio1)
            res += " pure ratio2 " + str(mean_pure_ratio2) + " len(ind1) " + str(len(ind1)) + " len(ind2) " + \
                   str(len(ind2)) + " all_num_remember " + str(int(num_remember)) + " agree_id " + str(len(disagree_id)) + "\n"
            f.write(res)
