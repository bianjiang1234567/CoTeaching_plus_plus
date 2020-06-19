# CoTeaching++

This repository improves CoTeaching [Co-teaching Robust Training of Deep Neural Networks with Extremely Noisy Labels](
https://papers.nips.cc/paper/8072-co-teaching-robust-training-of-deep-neural-networks-with-extremely-noisy-labels.pdf).

It uses the small-loss trick and samples which are predicted equivalently by the two networks.

It is implemented by TensorFlow

# CoTeaching+

CoTeaching+ is the ICML'19 paper
[How does Disagreement Help Generalization against Label Corruption?](https://arxiv.org/abs/1901.04215)

# Difference between CoTeaching++ and CoTeaching+

CoTeaching++ is a little different from CoTeaching+

CoTeaching+ selects the samples which are predicted differently by the two networks.

# Usage

You can install TensorFlow1.4 cuda8 cudnn6

To see the improvements between CoTeaching and CoTeaching++

Here are examples:

${dataset_name} can be cifar10 cifar100

```bash
$ python main_tf.py --dataset ${dataset_name} --noise_type symmetric --fr_type type_1 --batch_size 128 --noise_rate 0.2 --mode_type coteaching
$ python main_tf.py --dataset ${dataset_name} --noise_type symmetric --fr_type type_1 --batch_size 128 --noise_rate 0.2 --mode_type coteaching_plus

$ python main_tf.py --dataset ${dataset_name} --noise_type symmetric --fr_type type_1 --batch_size 128 --noise_rate 0.5 --mode_type coteaching
$ python main_tf.py --dataset ${dataset_name} --noise_type symmetric --fr_type type_1 --batch_size 128 --noise_rate 0.5 --mode_type coteaching_plus

$ python main_tf.py --dataset ${dataset_name} --noise_type pairflip --fr_type type_1 --batch_size 128 --noise_rate 0.45 --mode_type coteaching
$ python main_tf.py --dataset ${dataset_name} --noise_type pairflip --fr_type type_1 --batch_size 128 --noise_rate 0.45 --mode_type coteaching_plus
```

# Notion
You can replace this line
```
NonEqual = tf.equal(pred1, pred2)
```
in model_tf.py by
```
NonEqual = tf.not_equal(pred1, pred2)
```
Then this repository is changed to implement CoTeaching+

# Improments betwenn CoTeaching and CoTeaching++
<div align="center">
<img src=assets/README-ad749488.jpg width = 45% height = 50%/>
<img src=assets/README-131844cd.jpg width = 45% height = 50%/>
</div>
<div align="center">
CoTeaching   　　　　        　　CoTeaching++
</div>

<div align="center">
<img src=assets/README-6e78b47a.jpg width = 45% height = 50%/>
<img src=assets/README-32a75cc4.jpg width = 45% height = 50%/>
</div>
<div align="center">
CoTeaching　　　　　　        　　CoTeaching++
</div>

<div align="center">
<img src=assets/README-88f5a9e8.jpg width = 45% height = 50%/>
<img src=assets/README-5b3f32fc.jpg width = 45% height = 50%/>
</div>
<div align="center">
CoTeaching　　　　　　        　　CoTeaching++
</div>

<div align="center">
<img src=assets/README-8641b182.jpg width = 45% height = 50%/>
<img src=assets/README-c06628a0.jpg width = 45% height = 50%/>
</div>
<div align="center">
CoTeaching　　　　　　        　　CoTeaching++
</div>

<div align="center">
<img src=assets/README-83281159.jpg width = 45% height = 50%/>
<img src=assets/README-690ef97b.jpg width = 45% height = 50%/>
</div>
<div align="center">
CoTeaching　　　　　　        　　CoTeaching++
</div>

<div align="center">
<img src=assets/README-49568704.jpg width = 45% height = 50%/>
<img src=assets/README-1f8082b3.jpg width = 45% height = 50%/>
</div>
<div align="center">
CoTeaching　　　　　　        　　CoTeaching++
</div>
