# LeGRNets (LeGR-pruned MobileNetV2)

This is the training code used to train LeGR-pruned network. We use TPU to conduct training. We first find affine transformations, and prune MobileNetV2 accordingly, and obtain the resulting structure and hard-code it in the training code (this reposity) to train on TPUs.

For code regarding the search of affine transformations, please see [here](https://github.com/cmu-enyac/LeGR).

This repository is modifed from the original [MnasNet training code](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet) on TPU.

**Hyperparams for FINDING AFFINE TRANSFORMATIONS**

We only use single GPU (1080 Ti). To find affine transformations for MobileNetV2, we use constraint at 50% FLOPs, \hat{\tau}=200 and randomly sampled 5k images from ImageNet as training examples to speedup the time to find affine transformations. Batch size is set to 96, learning rate is set to 4.5e-2, and minimum sparsity is set to 0.20.

70%: 71.40 (Top1)

60%: 70.75 (Top1)

50%: 69.36 (Top1)

# RUN THIS (LeGRNet)
python mnasnet_main.py --tpu=${TPU_NAME} --data_dir=${IMAGENET_DIR} --model_dir=${DATA_DIR}/legrnet50 --model_name='legrnet50' --train_batch_size=1024
