#!/bin/bash
IMAGENET_PATH=${IMAGENET_PATH:-""}

# ---- cifar10 ----
# vgg16
model=vgg16_masked_bn
for ckpt in ./ckpt/vgg/vgg-0.3/ckpt.t7 ./ckpt/vgg/vgg-0.05/ckpt.t7; do
    echo -n "Testing $ckpt ..."
    acc=$(python test.py --model $model --resume $ckpt | grep 'Test:' | awk '{print $(NF-1)}')
    echo ": $acc %"
done

# res18
model=resnet18_masked_plan1_bn
for ckpt in ./ckpt/res18/res18-0.25/ckpt.t7 ./ckpt/res18/res18-0.17/ckpt.t7 ./ckpt/res18/res18-0.07/ckpt.t7; do
    echo -n "Testing $ckpt ..."
    acc=$(python test.py --model $model --resume $ckpt | grep 'Test:' | awk '{print $(NF-1)}')
    echo ": $acc %"
done


# res20
model=cifar10_resnet20_dsconv_masked_plan1_bn
for ckpt in ./ckpt/res20/res20-0.75/ckpt.t7 ./ckpt/res20/res20-0.5/ckpt.t7  ./ckpt/res20/res20-0.5_pretrained/ckpt.t7 ./ckpt/res20/res20-0.33/ckpt.t7; do
    echo -n "Testing $ckpt ..."
    acc=$(python test.py --model $model --resume $ckpt --gpu 0 | grep 'Test:' | awk '{print $(NF-1)}')
    echo ": $acc %"
done

# res56
model=cifar10_resnet56_dsconv_masked_plan1_bn
for ckpt in ./ckpt/res56/res56-0.75/ckpt.t7 ./ckpt/res56/res56-0.5/ckpt.t7 ./ckpt/res56/res56-0.33/ckpt.t7; do
    echo -n "Testing $ckpt ..."
    acc=$(python test.py --model $model --resume $ckpt --gpu 0 | grep 'Test:' | awk '{print $(NF-1)}')
    echo ": $acc %"
done

if [[ -n "${IMAGENET_PATH}" ]]; then
    ## ---- imagenet ----
    # res18
    model=resnet18_masked_plan1_imgnet
    for ckpt in ./ckpt/res18/imgnet-res18-0.6/ckpt.t7; do
	echo -n "Testing $ckpt ..."
	python test.py --path $IMAGENET_PATH --dataset imagenet --model $model --resume $ckpt --gpu 0 | tee /tmp/tmp_test.log
	acc=$(grep 'Test:' /tmp/tmp_test.log | awk '{print $(NF-1)}')
	echo ": $acc %"
    done

    # res50
    model=resnet50_masked_plan1_bn_imgnet
    for ckpt in ./ckpt/res50/imgnet-res50-0.6/ckpt.t7 ./ckpt/res50/imgnet-res50-0.5/ckpt.t7; do
	echo -n "Testing $ckpt ..."
	python test.py --path $IMAGENET_PATH --dataset imagenet --model $model --resume $ckpt --gpu 0 | tee /tmp/tmp_test.log
	acc=$(grep 'Test:' /tmp/tmp_test.log | awk '{print $(NF-1)}')
	echo ": $acc %"
    done
fi
