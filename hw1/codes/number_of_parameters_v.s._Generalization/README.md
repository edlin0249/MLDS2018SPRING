command line:
cd Models
wget 'https://www.dropbox.com/s/al07tcqsyfuwfn5/model_CIFAR10ConvNet_3095_params?dl=1'
wget 'https://www.dropbox.com/s/qc791u2ogh4d975/model_CIFAR10ConvNet_15455_params?dl=1'
wget 'https://www.dropbox.com/s/0t6lk0blm8apwzm/model_CIFAR10ConvNet_30950_params?dl=1'
wget 'https://www.dropbox.com/s/2uvfsl5sh861wcc/model_CIFAR10ConvNet_318410_params?dl=1'
wget 'https://www.dropbox.com/s/ea8qlykmqmf1iat/model_CIFAR10ConvNet_656810_params?dl=1'
wget 'https://www.dropbox.com/s/wt6av6uycilqhlr/model_CIFAR10ConvNet_1792010_params?dl=1'
wget 'https://www.dropbox.com/s/wob6870ujw6xkzb/model_CIFAR10ConvNet_4084010_params?dl=1'
wget 'https://www.dropbox.com/s/5gtu29z5bip90xt/model_CIFAR10ConvNet_6876010_params?dl=1'
wget 'https://www.dropbox.com/s/q9ib621dcadvm7a/model_CIFAR10ConvNet_16201010_params?dl=1'
wget 'https://www.dropbox.com/s/gh5q6xpflrmtypp/model_CIFAR10ConvNet_26256010_params?dl=1'
mv model_CIFAR10ConvNet_3095_params?dl=1 model_CIFAR10ConvNet_3095_params
mv model_CIFAR10ConvNet_15455_params?dl=1 model_CIFAR10ConvNet_15455_params
mv model_CIFAR10ConvNet_30950_params?dl=1 model_CIFAR10ConvNet_30950_params
mv model_CIFAR10ConvNet_318410_params?dl=1 model_CIFAR10ConvNet_318410_params
mv model_CIFAR10ConvNet_656810_params?dl=1 model_CIFAR10ConvNet_656810_params
mv model_CIFAR10ConvNet_1792010_params?dl=1 model_CIFAR10ConvNet_1792010_params
mv model_CIFAR10ConvNet_4084010_params?dl=1 model_CIFAR10ConvNet_4084010_params
mv model_CIFAR10ConvNet_6876010_params?dl=1 model_CIFAR10ConvNet_6876010_params
mv model_CIFAR10ConvNet_16201010_params?dl=1 model_CIFAR10ConvNet_16201010_params
mv model_CIFAR10ConvNet_26256010_params?dl=1 model_CIFAR10ConvNet_26256010_params
cd ..
pythonw main.py test


references:

https://arxiv.org/pdf/1412.6614.pdf

https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py