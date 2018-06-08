### GAN
##### Test:
- `bash run_gan.sh`
---
### CONDITIONAL GAN
##### Test:
- `bash run_cgan.sh <testing_tags.txt>`
- `bash run_cgan.sh dataset/testing_tags.txt`

##### Train:
- `bash train_cgan.sh <extra_images_dir> <extra_tags.csv> <epoch_num>`
- `bash train_cgan.sh  dataset/extra_data/images dataset/extra_data/tags.csv 100`
