if ! [ -e "hw2_2.model" ]; then
    wget https://www.dropbox.com/s/uxl4jf6kc23hv39/hw2_2.model?dl=1
    mv hw2_2.model?dl=1 hw2_2.model
fi

if ! [ -e "voc_hw2_2" ]; then
    wget https://www.dropbox.com/s/wo264x2ilx263hc/voc_hw2_2?dl=1
    mv voc_hw2_2?dl=1 voc_hw2_2
fi

python3 hw2_2.py $1 $2