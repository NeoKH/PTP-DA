#!/usr/bin/bash
model_name="stgcnn"
model_type="source_only"
source_names=("eth" "hotel" "univ" "zara1" "zara2")
target_names=("eth" "hotel" "univ" "zara1" "zara2")
augments=("angle_target")
declare -A DEVICE=(["eth"]="cuda:0" ["hotel"]="cuda:0" ["univ"]="cuda:0" ["zara1"]="cuda:1" ["zara2"]="cuda:1")

for augment in ${augments[*]};do{
    for source in ${source_names[*]};do
    {
        for target in ${target_names[*]};do
        {
            if [ $source != $target ];then
            {
                python train.py \
                    --data_split_mode "tgnn" \
                    --device ${DEVICE[$target]} \
                    --model_type $model_type \
                    --source $source  --target $target \
                    --augment $augment \
                    --predata_regenerate
            }
            fi
        }
        done
    }
    done
}
done