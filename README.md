# **ADAPTT:**
#### Provendo Eficiência de Recursos na Classificação de Tráfego Através do Uso Sinergético e Adaptativo de FPGAs e CNNs
#### *Providing Resource Efficiency for Traffic Classification through Adaptive and Sinergestic Use of FPGAs and CNNs*

This repostory contains the neccesary code for training and pruning CNNs for four traffic classification tasks. CNNs can be exported to ONNX files to be synthesized to FPGA accelerators.

ADAPTT is built on top of AMD/Xilinx's [FINN](https://github.com/Xilinx/finn/) and [Brevitas](https://github.com/Xilinx/brevitas) tools for training and compiling CNNs to FPGAs[^0]. You may follow their instructions for setting up you environment before using ADAPTT. 

This repository uses the classification tasks from the [ISCX VPN-nonVPN dataset](https://www.cs.unb.ca/research-expo/expos/2016/submissions/Mohammad%20Mamun_msi.mamun@unb.ca_VPNCharacterization-UNBExpo2016.pdf)[^1]

##### The basic structure of ADAPTT is:
* /data 
	* /VPN -- *Where the VPN-nonVPN dataset is placed*

* /folding_cfgs -- *Where the FINN folding configurations for each task are placed (used for pruning and later FPGA compilation)*

* model.py -- *Holds the CNN used in our use case[^2]*

* prune.py -- *Implements the CNN pruning[^3]*

* train.py -- *Holds the main()*

* trainer.py -- *Holds some auxiliary code for training*

* quantizers.py -- *Holds some auxiliary Brevitas code*

* vpn_dataset -- *Implements the necessary code for loading the VPN dataset*

##### Some examples:
1. python3 train.py --num_classes 2 --epochs 500
	
	*Trains the use case CNN for 500 epochs on the binary classification task (i.e., VPN or Non-VPN)*

2. python3 train.py --num_classes 2 --prune --pruning_rate 0.25 --retrain --epochs 5 --folding_cfg folding_cfgs/default_folding_2.json --pretrained experiments/vpn_2_classes_4_bits_/checkpoints/best.tar --export experiments/pruned.onnx

	*Prunes the --pretrained model at 25% pruning rate considering the an accelerator with the folding specified in --folding_cfg. Then, retrains the pruned model for 50 epochs*

3. python3 train.py --num_classes 2 --pretrained experiments/vpn_2_classes_4_bits_/checkpoints/best.tar --export experiments/vpn_2_classes_4_bits_/checkpoints/best.onnx
	
	*Loads the --pretrained trained CNN and exports it to an ONNX file (specified in --export)*


[^0]: Some CNN training code (e.g., training iteration, exporting, logging) was leveraged as is from the available FINN/Brevitas examples.

[^1]: Draper-Gil, Gerard, et al. "Characterization of encrypted and vpn traffic using time-related." Proceedings of the 2nd international conference on information systems security and privacy (ICISSP). 2016.

[^2]: Wang, Wei, et al. "End-to-end encrypted traffic classification with one-dimensional convolution neural networks." 2017 IEEE international conference on intelligence and security informatics (ISI). IEEE, 2017.

[^3]: Korol, Guilherme, et al. "AdaFlow: a framework for adaptive dataflow CNN acceleration on FPGAs." 2022 Design, Automation & Test in Europe Conference & Exhibition (DATE). IEEE, 2022.