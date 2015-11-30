Author: Li Fei
Date: June 24th, 2015
Email: lifei_csnlp@whu.edu.cn

This is a java project for extracting drugs, diseases and adverse drug events jointly.
The program is compatible on the JRE 7 or up environment.
The ADE corpus we used is available at: https://sites.google.com/site/adecorpus/

We firstly explore the model of the structured perceptron with beam search which has been published at:

@INPROCEEDINGS{
author={Fei Li, Donghong Ji∗, Xiaomei Wei and Tao Qian}, 
booktitle={Bioinformatics and Biomedicine (BIBM), 2015 IEEE International Conference on}, 
title={A Transition-Based Model for Jointly Extracting Drugs, Diseases and Adverse Drug Events}, 
year={2015}, 
pages={599-602}, 
}

Then we leverage the transition-based neural network(TNN) model to improve the performance significantly. 
Moveover, we tried to combine the structured perceptron and TNN but failed.

The architecture of this project is shown as below. 
*.properties --- the configuration files
src.drug_side_effect_utils --- the tool classes
src.utils --- the tool classes
src.joint --- the work published in BIBM 2015
src.pipeline --- the baseline above
src.nn --- TNN

