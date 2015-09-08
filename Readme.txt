Author: Li Fei
Date: June 24th, 2015
Email: lifei_csnlp@whu.edu.cn

This is a java project for extracting drugs, diseases and adverse drug events jointly.
We employ the perceptron to train the model and multiple-beam search algorithm to decode.
The program is tested on java runtime environment 7.
All the copyrights of data, libraries and tools belong to their authors.
We do not provide the data and other resources, because they are too big to be uploaded.
You can download the data directly at: https://sites.google.com/site/adecorpus/
If you need the resources used in this project, please contact me.

The architecture of this project is shown as below. 
lib --- all the third-party libraries and tools
src --- all the source code
*.properties --- configuration files

The architecture of the src directory is shown as below:
utils --- utility classes about ADE corpus
drug_side_effect_utils --- utility classes about this project
joint --- joint model for extracting drugs, diseases and adverse drug events
pipeline --- pipelined model to extracting drugs, diseases and adverse drug events with two separated perceptrons
