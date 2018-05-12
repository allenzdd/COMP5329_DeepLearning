### the "mlp.ipynb" is our assignment work not only has predict data, but also has analysis every function and many experiments, if you want to review code, please use it in jupyter

### if you only want to get predict test data label, please run "predict_mlp.py" in python3

### in all data, if you want to change each modules and parameters, please change them at MLP and MLP.fit classes and functions, as follows

nn=MLP(input_data.shape[1], *[100, 100, 100]*, label_data.shape[1], '*activation function*', weight_norm=*True/False*, dropout=*True/False*,  keep_prob=*0.8*, output_softmax_crossEntropyLoss=*True/False*, weight_decay=*True/False*, weight_lambda=*0.0008*)

nn.fit(data, label, learning_rate=*0.05*, epochs=*60*, gd=*'mini_batch'*, momentum=*True/False*, gamma_MT=*0.9*, mini_batch_size=*512*, batch_norm=*True/False*)

### run all of file code need input data in the same folder