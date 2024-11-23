corpus = "IMDB"
stego_method = "VLC"
dataset = ["3bpw"]
# stego_method = "ADG"
# dataset = ["xbpw"]

tuning_param = ["lambda_loss", "main_learning_rate", "batch_size", "nepoch", "temperature", "SEED", "dataset"]
lambda_loss = [0.5]
temperature = [0.3]
batch_size = [32]
decay = 1e-02
main_learning_rate = [1e-3]
hidden_size = 768
nepoch = [10]
label_list = [None]
SEED = [0]

loss_type = "ce"  # ['ce', 'scl']
is_waug = False
model_type = "GE" # ["ls-cnn", "rnn", "bilstm_dense", "r_bilstm_c", "sesy", "bert", "scl", "etaff"]
task = None # ['graph_steganalysis', None]
pkl_file = "_preprocessed_bert.pkl" # ['_preprocessed_ernie.pkl']


param = {"temperature": temperature, "corpus": corpus, "stego_method": stego_method, "main_learning_rate": main_learning_rate, "batch_size": batch_size, "hidden_size": hidden_size,
         "nepoch": nepoch, "dataset": dataset, "lambda_loss": lambda_loss, "loss_type": loss_type,
         "decay": decay, "SEED": SEED, "model_type": model_type, "is_waug": is_waug, "task": task, "pkl_file": pkl_file}


