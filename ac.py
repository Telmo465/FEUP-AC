import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score



#load all data
clients = pd.read_csv("client.csv", sep=";")
disps = pd.read_csv("disp.csv", sep=";")
cards = pd.read_csv("card_train.csv", sep=";")
cards_test = pd.read_csv("card_test.csv", sep=";")
accounts = pd.read_csv("account.csv", sep=";")
districts = pd.read_csv("district.csv", sep=";")
transactions = pd.read_csv("trans_train.csv", sep=";")
transactions_test = pd.read_csv("trans_test.csv", sep=";")
loans = pd.read_csv("loan_train.csv",sep=";")
loans_test = pd.read_csv("loan_test.csv",sep=";")

districts.rename(columns={"code ": "district_id"}, inplace=True)
districts.drop(['region'], axis=1, inplace=True)
districts.drop(['name '], axis=1, inplace=True)

# Merging
clients_disps = pd.merge(clients, disps, on="client_id", how="outer")
clients_disps_cards = pd.merge(clients_disps, cards, on="disp_id", how="outer")
clients_disps_cards_test = pd.merge(clients_disps, cards_test, on="disp_id", how="outer")
clients_disps_cards_districts = pd.merge(clients_disps_cards, districts, on="district_id", how="outer")
clients_disps_cards_districts_test = pd.merge(clients_disps_cards_test, districts, on="district_id")
clients_disps_cards_districts_accounts = pd.merge(clients_disps_cards_districts, accounts, on="account_id")
clients_disps_cards_districts_accounts_test = pd.merge(clients_disps_cards_districts_test, accounts, on="account_id")
account_loan_trans = pd.merge(clients_disps_cards_districts_accounts, transactions , on="account_id")
account_loan_trans_test = pd.merge(clients_disps_cards_districts_accounts_test, transactions_test , on="account_id")
account_loan_data = pd.merge(account_loan_trans, loans, on="account_id", how="outer")
account_loan_data_test = pd.merge(account_loan_trans_test, loans_test, on="account_id", how="outer")

#clean some data
# clients_disps_cards_districts.drop(['district_id'], inplace=True, axis=1)
# clients_disps_cards_districts_test.drop(['district_id'], inplace=True, axis=1)


#plots
#plot for types of clients
# plt.title('Distribution of client\'s type', fontdict={'fontsize': 14, 'fontweight': 'bold'})
# plt.hist(disps['type'], align='left')
# plt.savefig("estatisticas/tipos_cliente.png")
#-------------------------------
#plot for types of cards
# plt.title('Distribution of card\'s type', fontdict={'fontsize': 14, 'fontweight': 'bold'})
# plt.hist(cards['type'], align='left')
# plt.savefig("estatisticas/tipos_cartoes.png")
#-------------------------------
#plot for acount frequenciy of issuance of statements
# plt.title("Distribution of account's frequency", fontdict={'fontsize': 14, 'fontweight': 'bold'})
# plt.hist(accounts["frequency"])
# plt.savefig("estatisticas/frequencia_declaracoes_conta.png")
#-------------------------------
#plot for loans stats
#plot for loan amounts
# plt.title("Distribution of loans amount", fontdict={'fontsize': 14, 'fontweight': 'bold'})
# plt.hist(loans["amount"])
# plt.savefig("estatisticas/valores_emprestimos.png")
#-------------------------------
#plot for loans monthly payments
# plt.title("Distribution of loans monthly payments ", fontdict={'fontsize': 14, 'fontweight': 'bold'})
# plt.hist(loans["payments"])
# plt.savefig("estatisticas/valores_prestaçao.png")
#-------------------------------
#plot for loans durations
# plt.title("Distribution of loans durations ", fontdict={'fontsize': 14, 'fontweight': 'bold'})
# plt.hist(loans["duration"], align='mid')
# plt.savefig("estatisticas/tempo_emprestimo.png")
#-------------------------------
#plot for transactions
#plot for transactions amounts
# plt.title("Distribution of transactions amount ", fontdict={'fontsize': 14, 'fontweight': 'bold'})
# plt.hist(transactions["amount"], align='mid')
# plt.savefig("estatisticas/valores_transaçoes.png")
#-------------------------------
#plot for balance after transactions
# plt.title("Distribution of balance after transactions ", fontdict={'fontsize': 14, 'fontweight': 'bold'})
# plt.hist(transactions["balance"], align='mid')
# plt.savefig("estatisticas/valor_conta_depois_transaçao.png")
#-------------------------------
#plot for transaction type
# plt.title("Distribution of transactions type", fontdict={'fontsize': 14, 'fontweight': 'bold'})
# plt.hist(transactions["type"], align='left')
# plt.savefig("estatisticas/tipo_transaçao.png")
#-------------------------------
#plot for transaction mode
# plt.figure(figsize=(15,5))
# transactions["operation"].fillna("not defined", inplace=True)
# plt.title("Distribution of transactions mode", fontdict={'fontsize': 14, 'fontweight': 'bold'})
# plt.hist(transactions["operation"], align='left', rwidth=0.2)
# plt.savefig("estatisticas/tipo_transaçao.png")
#-------------------------------
# plt.figure(figsize=(60,10))
# transactions["k_symbol"].fillna("", inplace=True)
# plt.title("Distribution of characterization of the transaction", fontdict={'fontsize': 14, 'fontweight': 'bold'})
# plt.hist(transactions["k_symbol"], align='mid', rwidth=1)
# plt.savefig("estatisticas/caracterizacao_transaçao.png")






#substituir os valroes "?" pela media dos valores

test = districts.loc[districts["unemploymant rate '95 "] != '?']
test["unemploymant rate '95 "] = [float(x) for x in test["unemploymant rate '95 "]]
mean_rate = test["unemploymant rate '95 "].mean()
districts["unemploymant rate '95 "] = [mean_rate if x == "?" else float(x) for x in districts["unemploymant rate '95 "]]
test_2 = districts.loc[districts["no. of commited crimes '95 "] != '?']
test_2["no. of commited crimes '95 "] = [float(x) for x in test["no. of commited crimes '95 "]]
mean_no = test_2["no. of commited crimes '95 "].mean()
districts["no. of commited crimes '95 "] = [mean_no if x == "?" else float(x) for x in districts["no. of commited crimes '95 "]]


transactions["type"] = [i if i != "withdrawal in cash" else "withdrawal" for i in transactions["type"]]
transactions["k_symbol"] = ["" if type(i) != str or i == " " else i for i in transactions["k_symbol"]]
transactions["operation"] = ["" if type(i) == float else i for i in transactions["operation"]]

transactions_test["type"] = [i if i != "withdrawal in cash" else "withdrawal" for i in transactions_test["type"]]
transactions_test["k_symbol"] = ["" if type(i) != str or i == " " else i for i in transactions_test["k_symbol"]]
transactions_test["operation"] = ["" if type(i) == float else i for i in transactions_test["operation"]]



clients_disps_cards_districts_accounts = clients_disps_cards_districts_accounts.drop(["birth_number"], axis=1)
clients_disps_cards_districts_accounts_test = clients_disps_cards_districts_accounts_test.drop(["birth_number"], axis=1)

no_ids = account_loan_data.drop(["client_id","district_id_x", "trans_id", "disp_id", "card_id", "account_id", "loan_id"], axis=1)
no_ids_test = account_loan_data_test.drop(["client_id","district_id_x", "trans_id", "disp_id", "card_id", "account_id", "loan_id"], axis=1)

corr_matrix = no_ids.corr().abs()
plt.figure(figsize = (20,10))
sb.heatmap(corr_matrix,annot=True)

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

no_ids.drop(to_drop, axis=1, inplace=True)
no_ids_test.drop(to_drop, axis=1, inplace=True)

# # plt.savefig("estatisticas/matriz_correlacao.png")
print("{} Dropped columns: {}".format(len(to_drop), to_drop) )


no_ids.dropna(subset=["status"], inplace=True)
print(no_ids["status"].value_counts())

for i in no_ids.columns:
    if no_ids[i].dtype == object:
        no_ids = pd.get_dummies(no_ids, columns = [i])
    if no_ids_test[i].dtype == object:
        no_ids_test = pd.get_dummies(no_ids_test, columns = [i])
    
        

no_ids["account"].fillna(0, inplace=True)


no_ids_test["account"].fillna(0, inplace=True)

print(no_ids)

all_inputs = no_ids.drop(columns=["status"]).values
all_labels = no_ids["status"].values

(inputs_train, inputs_test, labels_train, labels_test) = train_test_split(all_inputs, all_labels, test_size=0.2, stratify=no_ids['status'])

classifier = AdaBoostClassifier()

grid_search = GridSearchCV(classifier, scoring="roc_auc", cv=10, param_grid={})
grid_search.fit(inputs_train, labels_train)
print('Best score: {}'.format(grid_search.best_score_))



# print(53 * '=')
# print("TRAIN")
# predictions_train = grid_search.predict(inputs_train)
# print("F1 Score: {}".format(f1_score(labels_train, predictions_train)))
# print(f"ROC: {roc_auc_score(labels_train, predictions_train)}")
# print("Classification Report: ")
# print(classification_report(labels_train, predictions_train, target_names=['not pay', 'pay']))
# print(53 * '=')
# print("TEST")
# predictions_test = grid_search.predict(inputs_test) 
# print("F1 Score: {}".format(f1_score(labels_test, predictions_test)))
# print(f"ROC: {roc_auc_score(labels_test, predictions_test)}")
# print("Classification Report: ")
# print(classification_report(labels_test, predictions_test, target_names=['not pay', 'pay']))    










