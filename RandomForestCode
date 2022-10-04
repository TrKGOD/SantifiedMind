# Random Forest Malware Detector 2.0
# SantifiedMind

# (PT-BR) - Esse projeto foi criado por Tarek Ayache

# (EN) - This project have been made by Tarek Ayache

# (PT-BR) Bibliotecas importadas

# (EN) Librarys  imported


from pyfiglet import Figlet
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

# (PT-BR) Esse código de linha é opcional, ele usa da biblioteca "pyfiglet" para criar um texto para seu programa.

# (EN) This code line is opcional, basically it uses the libray "pyfiglet" to create a text for your program.

f = Figlet(font='slant')
print(f.renderText('Santified-Mind'))

# (PT-BR) A seguir temos o começo do script, primeiramente é necessário colocar o caminho ou o nome do arquivo que
# possui o bando de dados que será usado como base, nesse caso será o arquivo "MalwareData.csv", dentro desse
# arquivo faremos a retirada de duas informações usando as funções drop e axis, para que o programa não tenha
# problemas de desempenho.

# (EN) This is the basic setup for the script, basically this is the main path for our Database, in this case we
# have the "MalwareData.csv" which is the file that countains our database. The next command line will
# use the funcion drop and axis to remove certain features in the database, so there isn't any performance issue.


MalwareDataset = pd.read_csv('MalwareData.csv', sep='|', low_memory=True)
Legit = MalwareDataset[0:41323].drop(['legitimate'], axis=1)
Malware = MalwareDataset[41323::].drop(['legitimate'], axis=1)

# (PT-BR) Em seguir temos a seleção de quais features serão removidos dentro do database.

# (EN) Here we have the features that will be removed from the database.

Data = MalwareDataset.drop(['Name', 'md5', 'legitimate'], axis=1).values
Target = MalwareDataset['legitimate'].values
FeatSelect = sklearn.ensemble.ExtraTreesClassifier().fit(Data, Target)
Model = SelectFromModel(FeatSelect, prefit=True)
Data_new = Model.transform(Data)

# (PT-BR) A próxima linha de código se trata da separação e preparação de nossos dados, vale ressaltar alguns detalhes
# importantes, a função "test_size=0.2" é a responsável pela separação dos dados, basicamente ela está pegando
# 20% dos dados ou 0,2 para realizar testes, enquanto o  restante dos 80% serão dados de treinamento

# (EN) The next command line will process the data, it is important to note that the function "test_size=0.2" is
# responsible to get 20% of our data to use it for tests, while the rest will be used for training purposes.

Legit_Train, Legit_Test, Malware_Train, Malware_Test = train_test_split(Data_new, Target, test_size=0.2)

# (PT-BR) Por fim temos a linha de código que fará o cálculo da porcentagem de eficácia do programa.

# (EN) Now we have the last command line that will calculate the score of the program.

clf = sklearn.ensemble.RandomForestClassifier(n_estimators=50)
clf.fit(Legit_Train, Malware_Train)
score = clf.score(Legit_Test, Malware_Test)
print("A porcentagem de acerto do programa foi de:", score * 100)

# (PT-BR) A Seguir temos funções destinadas a tentar achar as taxas de possíveis falsos positivos
# negativos do programa

# (EN) Here is the line code to calculate the porcentage of false negative and false positive

Result = clf.predict(Legit_Test)
CM = confusion_matrix(Malware_Test, Result)
print("Taxa de Falsos-Positivos : %f %%" % ((CM[0][1] / float(sum(CM[0]))) * 100))
print("Taxa de Falsos-Negativos : %f %%" % (CM[1][0] / float(sum(CM[1])) * 100))
