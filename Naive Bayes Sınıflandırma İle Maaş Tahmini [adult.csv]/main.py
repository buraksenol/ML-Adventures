import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import preprocessing



veri_seti = pd.read_csv('adult.csv')

print(veri_seti.head())

# Her kolon içerisinde en çok kullanılan değerler.
print("Workclass        = ",veri_seti['workclass'].value_counts().idxmax())
print("education        = ",veri_seti['education'].value_counts().idxmax())
print("marital-status   = ",veri_seti['marital-status'].value_counts().idxmax())
print("occupation       = ",veri_seti['occupation'].value_counts().idxmax())
print("relationship     = ",veri_seti['relationship'].value_counts().idxmax())
print("race             = ",veri_seti['race'].value_counts().idxmax())
print("sex              = ",veri_seti['sex'].value_counts().idxmax())
print("native-country   = ",veri_seti['native-country'].value_counts().idxmax())
print("income           = ",veri_seti['income'].value_counts().idxmax())

# Sayısal değer barındıran kolonlara veri atamak için o değerlerin ortalamasını alırız yani mean()
print(veri_seti.describe())

#Veri Temizleme ve Kayıp Değerleri Ekleme İşlemleri:

del veri_seti['fnlwgt']
del veri_seti['education-num']

def kayip_deger_ekle(data_set):
    data_set['age'] = data_set['age'].replace(to_replace = ' ?', value = ' 38')
    data_set['workclass'] = data_set['workclass'].replace(to_replace = ' ?', value = ' Private')
    data_set['education'] = data_set['education'].replace(to_replace = ' ?', value = ' HS-grad')
    data_set['marital-status'] = data_set['marital-status'].replace(to_replace = ' ?', value = ' Married-civ-spouse')
    data_set['occupation'] = data_set['occupation'].replace(to_replace = ' ?', value = ' Prof-specialty')
    data_set['relationship'] = data_set['relationship'].replace(to_replace = ' ?', value = ' Husband')
    data_set['race'] = data_set['race'].replace(to_replace = ' ?', value = ' White')
    data_set['sex'] = data_set['sex'].replace(to_replace = ' ?', value = ' Male')
    data_set['capital-gain'] = data_set['capital-gain'].replace(to_replace = ' ?', value = ' 1077')
    data_set['capital-loss'] = data_set['capital-loss'].replace(to_replace = ' ?', value = ' 87')
    data_set['hours-per-week'] = data_set['hours-per-week'].replace(to_replace = ' ?', value = ' 40')
    data_set['income'] = data_set['income'].replace(to_replace = ' ?', value = ' <=50K')
    data_set['native-country'] = data_set['native-country'].replace(to_replace = ' ?', value = ' United-States')

    return data_set

temiz_veri_seti = kayip_deger_ekle(veri_seti)

# Veri Seti Tablosundaki "Categorical" Text Tabanlı Değerleri Encode eder.
encode = preprocessing.LabelEncoder()

workclass = encode.fit_transform(temiz_veri_seti['workclass'])
education = encode.fit_transform(temiz_veri_seti['education'])
marital_status = encode.fit_transform(temiz_veri_seti['marital-status'])
occupation = encode.fit_transform(temiz_veri_seti['occupation'])
relationship = encode.fit_transform(temiz_veri_seti['relationship'])
sex = encode.fit_transform(veri_seti['sex'])
native_country = encode.fit_transform(temiz_veri_seti['native-country'])


encode_veri_seti = list(zip(temiz_veri_seti['age'],
                            workclass,education,marital_status,occupation,relationship,
                            sex,temiz_veri_seti['capital-gain'],temiz_veri_seti['capital-loss'],
                            temiz_veri_seti['hours-per-week'],native_country))

# Target dediğimiz hedef değeri (tahmin edilecek değer) birleştirilen listeye dahil etmeden ayrı olarak encode ediyoruz
tahmin_edilecek_sonuc = encode.fit_transform(temiz_veri_seti['income'])

X_train, X_test, y_train, y_test = train_test_split(encode_veri_seti, tahmin_edilecek_sonuc, test_size = 0.1, random_state=109)

model = GaussianNB()

model.fit(X_train, y_train)

tahmin_sonuclari = model.predict(X_test)

dogruluk_orani = metrics.accuracy_score(y_test, tahmin_sonuclari)

print("Doğruluk Oranı : %", dogruluk_orani * 100)
