

#                   *** IS PROBLEMI BILGISI***

# train ve test veri setlerini belirlerken �grenci numaranizi seed olarak kullaniniz.

#Lineer olasilik ve logistic regresyon modellerini kullanarak model tahminlerini yapiniz. Iki modelin performanslarini karsilastiriniz. Bu veri setinde:

#  Hedef degisken: RESPONSE

#Veri setinde PC ile baslayan degiskenler boyut azaltma ile olusturulan yeni degiskenleri, RSCORE risk puanini, TMAIL toplam iletisim sayisini, TSPEND toplam harcama miktarini, NSPEND yapilan harcama sayisini, NSERVICE �ye olunan hizmet sayisini, NINQ birey i�in yapilan kredi arastirma sayilarini, LOG_TEN firma �yeliginin s�resini (log), LOG_AGE bireyin yasini (log), HHINCOME hanehalki gelirini g�stermektedir.





#Bu kisimda kullanimim i�in gerekli olan kutuphaneleri basta ekliyorum ve veri setini yukluyorum
library(ggplot2)
library(entropy)
library(dplyr)
library(tidyr)
library(caret)
library(xgboost)
library(pastecs)
library(rpart)
library(rpart.plot)



attr_sample <- read_excel("C:/Users/ahmet/Desktop/VeriAnaligi/Dersler/IVA-510_IsletmelericinVeriAnalitigi/Veri Setleri/attr_sample.xlsx")
head(attr_sample)

#Veri setimde degisiklikler yapabilirim. Bu sebeple basta baska bir isme atama yapiyorumki ana veri setim korunmus oluyor. 

df <- attr_sample


#                **** EDA (Exploratory Data Analysis)****


#Aslinda bu g�revde tam olarak EDA uygulamiyoruz. Veri seti d�zenlenmis olarak bize geldi. �rnegin asagida bos deger(na-missing value) kontrol� yaptigimizda sorgunun false oldugu g�r�lebilir. Bu sebeple genel EDA kurallarinin bir kismini uygulayacagiz.


head(df)
summary(df)
stat.desc(df)
is.na(df)
colnames(df)

table(df$RESPONSE)
#Hedef degiskenimiz 0'a homojen mi heterojen mi oldugunu g�r�yoruz.
#    0     1 
# 18991   605 

prop.table(table(df$RESPONSE))

#Proportion'i bakarak y�zde ne kadar homojen oldugu g�rebiliyoruz.
#     0          1 
# 0.96912635 0.03087365 


e_parent <- entropy(table(df$RESPONSE), unit="log2")

# Burada entropisini hesapladik. Entropi 0'a ne kadar yakinsa o kadar saf oldugunu g�r�yoruz. 
# 0.1987544


df$d_age = as.numeric(df$AGE>54)
df$d_hincome = as.numeric(df$HHINCOME>1011)
df$d_rscore= as.numeric(df$RSCORE>737)

#Ortalamanin �st� ve altina ayrica hane halkinin gelirine g�re siniflandirma yapiyoruz.

View(df)

#**Age i�in entropi hesabi yapiyoruz. **

e_age1 = entropy(table(df$RESPONSE[df$d_age==1]))
#0.1774607
e_age0 = entropy(table(df$RESPONSE[df$d_age==0]))
#0.1083692

#Bilgi kazancini hesaplamak i�in proportion'lari hesapliyoruz. 
p1 = sum(df$d_age==1)/length(df$d_age)
#0.4020208
p2 = sum(df$d_age==0)/length(df$d_age)
#0.5979792

# Bu verileri g�re bilgi kazancinin hesabini yapabiliriz. 

ig_age = e_parent-(p1*e_age1+p2*e_age0)
#0.062609

e_parent-ig_age
#0.1361454
table(df$d_age)

#***HHINCOME i�in Entropi Hesabi yapiyoruz.**

e_hincome1 = entropy(table(df$RESPONSE[df$d_hincome==1]))
#0.1261688
e_hincome0 = entropy(table(df$RESPONSE[df$d_hincome==0]))
#0.1477523

#Bilgi kazancini hesaplamak i�in proportion'lari hesapliyoruz. 
p1b = sum(df$d_hincome==1)/length(df$d_hincome)
#0.470249
p2b = sum(df$d_hincome==0)/length(df$d_hincome)
#0.529751

ig_hincome = e_parent-(p1b*e_hincome1+p2b*e_hincome0)
#0.06115177
table(df$d_hincome)

#***RSCORE i�in Entropi Hesabi yapiyoruz.**

e_rscore1 = entropy(table(df$RSCORE[df$d_rscore==1]))
#4.484406
e_rscore0 = entropy(table(df$RSCORE[df$d_rscore==0]))
#5.055153

#Bilgi kazancini hesaplamak i�in proportion'lari hesapliyoruz. 
p1c = sum(df$d_rscore==1)/length(df$d_rscore)
#0.5655746
p2c = sum(df$d_rscore==0)/length(df$d_rscore)
#0.4344254

ig_rscore = e_parent-(p1c*e_rscore1+p2c*e_rscore0)
#-4.533599
table(df$d_rscore)
#  0     1 
#8513 11083 



#***Burada Bilgi kazancinin kontrol�n� yapmamizin sebebi hangi degiskenin degeri y�ksek ise yani bilgi kazanci y�ksek ise o degiskeni se�ip kullanmak istiyorum. Burada veri setindeki diger degiskenlerde kullanilarak kontrol edilebilir ancak ben 2 tanesi se�tim**


#Numerik degiskenlerden olusacak bir veri seti olusturuyoruz

df1 <- df[c("RESPONSE", "AGE","HHINCOME","RSCORE", "TSPEND", "d_age", "d_hincome", "d_rscore")]

summary(df1$TSPEND)
quantile(df1$TSPEND, prob=c(0.99))
# 99% 
# 828 




#                         ** MODELLEME ** 

# Model yaratirken Train ve Test seti olarak ikiye ayiriyoruz. Bunu sebebi performansi �l��lebilir bir model olmasi. 



set.seed(202167009)
trainind <- createDataPartition(df1$AGE, p=0.75, list=F, times=1)
traindata <- df1[trainind,]
testdata <- df1[-trainind,]

summary(traindata$AGE); summary(testdata$AGE)




#                   ******Lineer Olasilik Modeli******



lpm <- lm(RESPONSE~AGE+HHINCOME+RSCORE+TSPEND, data=traindata)
summary(lpm)

p_lpm <- fitted(lpm)
summary(p_lpm)

#     Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
#-0.006023  0.024393  0.030241  0.030682  0.036732  0.059816 

lpm_class <- as.numeric(p_lpm>mean(p_lpm))
table(traindata$RESPONSE, lpm_class)
# �ikan sonuca g�re soldaki index'te tahmin edilen sagdaki indexte ise 
#   lpm_class
#    0    1
#0 7454 6794
#1  171  280

# table(traindata$RESPONSE)
#    0     1 
#  14248   451 



table_mat <- table(traindata$RESPONSE, lpm_class)
rownames(table_mat) <- paste("Actual", rownames(table_mat), sep = ":")
colnames(table_mat) <- paste("Predicted", colnames(table_mat), sep = ":")
print(table_mat)

#              Predicted:0   Predicted:1
# Actual:0        7454        6794
#Actual:1         171         280


#dogruluk alani hesaplayalim :

acc_test <- sum(diag(table_mat)/sum(table_mat))

280/451  #0.6208426
280/(280+6794)  #0.03958157


# Lineer Olasilik Modeli Sonucu

# Ben yaptigim d�zenleme ve model ile toplam %62 (0.6208426) oranindaki kitleyi yakalamisim(Ger�ek senaryoda bu rakam  d�s�k oldugu i�in EDA kismina ve Feature selection kismina tekrar d�nmek gerekir)





#                   ******Logistic Regresyon Modeli******

logit <- glm(RESPONSE~AGE+HHINCOME+RSCORE+TSPEND, data=traindata, family = binomial)
summary(logit)


traindata$pred <- predict(logit, newdata=traindata, type="response")


summary(traindata$pred)
#    Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
#0.007885 0.024086 0.029165 0.030682 0.035845 0.074627 


traindata$class <- as.numeric(traindata$pred>mean(traindata$pred))


table_mat <- table(traindata$RESPONSE, traindata$class)
rownames(table_mat) <- paste("Actual", rownames(table_mat), sep = ":")
colnames(table_mat) <- paste("Predicted", colnames(table_mat), sep = ":")
print(table_mat)

#          Predicted:0    Predicted:1
#Actual:0        8165        6083
#Actual:1         190         261

#dogruluk alani hesaplayalim :
acc_test <- sum(diag(table_mat)/sum(table_mat))
#0.5732363

261/451  #0.578714

#response Rate

261/(261+6083)   #0.04114124



    #  *****-----*****          SONU�           *****-----*****  #

# Biz elimizdeki veriyi kullanarak Lineer Olasilik ve Lineer Regresyon modellerini olusuturp kiyasladik.

# Linner Olasilik modelinde tahmin basarim %62 (0.6208426) olarak g�r�ld�. Lineer Regresyon modelinde ise tahmin basarim %58 (0.578714)  olarak g�r�ld�. Her iki �iktiyi da veri setinden aldigimiz i�in test setinde bu degerler daha d�s�k olacaktir. Ayrica Lineer Olasiligin geri d�n�s orani %0.3 0.03958157  Lineer Regresyonda ise bu oran %04 0.04114124 olarak g�r�ld�. Bu da geri d�n�s�n d�s�k olacagini g�steriyor. 

