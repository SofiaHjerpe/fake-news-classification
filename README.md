# fake-news-classification
Falska nyheter kan syfta till att få en inkorrekt bild av forskning och det demokratiska samhället. Jag vill skapa ett system som skiljer korrekt information från felaktig information. Mitt mål är skapa ett program som kan avgöra om data klassificeras som “fake news” eller inte. Användaren delar med sig av en text och programmet avgör sedan om det är “fake news” eller ej. Programmet kommer ge ett binärt utfall. Problemet jag kommer att försöka lösa är ett klassificeringsproblem. Jag vill använda mig av Logistic regression eftersom modellen ska hantera binära utfall. "Binary logistic regression" svarar bra på mitt problem då jag är ute efter att klassificera texter som falska eller riktiga. 

Jag hittade tidigare Github projekt där de använde data från CSV filer. Det var två CSV filer: True.csv och Fake.csv (se Github länk). Fake.csv har 23 538 rader medan True.csv har 21 418 rader. Jag väljer att tillämpa de här CSV filerna då jag bedömer att de har en god kvalitet. Planen är sedan att testa modellen på nya artiklar. Null värden finns inte i datasetet. Det finns fyra kolumner i respektive dataset, och alla har typen objekt. Kolumnerna heter “title”, “text”, “subject” och “date”. Jag kommer att använda mig av “title” och “text”. Min data är inte labeled. Vilken typ av inlärning som jag tillämpar är något som jag får undersöka vidare. Att använda mig av övervakad inlärning med logistisk regression känns lämpligt. Övervakad inlärning känns mer lämplig då den möjliggör att ha mer kontroll över inlärningen samt att med hjälp av erfarenhet optimera prestationskriterier. Om det här datasetet inte fungerar tillsammans med övervakad inlärning får jag istället använda mig av oövervakad inlärning. Då får modellen själv hitta mönster och samband.  

Det finns olika sätt att konvertera data till numeriskt format. Jag kommer att använda OrdinalEncoder från Scikit learn. Denna funktion kodar om kategoriska data till en numerisk lista. Andra metoder jag kan tillämpa är get_dummies() eller replace(). För att träna programet planerar jag att arbeta med Scikit learn och train_test_split funtionen.  

## Källor
https://github.com/menanashat/Fake_News_Detection 

https://www.geeksforgeeks.org/how-to-convert-categorical-variable-to-numeric-in-pandas/

https://www.geeksforgeeks.org/supervised-unsupervised-learning/

https://www.ibm.com/topics/logistic-regression 

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html 

https://www.svt.se/nyheter/inrikes/tre-falska-nyheter-som-alla-trodde-var-pa-riktigt 
