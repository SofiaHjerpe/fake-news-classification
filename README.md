# fake-news-classification
Falska nyheter kan syfta till att få en inkorrekt bild av forskning och det demokratiska samhället. Jag vill skapa ett system som skiljer korrekt information från felaktig information.Användaren delar med sig av en text och programmet avgör sedan om det är “fake news” eller ej. Programmet kommer ge ett binärt utfall. Problemet jag kommer att försöka lösa är ett klassificeringsproblem. "Binary logistic regression" svarar bra på mitt problem då jag är ute efter att klassificera texter som falska eller riktiga. Jag vill använda mig av Logistic regression eftersom modellen ska hantera binära utfall. Jag väljer att använda mig av Logistic regression då den är specialiserad på binär klassificering och för att den är snabb. Det är också bra för att klassificera data baserat på obalanderad data.

Jag hittade tidigare Github projekt där de använde data från CSV filer. Det var två CSV filer: True.csv och Fake.csv (se Github länk). Fake.csv har 23 538 rader medan True.csv har 21 418 rader. Jag valde att tillämpa de här CSV filerna då jag bedömde att de hade en god kvalitet. Planen var sedan att testa modellen på nya artiklar. Artiklarna är från ca. 2017. Det senare är något som kan påverka resultatet och att nyheter i systemet klassificeras som falska fast de är sanna.Jag har även skapat 2 egna csv filer vid namn 'falska_nyheter.csv' och 'sanna_nyheter.csv'. Det som skiljer dessa dataset är att de är på svenska och att de har 354 respektive 295 rader. De flesta nyheterna i dataseten är skapade i år. Att jag skapade egna csv filer med svenska nyheter gör att systemet fungerar även på svenska.
Null värden finns inte i datasetet. Det finns fyra kolumner i respektive dataset, och alla har typen objekt. Kolumnerna heter “title”, “text”, “subject” och “date”. Jag kommer att använda mig av “title” och “text”. Min data är inte labeled.  Vilken typ av inlärning som jag tillämpar är något som jag får undersöka vidare. Att använda mig av övervakad inlärning med logistisk regression känns lämpligt. Övervakad inlärning känns mer lämplig då den möjliggör att ha mer kontroll över inlärningen samt att med hjälp av erfarenhet optimera prestationskriterier. Om det här datasetet inte fungerar tillsammans med övervakad inlärning får jag istället använda mig av oövervakad inlärning. Då får modellen själv hitta mönster och samband.  

Det finns olika sätt att konvertera data till numeriskt format. Jag kommer att använda TfidfVectorizer från Scikit learn.För att träna programet planerar jag att arbeta med Scikit learn och train_test_split funtionen.  

## Källor
https://github.com/menanashat/Fake_News_Detection 

https://www.geeksforgeeks.org/how-to-convert-categorical-variable-to-numeric-in-pandas/

https://www.geeksforgeeks.org/supervised-unsupervised-learning/

https://www.ibm.com/topics/logistic-regression 

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html 

https://www.svt.se/nyheter/inrikes/tre-falska-nyheter-som-alla-trodde-var-pa-riktigt 
