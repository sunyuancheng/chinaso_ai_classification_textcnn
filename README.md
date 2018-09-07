# chinaso_ai_classification_textcnn

接口：
反例接口：
http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01344/search
?fields=id,wtitle,wcaption,newsLabel&filters=EQS_resourceState,4&fetchsize=15

恐怖数据：
http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01344/search
?fields=id,wcaption&filters=EQS_resourceState,4
|EQS_newsLabel,恐怖&pagestart=1&fetchsize=15

正常数据：
http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01276/search
?fields=id,wcaption&filters=EQS_ifCompare,1|EQS_resourceState,4|&orders=wpubTime_desc
&pagestart=1&fetchsize=15