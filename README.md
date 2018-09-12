# chinaso_ai_classification_textcnn

接口：
反例接口：
http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01344/search
?fields=id,wtitle,wcaption,newsLabel&filters=EQS_resourceState,4&pagestart=1&fetchsize=15

恐怖数据：
http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01344/search
?fields=id,wcaption&filters=EQS_resourceState,4
|EQS_newsLabel,恐怖&pagestart=1&fetchsize=15

色情数据：
http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01344/search?fields=id,wcaption&filters=EQS_resourceState,4|EQS_newsLabel,%E8%89%B2%E6%83%85&pagestart=1&fetchsize=15
http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01344/search
?fields=id,wcaption&filters=EQS_resourceState,4
|EQS_newsLabel,色情&pagestart=1&fetchsize=15

正常数据：第二标签=时政滚动
http://data.mgt.chinaso365.com/datasrv/2.0/news/resources/01276/search?fields=id,wcaption&filters=EQS_ifCompare,1|EQS_resourceState,4|EQS_newsLabelSecond,%E6%97%B6%E6%94%BF%E6%BB%9A%E5%8A%A8&orders=wpubTime_desc&pagestart=1&fetchsize=15