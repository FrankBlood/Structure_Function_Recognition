time ~/word2vec/word2vec -train ../data/cs.para.all -output ../data/para.all.vec.mincount.5 -cbow 0 -min-count 5 -size 200 -window 5 -negative 0 -his 1 -sample 1e-3 -threads 12 -binary 0
~/word2vec/distance ../data/para.all.vec.mincount.5
