## tencent ad 2018 lookalike

不同于bryan大佬的[baseline], 在做完 [OneHotEncoder] 和 [CountVectorizer] 之后，feature会转变成稀疏特征，维度非常大，训练时间也会非常的长。

这个baseline，对于类别特征做了整体统计的权重化处理以及按aid统计的权重化处理，针对多取值特征做了整体统计的权重化处理以及按aid统计的权重化处理，并且做了一个 base_ctr 的特征.。

线上成绩：0.722678


[baseline]: http://127.0.0.1:50723/Dash/iuwccypq/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder

[OneHotEncoder]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder

[CountVectorizer]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer