def mergefields(df, mergedfiledlist,newfieldname=None, sep='#'):
    '''
    合并dataframe的两个字段
    
    '''
    def t(x):        
        for i in range(len(mergedfiledlist)):
            s = str(x[mergedfiledlist[i]]).strip()
            if not s:
                s = '无'
            str1 = str1 + sep +s if i != 0 else s
        return str1
    if not newfieldname:
        newfieldname = sep.join(mergedfiledlist)
    df[newfieldname] = df.apply(t,axis=1)
    return df

def merge_grouped_field(df,groupedfieldlist,mergedfield,sep=';'):
    '''
    把一个dataframe的某一字段合并，成为新的字段。
	YLJGDM	JZLSH	BGDLBBM	BGDLB	BBMC	BGDLB#BBMC
0	22985100300	0c8f9b508d8a4bb6a17fde1618710e27	1	无	全血	无#全血
1	22985100300	0c8f9b508d8a4bb6a17fde1618710e27	1	无	血浆	无#血浆
merge_grouped_field(l1,['YLJGDM','JZLSH'],'BGDLB#BBMC')
BGDLB#BBMC	JZLSH	YLJGDM
0	无#全血;无#血浆	0c8f9b508d8a4bb6a17fde1618710e27	22985100300
1	无#血清;无#血清	1d63671db3bd4a9cb64429f87800b403	22985100300
    @df: 要进行处理的字段
    @groupedfieldlist：聚合的字段
    @mergedfield：需要进行合并处理的字段
    @sep：合并字段连接时的分隔符
    '''
    df = df.groupby(groupedfieldlist)
    m = df[mergedfield].agg(lambda x: sep.join(x))
    dict1 = dict()
    dict1[mergedfield] = []
    for i in m.index:
        for ii in range(len(groupedfieldlist)):
            if groupedfieldlist[ii] not in dict1:
                dict1[groupedfieldlist[ii]] = []
            dict1[groupedfieldlist[ii]].append(i[ii])
        dict1[mergedfield].append(m[i])
    df1 = pd.DataFrame(dict1)
    return df1