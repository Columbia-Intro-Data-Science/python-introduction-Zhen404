""" thsi script imputes missing values in a dataset """
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.conf import SparkConf
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import StringIndexer,OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
import numpy as np
from scipy import stats
import re
import math,sys
from itertools import chain

sc = SparkContext()
sqlContext = SQLContext(sc)
def missing_percent(data):
    """ calculate the percentage of missing value in  
        input type
        data: sqlDataFrame
        return type
        data_missing: sqlDataFrame
    """
    data_missing=data.select(*(F.sum(F.col(c).isNull().cast("int")).alias(c) for c in data.columns))
    num_rows=data.count()
    for field in data_missing.schema.fields:
        name=str(field.name)
        data_missing=data_missing.withColumn(name,F.col(name)/num_rows)
    return data_missing
def get_check_na(data_missing):
    """ get column names (to do t-test) and column names with na
        input type
        data: sqlDataFrame
        return type
        [check_names,na_names]: nested list
    """
    data_missing=data_missing.toPandas().to_dict()
    check_names=list()
    na_names=list()
    for key, value in data_missing.items():
        if data_missing[key][0]>0.8:
            check_names.append(key)
        if data_missing[key][0]>0:
            na_names.append(key)
    return [check_names,na_names]

def get_cat_names(data):
    """ get categorical column names
        input type
        data: sqlDataFrame
        return type
        cat_names=list
    """
    cat_names=list()
    count=0
    for field in data.schema.fields:
        if str(field.dataType)=='StringType':
            cat_names.append(data.schema.names[count])
        count+=1
    return cat_names

def get_dummy(data,cat_names):
    """ get_dummy for categorical column
        input type
        data: sqlDataFrame
        cat_names: list of String(all categorical column names)
        return type
        data_dummy: sqlDataFrame
    """
    expr_list=list()
    for name in cat_names:
        dic=data.select(name).distinct().rdd.flatMap(lambda x:x).collect()
        dic=list(set(list(map((lambda x : re.sub(r'[^\w]','',x)),dic))))
        dic=list(set(list(map((lambda x : x.upper()),dic))))
        name_list=[F.when(F.col(name)==k,1).otherwise(0).alias(name+k) for k in dic]
        expr_list=expr_list+name_list
        data_dummy=data.select(other_names+expr_list)
        return data_dummy

def t_test(na_df,nona_df,name):
    """ do t-test for column 'name'
        input type
        na_df:sqlDataFrame
        nona_df:sqlDataFrame 
        name:string (column name we do t-test on)
        return type
        p: float (p-value)
    """
    na_ttest=na_df.select(name).toPandas()[name]
    nona_ttest=nona_df.select(name).toPandas()[name]
    t,p=stats.ttest_ind(na_ttest,nona_ttest,equal_var=True,nan_policy='omit')
    return p

def search_time_pattern(na_df,name):
    """ search pattern for column 'name'
        input type
        na_df:sqlDataFrame 
        name:string (datetype column name we want to find pattern on)
        return type
        acor: list (autocorrealtion list)
    """
    na_test_date=na_df.select(monotonically_increasing_id().alias("rowId"),name)
    na_test_date=na_test_date.withColumn('DATE',na_test_date[name].cast(DateType())).orderBy('DATE')
    my_window=Window.partitionBy().orderBy('rowId')
    na_test_date=na_test_date.withColumn("lag_date",F.lag(na_test_date.DATE).over(my_window))
    na_test_date=na_test_date.withColumn("diff", F.datediff(na_test_date.DATE,na_test_date.lag_date))

    time_diff=na_test_date.select('diff').toPandas()['diff'].tolist()[1:]
    time_diff_unbiased=time_diff-np.mean(time_diff)
    time_diff_norm=np.sum(time_diff_unbiased**2)
    acor=np.correlate(time_diff_unbiased,time_diff_unbiased,"full")/time_diff_norm
    acor=acor[math.floor(len(acor)/2):][1:]

    return acor

def MCAR_step(data,cat_names,check_names):
    """ MCAR step
        input type
        data: sqlDataFrame
        cat_names: list of string (all categorical column names)
        check_names: list of string(all column names need to do t-test)
        return type
        data: sqlDataFrame
    """
    for name1 in check_names:
        cat_names_cp=cat_names
        other_names_cp=other_names
        print "========================="
        print "=========="+name1
        count=0
        if name1 in cat_names_cp:
            cat_names_cp.remove(name1)
        else:
            other_names_cp.remove(name1)
        data_dummy=get_dummy(data, cat_names_cp)
        na_test=data_dummy.where(col(name1).isNull())
        nona_test=data_dummy.where(col(name1).isNotNull())
        for name2,dtype in data_dummy.dtypes:
            if dtype=='int' or dtype=='double':
                #Do t-test
                if na_test.select(name2).where(col(name2).isNotNull()).count()==0:
                    print "========="+name2+' does not pass the t-test'
                    count+=1
                else:
                    p=t_test(na_test,nona_test,name2)
                    if p>0.05:
                        print "========="+name2+' does not pass the t-test'
                        count+=1
                    else:
                        continue
            if dtype=='timestamp' or dtype=='date':
                #search pattern
                acor=search_time_pattern(na_test,name2)
                if len(acor[acor>0.8])>=1:
                    print '===========missing values appear in pattern on '+name2
                    count+=1
                else:
                    continue
            if count==0:
                na_names.remove(name1)
                print "==========Column "+name1+" dropped"
            else:
                print "==========Missing value in "+name1+" is not MCAR"
                print "==========Can not drop directly!"

    return data

def init_impute(data):
    """ Do simple initial imputatoin
        input type
        data:sqlDataFrame
        return type
        data:sqlDataFrame
    """
    for name,dtype in data.dtypes:
        if dtype=='int':
            data=data.fillna({name:round(data.select(F.mean(F.col(name)).alias('mean')).collect()[0]['mean'])})
        elif dtype=='double':
            data=data.fillna({name:round(data.select(F.mean(F.col(name)).alias('mean')).collect()[0]['mean'],4)})
        else:
            count_df=data.groupBy(name).count()
            mode=count_df.orderBy(count_df['count'].desc()).collect()[0][0]
            data=data.fillna({name:mode})
    return data

def logistic_impute(data,name,other_names,cat_names,na_index,nona_index):
    """ Use logistic regression to predict missing value on column 'name'
        input type
        data_use: sqlDataFrame
        name: string (column name we want to impute)
        other_names: list of string (non-categorical column names(exclude date))
        cat_names: list of string (categorical column names)
        return type
        result list (to replace the cooresponding value)
    """
    column_vec_in=cat_names
    column_vec_out=[s+'_catVec' for s in column_vec_in]

    indexers=[StringIndexer(inputCol=x,outputCol=x+'_tmp') for x in column_vec_in]
    encoders=[OneHotEncoder(dropLast=True,inputCol=x+'_tmp',outputCol=y) for x,y in zip(column_vec_in,column_vec_out)]

    tmp=[[i,j] for i,j in zip(indexers,encoders)]
    tmp=[i for sublist in tmp for i in sublist]

    cols_now=other_names+column_vec_out

    assembler_features=VectorAssembler(inputCols=cols_now,outputCol='features')
    labelIndexer=StringIndexer(inputCol=name,outputCol='label')
    tmp+=[assembler_features,labelIndexer]

    pipeline=Pipeline(stages=tmp)
    output=pipeline.fit(data).transform(data)
    dict_df=output.select(col(name).alias('category'),col('label').alias('categoryIndex')).distinct().collect()
    map_dict={row.categoryIndex:row.category for row in dict_df}

    train_data=output.where(F.col('_c0').isin(nona_index)).select('label','feature')
    test_data=output.where(F.col('_c0').isin(na_index)).select('label','feature')

    lgr=LogisticRegression(labelCol='label',maxIter=10,regParam=0.1, elasticNetParam=1.0, family="multinomial")
    logisticModel=lgr.fit(train_data)
    predicted=logisticModel.transform(test_data)
    mapping_expr=F.create_map([F.lit(x) for x in chain(*map_dict.items())])
    predicted=predicted.withColumn("reverse",mapping_expr.getItem(col("prediction")))
    result=predicted.select('reverse').toPandas()['reverse'].tolist()

    return result

def lasso_impute(data,name,other_names,cat_names,na_index,nona_index):
    """ Use Lasso regression to predict missing value on column 'name'
        input type
        data: sqlDataFrame
        name: string (column name we want to impute)
        other_names: list of string (non-categorical column names(exclude date))
        cat_names: list of string (categorical column names)
        return type
        result: list (to replace the cooresponding value)
    """
    column_vec_in=cat_names
    column_vec_out=[s+'_catVec' for s in column_vec_in]

    indexers=[StringIndexer(inputCol=x,outputCol=x+'_tmp') for x in column_vec_in]
    encoders=[OneHotEncoder(dropLast=True,inputCol=x+'_tmp',outputCol=y) for x,y in zip(column_vec_in,column_vec_out)]

    tmp=[[i,j] for i,j in zip(indexers,encoders)]
    tmp=[i for sublist in tmp for i in sublist]

    cols_now=other_names+column_vec_out

    assembler_features=VectorAssembler(inputCols=cols_now,outputCol='features')
    tmp+=[assembler_features]

    pipeline=Pipeline(stages=tmp)
    output=pipeline.fit(data).transform(data)
    train_data=output.where(F.col('_c0').isin(nona_index)).select(name,'features')
    test_data=output.where(F.col('_c0').isin(na_index)).select(name,'features')

    lr=LinearRegression(labelCol=name,maxIter=10, regParam=0.3,elasticNetParam=1)
    linearModel=lr.fit(train_data)
    predicted=linearModel.transform(test_data)
    result=predicted.select('prediction').toPandas()['prediction'].tolist() 

    return result

def replace_wr(line,name,l):
    """ replace value with round
        input type
        line: Row object
        name: Column name where we want to replace value
        l: List of value we want to replace
        return type:
        line: Row object
    """ 
    line=line.asDict()
    line[name]=round(l.pop())
    line=Row(**line)
    return line

def replace_wor(line,name,l):
    """ replace result without round
        input type
        line: Row object
        name: Column name where we want to replace value
        l: List of value we want to replace
        return type:
        line: Row object 
    """

    line=line.asDict()
    line[name]=round(l.pop(),4)
    line=Row(**line)
    return line

def replace_value(data,name,na_index,result):
    """ The whole procedure of updating dataframe
        input type
        data: sqlDataFrame
        name: string (name of column we want to replace)
        na_index: index of na value under name
        result: list (values we want to replace with)
        return type
        data: sqlDataFrame 
    """
    if isinstance(data.schema[name],DoubleType) or isinstance(data.schema[name],StringType):
        data_replace=data.where(F.col("_c0").isin(na_index)).rdd.map(lambda x: replace_wor(x,name,result))
        data_replace=sqlContext.createDataFrame(data_replace,samplingRatio=1)
        data_replace=data_replace.select(data.columns)
        data=data.where(~F.col('_c0').isin(na_index))
        data=data.union(data_replace)
    else:
        data_replace=data.where(F.col("_c0").isin(na_index)).rdd.map(lambda x: replace_wr(x,name,result))
        data_replace=sqlContext.createDataFrame(data_replace,samplingRatio=1)
        data_replace=data_replace.select(data.columns)
        data=data.where(~F.col('_c0').isin(na_index))
        data=data.union(data_replace)

    return data



def data_cleansing(data):               
    #calculate percentage of NA
    data_missing=missing_percent(data)
    check_names,na_names=get_check_na(data_missing)
    cat_names=get_cat_names(data)
    other_names=list(set(data.columns).difference(set(cat_names)))
    if len(check_names)>0:
        print "=========There exists columns containing more than 80 percent missing values"
        print "=========Enter MCAR Identification Step"
        #MCAR identification step
        data=MCAR_step(data)
    else:
        print "==========There is no columns containing more than 80 percent missing values"
        print "==========Skip MCAR Identification Step"
        print "==========Enter Drop Rows Step"
    #Drop rows
    data=data.withColumn("column_null",sum([F.when(F.col(x).isNull(),1).otherwise(0) for x in data.columns]))
    col_num=len(data.columns)
    data=data.withColumn("column_null",F.col("column_null")/col_num)
    data=data.where(F.col("column_null")<0.8)
    num_droprow=data.where(F.col("column_null")>=0.8).count()
    if num_droprow!=0:
        print "========"+str(num_droprow)+" dropped"
    else:
        print "========All rows containing less then 20% missing values"
    print "=========Drop Rows Step Done!"
    print "=========Enter MICE Step"
    #MICE
    data_nd=data.select([c for c,dtype in data.dtypes if dtype!='timestamp' and dtype!='date'])
    data_use=data.select([c for c,dtype in data.dtypes if dtype!='timestamp' and dtype!='date'])
    for name,dtype in data.dtypes:
        if dtype=='timestamp' or dtype=='date':
            other_names.remove(name)
    #initial simple imputation
    data_use=init_impute(data_use)
    #Iteraton Step
    iter_num=1
    for i in list(range(iter_num)):
        print "==========Round "+str(i+1)

        for name in na_names:
            cat_names_cp=cat_names
            other_names_cp=other_names
            
            print "Column "+name+" imputation" 
            na_index=data_nd.where(F.col(name).isNull()).select('_c0').toPandas()['_c0'].tolist()
            nona_index=data_nd.where(F.col(name).isNotNull()).select('_c0').toPandas()['_c0'].tolist()

            if isinstance(data_use.schema[name].dataType,StringType):
                cat_names_cp.remove(name)
                result=logistic_impute(data_use,name,other_names_cp,cat_names_cp,na_index,nona_index)
                print "==========prediction completed"
            else:
                other_names_cp.remove(name)
                result=lasso_impute(data_use,name,other_names_cp,cat_names_cp,na_index,nona_index)
                print "==========prediction completed"
            result=result[::-1]   
            
            data_use=replace_value(data_use,name,na_index,result)

    return data_use






def main(argv):

    data=sqlContext.read.format('csv').options(header='true',inferschema='true').load(argv[1])

    data1 = data_cleansing(data)
    print(data1.count())
    data1.write.csv(argv[2])

if __name__ == "__main__":
    main(sys.argv)
