import numpy as np


def preprocessing(df):
    """
    This function performes feature engineering and adds new statistical features to the df dataframe
    """
    if "id" and "target" in df:
        data = df.drop(["id","target"],axis=1)
    else:
        data = df
    
    # compute arithmetic mean along axis 1 (row wise)
    df_mean = np.mean(data,axis=1)
    # compute standard deviation along axis 1 (row wise)
    df_std = np.std(data,axis=1)
    
    # compute 25th quantile along axis 1 (row wise)
    df_quantile25 = np.quantile(data,0.25,axis=1)
    # compute 50th quantile along axis 1 (row wise)
    df_quantile50 = np.quantile(data,0.50,axis=1)
    # compute 75th quantile along axis 1 (row wise)
    df_quantile75 = np.quantile(data,0.75,axis=1)
    
    # compute trigonometric sine, element-wise then take mean of each row.
    sine = np.sin(data)
    df_sine = np.mean(sine,axis=1)
    # compute trigonometric cosine, element-wise then take mean of each row.
    cosine = np.cos(data)
    df_cosine = np.mean(cosine,axis=1)
    # compute trigonometric tangent, element-wise then take mean of each row.
    tangent = np.tan(data)
    df_tangent = np.mean(tangent,axis=1)
    # compute trignometric inverse sine, element-wise then take mean of each row.
    inversesine = np.arcsin(data)
    df_inversesine = np.mean(inversesine,axis=1)
    # compute trigonometric inverse cosine, element-wise then take mean of each row.
    inversecosine = np.arccos(data)
    df_inversecosine = np.mean(inversecosine,axis=1)
    # compute trigonometric inverse tangent, element-wise then take mean of each row.
    inversetangent = np.arctan(data)
    df_inversetangent = np.mean(inversetangent,axis=1)
    
    # compute hyperbolic sine, element-wise then take mean of each row.
    hyperbolicsine = np.sinh(data)
    df_hyperbolicsine = np.mean(hyperbolicsine,axis=1)
    # compute hyperbolic cosine, element-wise then take mean of each row.
    hyperboliccosine = np.cosh(data)
    df_hyperboliccosine = np.mean(hyperboliccosine,axis=1)
    # compute hyperbolic tangent, element-wise then take mean of each row.
    hyperbolictangent = np.tanh(data)
    df_hyperbolictangent = np.mean(hyperbolictangent,axis=1)
    
    # compute exponential, element-wise then take mean of each row.
    exponential = np.exp(data)
    df_exponential = np.mean(exponential,axis=1)
    # compute natural logarithm, element-wise then take mean of each row.
    expm1 = np.expm1(data)
    df_expm1 = np.mean(expm1,axis=1)
    # compute 2**p for all p in the input array, element-wise then take mean of each row.
    exp2 = np.exp2(data)
    df_exp2 = np.mean(exp2,axis=1)
    
    # compute array elements raise to the power of 2, element-wise then take mean of each row.
    poweroftwo = np.power(data,2)
    df_poweroftwo = np.mean(poweroftwo,axis=1)
    # compute array elements raise to the power of 3, element-wise then take mean of each row.
    powerofthree = np.power(data,3)
    df_powerofthree = np.mean(powerofthree,axis=1)
    # compute array elements raise to the power of 4, element-wise then take mean of each row.
    poweroffour = np.power(data,4)
    df_poweroffour = np.mean(poweroffour,axis=1)
    
    # add various new statistical features to df dataset
    df['mean'] = df_mean
    df['std'] = df_std
    
    df["quantile25"] = df_quantile25
    df["quantile50"] = df_quantile50
    df["quantile75"] = df_quantile75
    
    df['sine'] = df_sine
    df['cosine'] = df_cosine
    df['tangent'] = df_tangent
    df['inversesine'] = df_inversesine
    df['inversecosine'] = df_inversecosine
    df['inversetangent'] = df_inversetangent
    
    df['hyperbolicsine'] = df_hyperbolicsine
    df['hyperboliccosine'] = df_hyperboliccosine
    df['hyperbolictangent'] = df_hyperbolictangent
    
    df['exponential'] = df_exponential
    df['exponentialm1'] = df_expm1
    df["exponential2"] = df_exp2
    
    df['poweroftwo'] = df_poweroftwo
    df['powerofthree'] = df_powerofthree
    df['poweroffour'] = df_poweroffour
    return df