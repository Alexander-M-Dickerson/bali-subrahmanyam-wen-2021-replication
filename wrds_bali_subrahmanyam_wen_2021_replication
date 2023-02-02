#* ************************************** */
#* Libraries                              */
#* ************************************** */ 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas_datareader as pdr
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
import datetime as datetime
from pandas.tseries.offsets import *
import pyreadstat
import wrds
import pandasql as ps
tqdm.pandas()

#* ************************************** */
#* Connect to WRDS                        */
#* ************************************** */  
db = wrds.Connection()

#* ************************************** */
#* WRDS Bond Returns                      */
#* ************************************** */  
traced = db.raw_sql("""SELECT *                 
                  FROM wrdsapps.bondret
                  """)

#* ************************************** */
#* Download Mergent File                  */
#* ************************************** */  
fisd_issuer = db.raw_sql("""SELECT issuer_id,country_domicile,sic_code                
                  FROM fisd.fisd_mergedissuer 
                  """)

fisd_issue = db.raw_sql("""SELECT complete_cusip, issue_id,
                  issuer_id, foreign_currency,
                  coupon_type,coupon,convertible,
                  asset_backed,rule_144a,
                  bond_type,private_placement,
                  interest_frequency,dated_date,
                  day_count_basis,offering_date,
                  offering_amt
                  FROM fisd.fisd_mergedissue  
                  """)
                  
fisd = pd.merge(fisd_issue, fisd_issuer, on = ['issuer_id'], how = "left")                              
#* ************************************** */
#* Apply BBW Bond Filters                 */
#* ************************************** */  
#1: Discard all non-US Bonds (i) in BBW
fisd = fisd[(fisd.country_domicile == 'USA')]

#2.1: US FX
fisd = fisd[(fisd.foreign_currency == 'N')]

#3: Must have a fixed coupon
fisd = fisd[(fisd.coupon_type != 'V')]

#4: Discard ALL convertible bonds
fisd = fisd[(fisd.convertible == 'N')]

#5: Discard all asset-backed bonds
fisd = fisd[(fisd.asset_backed == 'N')]

#6: Discard all bonds under Rule 144A
fisd = fisd[(fisd.rule_144a == 'N')]

#7: Remove Agency bonds, Muni Bonds, Government Bonds, 
mask_corp = ((fisd.bond_type != 'TXMU')&  (fisd.bond_type != 'CCOV') &  (fisd.bond_type != 'CPAS')\
            &  (fisd.bond_type != 'MBS') &  (fisd.bond_type != 'FGOV')\
            &  (fisd.bond_type != 'USTC')   &  (fisd.bond_type != 'USBD')\
            &  (fisd.bond_type != 'USNT')  &  (fisd.bond_type != 'USSP')\
            &  (fisd.bond_type != 'USSI') &  (fisd.bond_type != 'FGS')\
            &  (fisd.bond_type != 'USBL') &  (fisd.bond_type != 'ABS')\
            &  (fisd.bond_type != 'O30Y')\
            &  (fisd.bond_type != 'O10Y') &  (fisd.bond_type != 'O3Y')\
            &  (fisd.bond_type != 'O5Y') &  (fisd.bond_type != 'O4W')\
            &  (fisd.bond_type != 'CCUR') &  (fisd.bond_type != 'O13W')\
            &  (fisd.bond_type != 'O52W')\
            &  (fisd.bond_type != 'O26W')\
            # Remove all Agency backed / Agency bonds #
            &  (fisd.bond_type != 'ADEB')\
            &  (fisd.bond_type != 'AMTN')\
            &  (fisd.bond_type != 'ASPZ')\
            &  (fisd.bond_type != 'EMTN')\
            &  (fisd.bond_type != 'ADNT')\
            &  (fisd.bond_type != 'ARNT'))
fisd = fisd[(mask_corp)]

#8: No Private Placement
fisd = fisd[(fisd.private_placement == 'N')]

#9: Remove floating-rate, bi-monthly and unclassified coupons
fisd = fisd[(fisd.interest_frequency != 13) ] # Variable Coupon (V)

#10 Remove bonds lacking information for accrued interest (and hence returns)
fisd['offering_date']            = pd.to_datetime(fisd['offering_date'], 
                                                  format='%Y-%m-%d')
fisd['dated_date']               = pd.to_datetime(fisd['dated_date'], 
                                                  format='%Y-%m-%d')

fisd.rename(columns={'complete_cusip':'cusip'}, inplace=True)


#* ************************************** */
#* Merge                                  */
#* ************************************** */ 
fisd_w = fisd[['cusip','sic_code']]

df = traced.merge(fisd_w, left_on = ['cusip'], right_on = ['cusip'], 
                  how = "inner")
df = df.set_index(['date','cusip']).sort_index(level = "cusip")

nBonds = df.groupby("date")['issue_id'].count()
nBonds.plot()


#* ************************************** */
#* Compute maturity                       */
#* ************************************** */  
df = df.reset_index()
df['maturity'] = pd.to_datetime(df['maturity'])
df['date']     = pd.to_datetime(df['date'])

df['bond_maturity'] = ((df.maturity -\
                        df.date)/np.timedelta64(1, 'M')) / 12
#* ************************************** */
#* Variable choice                        */
#* ************************************** */  

# Bond Amtout 
df['bond_amount_out'] = df['amount_outstanding']

# Bond Rating
df['spr_mr_fill'] = np.where(df['n_sp'].isnull(),df['n_mr'],df['n_sp'])
df['mr_spr_fill'] = np.where(df['n_mr'].isnull(),df['n_sp'],df['n_mr'])
df['rat']         = (df['spr_mr_fill']+df['mr_spr_fill'])/2
# Bond Yield 
df['bond_yield'] = df['yield'] 

# Bond Maturity
df['tmt'] = df['bond_maturity']

# Risk-free rate #
ff3 = pd.read_csv\
    ('rf.csv')
    
ff3 = ff3[['date','RF']]       
ff3['date'] = pd.to_datetime(ff3['date'], format = "%Y%m")
ff3['date']  = ff3['date']  + MonthEnd(0)
ff3 = ff3.set_index(['date'])
ff3 = ff3 / 100

# Macroeconomic Uncertainty Factor #
unc = pd.read_csv('macro_uncertainty.csv')
unc['date'] = pd.to_datetime(unc['date'] , format = "%Y-%m-%d")
unc['date'] = unc['date']+MonthEnd(0)
unc = unc.set_index(['date'])
unc = unc['unc'].diff()

# read BBW Factors #
bbw = pd.read_csv('bbw4_original.csv')
bbw['date'] = pd.to_datetime(bbw['date'] , format = "%Y%m")
bbw['date'] = bbw['date']+MonthEnd(0)
bbw = bbw.set_index(['date'])
bbw = bbw/100

# Merge
df = df.set_index(['date','cusip'])
df = df.merge(ff3, how = "inner", left_index=True, 
              right_index = True)
df['exretn'] = df['ret_l5m'] - df['RF']

df = df.merge(unc, how = "inner", left_index=True, 
              right_index = True)
df = df.merge(bbw, how = "inner", left_index=True, 
              right_index = True)

df = df[~df.index.duplicated()]
#* ************************************** */
#* Subset columns                         */
#* ************************************** */ 
df.rename(columns={'ret_l5m':'bond_ret',
                   'yield':'yld',
                   'price_l5m':'bond_prc'}, inplace=True) 
df['Q_M'] = 'T'
df = df[[ 'exretn' , 'bond_ret','bond_prc', 'bond_amount_out' ,'offering_amt' , 
          'spr_mr_fill','mr_spr_fill', 'rat',
          'yld','tmt','Q_M' ,'sic_code',
          'unc','MKTbond']]
 
#* ************************************** */
#* Subset columns                         */
#* ************************************** */ 
import statsmodels.api as sm
def rolling_sub_all( exret, factors ):       
        model2 = sm.OLS(exret, sm.add_constant(factors), missing='drop').fit()  
        out    =  list( model2.params[1:]  )  
        return out

def rolling_risk(dfstock, factors, returns, window, min_periods):             
        risk_prices = pd.DataFrame(index = range(0,len(dfstock)), columns = ['A'] )
        risk_prices['A'] = risk_prices['A'].astype('object')
        
        
        exret       = dfstock[returns].values
        factors     = dfstock[factors].values
        
                
        for i in range(len(dfstock)+1):
              if i < min_periods:                  
                  continue
              else:
                if np.sum(~np.isnan(exret))  >= min_periods:
                    try:
                        if i < window:
                            risk_prices.at[i-1,'A'] = rolling_sub_all(exret[0:i],factors[0:i,:]) 
                        elif i >= window:
                            risk_prices.at[i-1,'A'] = rolling_sub_all(exret[i-window:i],factors[i-window:i,:])                                           
                    except ValueError:                                                
                        risk_prices[i-1,] = np.nan
                else:
                  continue          
        dfstock['risk_prices'] = risk_prices['A'].tolist()
        dfstock.reset_index(level=dfstock.index.names[0], inplace=True)
        return dfstock['risk_prices']      


#* ************************************** */
#* Subset columns                         */
#* ************************************** */ 
dfBeta = df[['exretn',
             'MKTbond',
             'unc',
             'bond_prc']].dropna()

dfBeta = dfBeta[(dfBeta.bond_prc > 5)]
dfBeta = dfBeta[(dfBeta.bond_prc < 1000)]
     
factors = [ 'MKTbond',
            'unc']

returns = ['exretn']

dfBeta = dfBeta.reset_index().set_index(['cusip','date'])

#* ************************************** */
#* Run Rolling regression                 */
#* ************************************** */ 
RiskPrices = dfBeta.groupby(\
                       level=dfBeta.index.names[0]).progress_apply( rolling_risk, 
                       factors = factors, returns = returns, 
                       window = 36 , min_periods = 24).to_frame()

RiskPrices.columns = ['tolistcol']
RiskPrices = RiskPrices.tolistcol.progress_apply(pd.Series)

RiskPrices.columns = [   'mktb','uncb'  ]
RiskPrices = RiskPrices.sort_index(level = "cusip")
RiskPrices = RiskPrices.reset_index()


# Merge #
df = df.reset_index().merge(RiskPrices, how = "left", left_on  = ['cusip','date'],
                                    right_on = ['cusip','date'])


#* ************************************** */
#* Forward return                         */
#* ************************************** */  
# Lead return so we sort on the beta available
# to the investor at time t,
# and record the return from t:t+1
# we lag the final sorts at the end to align the dat index
df['exret(t+1)']  = df.groupby("cusip")['exretn'].shift(-1)
df['date(t+1)']   = df.groupby("cusip")['date']  .shift(-1)

# Ensure that between each return observation, there
# is exactly a 1-Month gap (i.e. maximum of 31-Days)
# The resampling should have taken care of this, however
# this provides an additional check
df['days'] = (df['date(t+1)'] - df['date']).dt.days
df = df[df['days'] <= 31]

#* ************************************** */
#* BSW Filters      --Main df             */
#* ************************************** */ 
df = df[df.tmt       > 1]
df = df[(df.bond_prc > 5)]
df = df[(df.bond_prc < 1000)]


# Save dataframe
dfSave = df

#* ************************************** */
#* Drop NaN                               */
#* ************************************** */  
df = df[~df.mktb.isnull()]
df = df[~df.uncb.isnull()]
df = df[~df['exret(t+1)'].isnull()]
df = df[~df['exretn'].isnull()]

#* ************************************** */
#* Single Sort                            */
#* ************************************** */  
n = 5
df['Qmktb']  = df.groupby(by = ['date'])['mktb'].\
    progress_apply(lambda x: pd.qcut(x,n,labels=False,duplicates='drop')+1) 
df['Quncb']  = df.groupby(by = ['date'])['uncb'].\
    progress_apply(lambda x: pd.qcut(x,n,labels=False,duplicates='drop')+1) 

df['value-weights-mktb'] = df.groupby([ 'date','Qmktb' ])\
    ['bond_amount_out'].progress_apply( lambda x: x/np.nansum(x) )
df['value-weights-unc'] = df.groupby([ 'date','Quncb' ])\
    ['bond_amount_out'].progress_apply( lambda x: x/np.nansum(x) )

#* ************************************** */
#* Sample Statistics                      */
#* ************************************** */  
UNC_Beta = df.groupby("date")['uncb'].mean()
UNC_Beta.mean()
UNC_Beta.plot()

MKTB_Beta = df.groupby("date")['mktb'].mean()
MKTB_Beta.mean()
MKTB_Beta.plot()

#* ************************************** */
#* Average Returns & t-Stats              */
#* ************************************** */  

#### Full-Sample ####

ret      = 'exret(t+1)'
weight   = 'value-weights-unc'
Q        = 'Quncb'
sorts1 = df.groupby(['date',Q])[ret,weight]\
    .progress_apply( lambda x: np.nansum( x[ret] * x[weight]) )
sorts1 = sorts1.to_frame()
sorts1 = sorts1.reset_index()
sorts1.columns = ['date','Q1','ret']

sortsAll = sorts1.pivot_table( index = 'date',columns = ['Q1'], values = 'ret').dropna()  
sortsAll['HMLls'] = sortsAll.iloc[:,4] - sortsAll.iloc[:,0]

# The returns at from t:t+1 -> i.e. 1-Month AHEAD   # 
# Lag them to get the correct date index            #
sortsAll.index = sortsAll.index + MonthEnd(1)

# BBW Sample #
sortsAll = sortsAll[sortsAll.index <= "2017-12-31"]

sortsAll.cumsum().plot()
print( sortsAll.mean() * 100 )

# t-Stats #
tstats= list()

for i in range(0,(n+1)):        
    regB = sm.OLS(sortsAll.iloc[:,i].values,
                  pd.DataFrame(np.tile(1,(len(sortsAll.iloc[:,i]),1)) ).values ,
                  ).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    tstats.append(np.round(regB.tvalues[0],2))  

Mean       = (pd.DataFrame(np.array(sortsAll.mean())) * 100).round(3)         
Tstats  = pd.DataFrame(np.array(tstats))

print( Mean )
print( Tstats )

#### Investment Grade (IG) ####
# BSW (2021) find a premium of 0.40% per month
# from July 2002 -> December 2017 #
ret      = 'exret(t+1)'
weight   = 'value-weights-unc'
Q        = 'Quncb'
dfIG     = df[df.rat <= 10]

dfIG['Qmktb']  = dfIG.groupby(by = ['date'])['mktb'].\
    progress_apply(lambda x: pd.qcut(x,n,labels=False,duplicates='drop')+1) 
dfIG['Quncb']  = dfIG.groupby(by = ['date'])['uncb'].\
    progress_apply(lambda x: pd.qcut(x,n,labels=False,duplicates='drop')+1) 

dfIG['value-weights-mktb'] = dfIG.groupby([ 'date','Qmktb' ])\
    ['bond_amount_out'].progress_apply( lambda x: x/np.nansum(x) )
dfIG['value-weights-unc'] = dfIG.groupby([ 'date','Quncb' ])\
    ['bond_amount_out'].progress_apply( lambda x: x/np.nansum(x) )

sorts1 = dfIG.groupby(['date',Q])[ret,weight]\
    .progress_apply( lambda x: np.nansum( x[ret] * x[weight]) )
sorts1 = sorts1.to_frame()
sorts1 = sorts1.reset_index()
sorts1.columns = ['date','Q1','ret']

sortsIG = sorts1.pivot_table( index = 'date',
                              columns = ['Q1'], values = 'ret').dropna()  
sortsIG['HMLls'] = sortsIG.iloc[:,4] - sortsIG.iloc[:,0]

# The returns at from t:t+1 -> i.e. 1-Month AHEAD   # 
# Lag them to get the correct date index            #
sortsIG.index = sortsIG.index + MonthEnd(1)

# BBW Sample #
sortsIG = sortsIG[sortsIG.index <= "2017-12-31"]

sortsIG.cumsum().plot()
print( sortsIG.mean() * 100 )

# t-Stats #
tstats= list()

for i in range(0,(n+1)):        
    regB = sm.OLS(sortsIG.iloc[:,i].values,
                  pd.DataFrame(np.tile(1,(len(sortsIG.iloc[:,i]),1)) ).values ,
                  ).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    tstats.append(np.round(regB.tvalues[0],2))  

Mean       = (pd.DataFrame(np.array(sortsIG.mean())) * 100).round(3)         
Tstats  = pd.DataFrame(np.array(tstats))

print( Mean )
print( Tstats )

#### Non-Investment Grade (NIG) ####
# BSW (2021) find a premium of 0.81% per month
# from July 2002 -> December 2017 #
ret      = 'exret(t+1)'
weight   = 'value-weights-unc'
Q        = 'Quncb'
dfNIG     = df[df.rat > 10]

dfNIG['Qmktb']  = dfNIG.groupby(by = ['date'])['mktb'].\
    progress_apply(lambda x: pd.qcut(x,n,labels=False,duplicates='drop')+1) 
dfNIG['Quncb']  = dfNIG.groupby(by = ['date'])['uncb'].\
    progress_apply(lambda x: pd.qcut(x,n,labels=False,duplicates='drop')+1) 

dfNIG['value-weights-mktb'] = dfNIG.groupby([ 'date','Qmktb' ])\
    ['bond_amount_out'].progress_apply( lambda x: x/np.nansum(x) )
dfNIG['value-weights-unc'] = dfNIG.groupby([ 'date','Quncb' ])\
    ['bond_amount_out'].progress_apply( lambda x: x/np.nansum(x) )

sorts1 = dfNIG.groupby(['date',Q])[ret,weight]\
    .progress_apply( lambda x: np.nansum( x[ret] * x[weight]) )
sorts1 = sorts1.to_frame()
sorts1 = sorts1.reset_index()
sorts1.columns = ['date','Q1','ret']

sortsNIG = sorts1.pivot_table( index = 'date',
                              columns = ['Q1'], values = 'ret').dropna()  
sortsNIG['HMLls'] = sortsNIG.iloc[:,4] - sortsNIG.iloc[:,0]

# The returns at from t:t+1 -> i.e. 1-Month AHEAD   # 
# Lag them to get the correct date index            #
sortsNIG.index = sortsNIG.index + MonthEnd(1)

# BBW Sample #
sortsNIG = sortsNIG[sortsNIG.index <= "2017-12-31"]

sortsNIG.cumsum().plot()
print( sortsNIG.mean() * 100 )

# t-Stats #
tstats= list()

for i in range(0,(n+1)):        
    regB = sm.OLS(sortsNIG.iloc[:,i].values,
                  pd.DataFrame(np.tile(1,(len(sortsNIG.iloc[:,i]),1)) ).values ,
                  ).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    tstats.append(np.round(regB.tvalues[0],2))  

Mean       = (pd.DataFrame(np.array(sortsNIG.mean())) * 100).round(3)         
Tstats  = pd.DataFrame(np.array(tstats))

print( Mean )
print( Tstats )

#* ************************************** */
#* END                                    */
#* ************************************** */  
