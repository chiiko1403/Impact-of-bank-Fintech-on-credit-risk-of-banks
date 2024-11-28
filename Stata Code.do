clear
cls
import excel "D:\Chiiko\Học tập\Đại học\Tài chính\Năm 4\KLTN\Data\Final\Data.xlsx", sheet("Sheet1") firstrow
encode Code, gen(ID)
xtset ID Year

xtunitroot fisher NPL, pperron lags(5)
xtunitroot fisher FIN, pperron lags(5)
xtunitroot fisher CAR, pperron lags(5)
xtunitroot fisher LLP, pperron lags(5)
xtunitroot fisher CIR, pperron lags(5)
xtunitroot fisher LDR, pperron lags(5)
xtunitroot fisher d.LDR, pperron lags(5)
xtunitroot fisher ROA, pperron lags(5)
xtunitroot fisher GDP, pperron lags(5)
xtunitroot fisher CPI, pperron lags(5)

gen dLDR = d.LDR

sum NPL FIN CAR LLP CIR LDR ROA OWN SIZE GDP CPI
pwcorr NPL FIN CAR LLP CIR dLDR ROA OWN SIZE GDP CPI, sig

reg NPL FIN CAR LLP CIR dLDR ROA OWN SIZE GDP CPI
vif

*Run the first model
xtreg NPL FIN CAR LLP CIR dLDR ROA GDP CPI, fe
est sto fem
xttest3

xtreg NPL FIN CAR LLP CIR dLDR ROA GDP CPI, re
est sto rem
xttest0

xtserial NPL FIN CAR LLP CIR dLDR ROA GDP CPI

xtreg NPL FIN CAR LLP CIR dLDR ROA GDP CPI, fe robust
est sto first
xtreg NPL FIN CAR LLP CIR dLDR ROA GDP CPI, re robust
xtoverid

ivregress 2sls NPL CAR LLP CIR dLDR ROA GDP CPI (FIN = l.FIN)
estat endog
ivregress 2sls NPL FIN LLP CIR dLDR ROA GDP CPI (CAR = l.CAR)
estat endog
ivregress 2sls NPL FIN CAR CIR dLDR ROA GDP CPI (LLP = l.LLP)
estat endog
ivregress 2sls NPL FIN CAR LLP dLDR ROA GDP CPI (CIR = l.CIR)
estat endog //
ivregress 2sls NPL FIN CAR LLP CIR ROA GDP CPI (dLDR = l.dLDR)
estat endog
ivregress 2sls NPL FIN CAR LLP CIR dLDR GDP CPI (ROA = l.ROA)
estat endog
ivregress 2sls NPL FIN CAR LLP CIR dLDR ROA CPI (GDP = l.GDP)
estat endog
ivregress 2sls NPL FIN CAR LLP CIR dLDR ROA GDP (CPI = l.CPI)
estat endog

xtabond2 NPL FIN CAR LLP CIR dLDR ROA GDP CPI, gmm(l.CIR l.LLP, collapse lag(1 2)) iv(NPL FIN CAR dLDR ROA GDP CPI, equation(diff)) twostep
est sto fourth

*Run the second model
gen OWNxFIN = OWN * FIN
xtreg NPL FIN OWN OWNxFIN CAR LLP CIR dLDR ROA GDP CPI, fe
xttest3

xtreg NPL FIN OWN OWNxFIN CAR LLP CIR dLDR ROA GDP CPI, re
xttest0

xtserial NPL FIN OWN OWNxFIN CAR LLP CIR dLDR ROA GDP CPI

xtreg NPL FIN OWN OWNxFIN CAR LLP CIR dLDR ROA GDP CPI, fe robust
est sto second
xtreg NPL FIN OWN OWNxFIN CAR LLP CIR dLDR ROA GDP CPI, re robust
xtoverid 

*Run the third model
gen SIZExFIN = SIZE * FIN
xtreg NPL FIN SIZE SIZExFIN CAR LLP CIR dLDR ROA GDP CPI, fe
xttest3

xtreg NPL FIN SIZE SIZExFIN CAR LLP CIR dLDR ROA GDP CPI, re
xttest0

xtserial NPL FIN SIZE SIZExFIN CAR LLP CIR dLDR ROA GDP CPI

xtreg NPL FIN SIZE SIZExFIN CAR LLP CIR dLDR ROA GDP CPI, fe robust
est sto third
xtreg NPL FIN SIZE SIZExFIN CAR LLP CIR dLDR ROA GDP CPI, re robust
xtoverid

esttab first second third fourth, r2 star(* 0.10 ** 0.05 *** 0.01)