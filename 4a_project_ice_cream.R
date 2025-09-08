library(tseries)
library(TSA)
library(astsa)

# Load data
icrm <- read.csv("/Users/ashleychen/Documents/year_4/stats 4a03/4a_project/ice_cream.csv")
str(icrm) # 577 obs.

# Convert DATE column to Date type
icrm$DATE <- as.Date(icrm$DATE)

# Create time series object (monthly from Jan 1972 to Jan 2020)
ts_icrm <- ts(icrm$IPN31152N, 
              start = c(1972, 1), 
              end = c(2020, 1), 
              frequency = 12)

str(ts_icrm)

# Plot raw data
plot(icrm,
     ylab = 'Monthly Ice Cream Sales',
     xlab = 'Year',
     main = 'Monthly Ice Cream Sales (Jan 1972 – Jan 2020)')

# Original time series plot
plot(ts_icrm,
     ylab = 'Ice Cream Sales',
     main = 'Time Series Plot of Monthly Ice Cream Sales')

# ACF of raw data
acf(ts_icrm,
    main = 'Sample ACF of Raw Ice Cream Sales', lag.max = 36)

# Stationarity check
icrm_adf <- adf.test(ts_icrm)  # p-value = 0.6307 > 0.05 -> not stationary
icrm_adf

# First difference
icrm_diff <- diff(ts_icrm)

plot(icrm_diff,
     ylab = 'First Difference of Ice Cream Sales',
     main = 'Time Series Plot of the First Differences of Ice Cream Sales') #Trend Removed

adf.test(icrm_diff)  # p-value smaller than printed p-value -> stationary

# ACF and PACF after first difference
par(mfrow = c(2, 1))
acf(c(icrm_diff), lag.max = 48,
    main = 'Sample ACF of First Differences of Ice Cream Sales')
pacf(c(icrm_diff), lag.max = 48,
     main = 'Sample PACF of First Differences of Ice Cream Sales')
par(mfrow = c(1, 1))

# Seasonal differencing
seasonaldiff <- diff(icrm_diff, lag = 12)

plot(seasonaldiff,
     ylab = 'First and Seasonal Differences of Ice Cream Sales',
     main = 'Time Series Plot of First and Seasonal Differences of Ice Cream Sales')

# ACF and PACF after seasonal differencing
par(mfrow = c(2, 1))
acf(c(seasonaldiff), lag.max = 48,
    main = 'Sample ACF After First and Seasonal Differencing')
pacf(c(seasonaldiff), lag.max = 48,
     main = 'Sample PACF After First and Seasonal Differencing')
par(mfrow = c(1, 1))

# Final SARIMA model STEP1
sarima_mod1 <- arima(ts_icrm, order = c(1,1,1),
                      seasonal = list(order = c(0,1,1), period = 12))
# Diagnostics
tsdiag(sarima_mod1, gof = 40)
# ACF and PACF of residuals
par(mfrow = c(2, 1))
acf(residuals(sarima_mod1),
    main = 'ACF of Residuals (ARIMA(1,1,1)x(0,1,1)_12)')
pacf(residuals(sarima_mod1),
     main = 'PACF of Residuals (ARIMA(1,1,1)x(0,1,1)_12)')
par(mfrow = c(1, 1))
# Normality check
shapiro.test(residuals(sarima_mod1)) #p-value = 0.03697 < 0.05
qqnorm(residuals(sarima_mod1),
       main = 'QQ Plot of Residuals (ARIMA(1,1,1)x(0,1,1)_12)')
qqline(residuals(sarima_mod1))

# Final SARIMA model STEP2
sarima_mod2 <- arima(ts_icrm, order = c(1,1,2),
                     seasonal = list(order = c(0,1,1), period = 12))
# Diagnostics
tsdiag(sarima_mod2, gof = 40)
# ACF and PACF of residuals
par(mfrow = c(2, 1))
acf(residuals(sarima_mod2),
    main = 'ACF of Residuals (ARIMA(1,1,2)x(0,1,1)_12)')
pacf(residuals(sarima_mod2),
     main = 'PACF of Residuals (ARIMA(1,1,2)x(0,1,1)_12)')
par(mfrow = c(1, 1))
# Normality check
shapiro.test(residuals(sarima_mod2)) #p-value = 0.02972 < 0.05
qqnorm(residuals(sarima_mod2),
       main = 'QQ Plot of Residuals (ARIMA(1,1,2)x(0,1,1)_12)')
qqline(residuals(sarima_mod2))

# Final SARIMA model STEP3
sarima_mod3 <- arima(ts_icrm, order = c(2,1,2),
                     seasonal = list(order = c(0,1,1), period = 12))
# Diagnostics
tsdiag(sarima_mod3, gof = 40)
# ACF and PACF of residuals
par(mfrow = c(2, 1))
acf(residuals(sarima_mod3),
    main = 'ACF of Residuals (ARIMA(2,1,2)x(0,1,1)_12)')
pacf(residuals(sarima_mod3),
     main = 'PACF of Residuals (ARIMA(2,1,2)x(0,1,1)_12)')
par(mfrow = c(1, 1))
# Normality check
shapiro.test(residuals(sarima_mod3)) #p-value = 0.0248 <0.05
qqnorm(residuals(sarima_mod3),
       main = 'QQ Plot of Residuals (ARIMA(2,1,2)x(0,1,1)_12)')
qqline(residuals(sarima_mod3))

# Final SARIMA model STEP4
sarima_mod4 <- arima(ts_icrm, order = c(2,1,3),
                     seasonal = list(order = c(0,1,1), period = 12))
# Diagnostics
tsdiag(sarima_mod4, gof = 40)
# ACF and PACF of residuals
par(mfrow = c(2, 1))
acf(residuals(sarima_mod4),
    main = 'ACF of Residuals (ARIMA(2,1,3)x(0,1,1)_12)')
pacf(residuals(sarima_mod4),
     main = 'PACF of Residuals (ARIMA(2,1,3)x(0,1,1)_12)')
par(mfrow = c(1, 1))
# Normality check
shapiro.test(residuals(sarima_mod4)) #p-value = 0.03085 < 0.05
qqnorm(residuals(sarima_mod4),
       main = 'QQ Plot of Residuals (ARIMA(2,1,3)x(0,1,1)_12)')
qqline(residuals(sarima_mod4))

# Overfitting
mod_a <- arima(ts_icrm, order = c(2,1,3), seasonal = list(order = c(0,1,1), period = 12)) #sarima_mod4
mod_b <- arima(ts_icrm, order = c(2,1,4), seasonal = list(order = c(0,1,1), period = 12))

mod_a
mod_b #higher aic -> choose mod_a
mod_final <- mod_a  # ARIMA(2,1,3)x(0,1,1)_12

# sarima for forecasting and diagnostics
icrmmod <- sarima(ts_icrm, 2, 1, 3, 0, 1, 1, 12)

# Forecast 24 months ahead
sarima.for(ts_icrm, n.ahead = 24, 2, 1, 3, 0, 1, 1, 12,
           ylab = 'Monthly Ice Cream Sales',
           xlab = 'Year',
           main = 'Forecasts and Forecast Limits for Monthly Ice Cream Sales')
