# This script runs regressions for the cryptocrash paper

# Loading libraries

library(AER)
library(sandwich)
library(stargazer)
library(ggplot2)
library(lmtest)
library(dplyr)
library(ymd)
library(kableExtra)
library(modelsummary)

# Project directory info

direc = 'F:/cryptobros/'

# Read in the data

dat <- read.csv(paste(direc, 'data.csv', sep = ''))
btc <- read.csv(paste(direc, 'btc.csv', sep = ''))

# Merging dataframes

d <- c()
d2 <- c()

for (i in 1:dim(dat)[1]) {d <- c(d, strsplit(dat$date[i], ' ')[[1]][1])}

for (i in 1:dim(dat)[1]) {
  
  x1 <- strsplit(d[i], '-')
  x2 <- paste(x1[[1]][2], x1[[1]][3], x1[[1]][1], sep = '/') 
  d2 <- c(d2, x2)
  
}

dat$Date <- d2
dat$Date.x <- as.Date(dat$Date, '%m/%d/%Y')
btc$Date.y <- as.Date(btc$Date, '%m/%d/%Y')
dat <- left_join(dat, btc, by = c('Date.x' = 'Date.y'))

# Making an individual level data set using dplyr

cd <- dat %>% group_by(user_id, period, treated) %>% summarise_all(mean)
cd <- as.data.frame(cd)

# Adding a tweet frequency variable

countx <- c()

for (i in 1:dim(cd)[1]) {
  
  tmp <- dat[which(dat$user_id == cd$user_id[i]),]
  tmp <- tmp[which(tmp$period == cd$period[i]),]
  
  countx <- c(countx,dim(tmp)[1])
  
}

cd$Tweets <- countx

# Binary treated and post variables

cd$Treated <- as.numeric(cd$treated == 'treated')
cd$Post <- as.numeric(cd$period == 'POST')
dat$Treated <- as.numeric(dat$treated == 'treated')
dat$Post <- as.numeric(dat$period == 'POST')

# Running tweet level sentiment regressions

mc <- lm(Compound ~ Treated*Post + Change, data = dat)
mp <- lm(Positive ~ Treated*Post + Change, data = dat)
mn <- lm(Negative ~ Treated*Post + Change, data = dat)
mo <- lm(Neutral ~ Treated*Post + Change, data = dat)

mcid <- lm(Compound ~ Treated*Post + Change + factor(user_id), data = dat)
mpid <- lm(Positive ~ Treated*Post + Change + factor(user_id), data = dat)
mnid <- lm(Negative ~ Treated*Post + Change + factor(user_id), data = dat)
moid <- lm(Neutral ~ Treated*Post + Change + factor(user_id), data = dat)

#stargazer(mc,mcid,mp,mpid,mn,mnid,mo,moid, type = 'text', omit = c('user_id'), omit.stat  = c('f', 'ser'))
write.csv(stargazer(mc,mcid,mp,mpid,mn,mnid,mo,moid, omit = c('user_id'), omit.stat  = c('f', 'ser')),
          paste(direc, 'sentiment_tl_tex.txt', sep = ''))
write.csv(stargazer(mc,mcid,mp,mpid,mn,mnid,mo,moid, type = 'text', omit = c('user_id'), omit.stat  = c('f', 'ser')),
          paste(direc, 'sentiment_tl_text.txt', sep = ''))

# Running tweet level emotion regressions

anger <- lm(Anger ~ Treated*Post + Change, data = dat)
disgust <- lm(Disgust ~ Treated*Post + Change, data = dat)
negative <- lm(Negative_E ~ Treated*Post + Change, data = dat)
joy <- lm(Joy ~ Treated*Post + Change, data = dat)
positive <- lm(Positive_E ~ Treated*Post + Change, data = dat)
anticipation <- lm(Anticipation ~ Treated*Post + Change, data = dat)
fear <- lm(Fear ~ Treated*Post + Change, data = dat)
sadness <- lm(Sadness ~ Treated*Post + Change, data = dat)
trust <- lm(Trust ~ Treated*Post + Change, data = dat)
surprise <- lm(Surprise ~ Treated*Post + Change, data = dat)

angerid <- lm(Anger ~ Treated*Post + Change + factor(user_id), data = dat)
write.csv(stargazer(anger,angerid, omit = c('user_id'), omit.stat = c('ser', 'f')), 'F:/cryptobros/anger.txt')

angerid <- 0
disgustid <- lm(Disgust ~ Treated*Post + Change + factor(user_id), data = dat)
write.csv(stargazer(disgust,disgustid, omit = c('user_id'), omit.stat = c('ser', 'f')), 'F:/cryptobros/disgust.txt')

disgustid <- 0
negativeid <- lm(Negative_E ~ Treated*Post + Change + factor(user_id), data = dat)
write.csv(stargazer(negative,negativeid, omit = c('user_id'), omit.stat = c('ser', 'f')), 'F:/cryptobros/negative.txt')

negativeid <- 0
joyid <- lm(Joy ~ Treated*Post + Change + factor(user_id), data = dat)
write.csv(stargazer(joy,joyid, omit = c('user_id'), omit.stat = c('ser', 'f')), 'F:/cryptobros/joy.txt')

joyid <- 0
positiveid <- lm(Positive_E ~ Treated*Post + Change + factor(user_id), data = dat)
write.csv(stargazer(positive,positiveid, omit = c('user_id'), omit.stat = c('ser', 'f')), 'F:/cryptobros/positive.txt')

positiveid <- 0
fearid <- lm(Fear ~ Treated*Post + Change + factor(user_id), data = dat)
write.csv(stargazer(fear,fearid, omit = c('user_id'), omit.stat = c('ser', 'f')), 'F:/cryptobros/fear.txt')

fearid <- 0
sadnessid <- lm(Sadness ~ Treated*Post + Change + factor(user_id), data = dat)
write.csv(stargazer(sadness,sadnessid, omit = c('user_id'), omit.stat = c('ser', 'f')), 'F:/cryptobros/sadness.txt')

sadnessid <- 0
trustid <- lm(Trust ~ Treated*Post + Change + factor(user_id), data = dat)
write.csv(stargazer(trust,trustid, omit = c('user_id'), omit.stat = c('ser', 'f')), 'F:/cryptobros/trust.txt')

trustid <- 0
surpriseid <- lm(Surprise ~ Treated*Post + Change + factor(user_id), data = dat)
write.csv(stargazer(surprise,surpriseid, omit = c('user_id'), omit.stat = c('ser', 'f')), 'F:/cryptobros/surprise.txt')
surpriseid <- 0

# Running individual level tweet volume regressions (includes robust standard errors because of fewer (~4k) observations)

t <- lm(Tweets ~ Treated*Post + Change, data = cd)
tc <- lm(Tweets ~ Treated*Post + Change + Compound, data = cd)
tp <- lm(Tweets ~ Treated*Post + Change + Positive, data = cd)
tn <- lm(Tweets ~ Treated*Post + Change + Negative, data = cd)
to <- lm(Tweets ~ Treated*Post + Change + Neutral, data = cd)
tall <- lm(Tweets ~ Treated*Post + Change + Compound + Positive + Negative + Neutral, data = cd)

tid <- lm(Tweets ~ Treated*Post + Change + factor(user_id), data = cd)
tcid <- lm(Tweets ~ Treated*Post + Change + Compound + factor(user_id), data = cd)
tpid <- lm(Tweets ~ Treated*Post + Change + Positive + factor(user_id), data = cd)
tnid <- lm(Tweets ~ Treated*Post + Change + Negative + factor(user_id), data = cd)
toid <- lm(Tweets ~ Treated*Post + Change + Neutral + factor(user_id), data = cd)
tallid <- lm(Tweets ~ Treated*Post + Change + Compound + Positive + Negative + Neutral + factor(user_id), data = cd)

tidx <- coeftest(tid, vcov. = vcovCL(tid, type = 'HC0'))
tcidx <- coeftest(tcid, vcov. = vcovCL(tcid, type = 'HC0'))
tpidx <- coeftest(tpid, vcov. = vcovCL(tpid, type = 'HC0'))
tnidx <- coeftest(tnid, vcov. = vcovCL(tnid, type = 'HC0'))
toidx <- coeftest(toid, vcov. = vcovCL(toid, type = 'HC0'))
tallidx <- coeftest(tallid, vcov. = vcovCL(tallid, type = 'HC0'))

#stargazer(t,tid,tc,tcid,tp,tpid,tn,tnid,to,toid,tall,tallid, type = 'text', omit = c('user_id'), omit.stat = c('f', 'ser'))
write.csv(stargazer(tidx,tcidx,tpidx,tnidx,toidx,tallidx, omit = c('user_id'), omit.stat = c('f', 'ser')),
          paste(direc, 'tweets_tex.txt', sep = ''))
write.csv(stargazer(tidx,tcidx,tpidx,tnidx,toidx,tallidx, type = 'text', omit = c('user_id'), omit.stat = c('f', 'ser')),
          paste(direc, 'tweets_text.txt', sep = ''))

# Running individual level sentiment regressions (includes robust standard errors because of fewer (~4k) observations)

mcid <- lm(Compound ~ Treated*Post + Change + factor(user_id), data = cd)
mpid <- lm(Positive ~ Treated*Post + Change + factor(user_id), data = cd)
mnid <- lm(Negative ~ Treated*Post + Change + factor(user_id), data = cd)
moid <- lm(Neutral ~ Treated*Post + Change + factor(user_id), data = cd)

mcidx <- coeftest(mcid, vcov. = vcovCL(mcid, type = 'HC0'))
mpidx <- coeftest(mpid, vcov. = vcovCL(mpid, type = 'HC0'))
mnidx <- coeftest(mnid, vcov. = vcovCL(mnid, type = 'HC0'))
moidx <- coeftest(moid, vcov. = vcovCL(moid, type = 'HC0'))

#stargazer(mcidx,mpidx,mnidx,moidx, type = 'text', omit = c('user_id'), omit.stat = c('f', 'ser'))
write.csv(stargazer(mcidx,mpidx,mnidx,moidx, omit = c('user_id'), omit.stat = c('f', 'ser')),
          paste(direc, 'sentiment_il_tex.txt', sep = ''))
write.csv(stargazer(mcidx,mpidx,mnidx,moidx, type = 'text', omit = c('user_id'), omit.stat = c('f', 'ser')),
          paste(direc, 'sentiment_il_text.txt', sep = ''))

# Running individual level emotion regressions (includes robust standard errors because of fewer (~4k) observations)

angerid <- lm(Anger ~ Treated*Post + Change + factor(user_id), data = cd)
disgustid <- lm(Disgust ~ Treated*Post + Change + factor(user_id), data = cd)
negativeid <- lm(Negative_E ~ Treated*Post + Change + factor(user_id), data = cd)
joyid <- lm(Joy ~ Treated*Post + Change + factor(user_id), data = cd)
positiveid <- lm(Positive_E ~ Treated*Post + Change + factor(user_id), data = cd)
anticipationid <- lm(Anticipation ~ Treated*Post + Change + factor(user_id), data = cd)
fearid <- lm(Fear ~ Treated*Post + Change + factor(user_id), data = cd)
sadnessid <- lm(Sadness ~ Treated*Post + Change + factor(user_id), data = cd)
trustid <- lm(Trust ~ Treated*Post + Change + factor(user_id), data = cd)
surpriseid <- lm(Surprise ~ Treated*Post + Change + factor(user_id), data = cd)

anger0 <- lm(Anger ~ Treated*Post + Change, data = cd)
disgust0 <- lm(Disgust ~ Treated*Post + Change, data = cd)
negative0 <- lm(Negative_E ~ Treated*Post + Change, data = cd)
joy0 <- lm(Joy ~ Treated*Post + Change, data = cd)
positive0 <- lm(Positive_E ~ Treated*Post + Change, data = cd)
anticipation0 <- lm(Anticipation ~ Treated*Post + Change, data = cd)
fear0 <- lm(Fear ~ Treated*Post + Change, data = cd)
sadness0 <- lm(Sadness ~ Treated*Post + Change, data = cd)
trust0 <- lm(Trust ~ Treated*Post + Change, data = cd)
surprise0 <- lm(Surprise ~ Treated*Post + Change, data = cd)

angerx <- coeftest(angerid, vcov. = vcovCL(angerid, type = 'HC0'))
disgustx <- coeftest(disgustid, vcov. = vcovCL(disgustid, type = 'HC0'))
negativex <- coeftest(negativeid, vcov. = vcovCL(negativeid, type = 'HC0'))
joyx <- coeftest(joyid, vcov. = vcovCL(joyid, type = 'HC0'))
positivex <- coeftest(positiveid, vcov. = vcovCL(positiveid, type = 'HC0'))
anticipationx <- coeftest(anticipationid, vcov. = vcovCL(anticipationid, type = 'HC0'))
fearx <- coeftest(fearid, vcov. = vcovCL(fearid, type = 'HC0'))
sadnessx <- coeftest(sadnessid, vcov. = vcovCL(sadnessid, type = 'HC0'))
trustx <- coeftest(trustid, vcov. = vcovCL(trustid, type = 'HC0'))
surprisex <- coeftest(surpriseid, vcov. = vcovCL(surpriseid, type = 'HC0'))

angerx0 <- coeftest(anger0, vcov. = vcovCL(anger0, type = 'HC0'))
disgustx0 <- coeftest(disgust0, vcov. = vcovCL(disgust0, type = 'HC0'))
negativex0 <- coeftest(negative0, vcov. = vcovCL(negative0, type = 'HC0'))
joyx0 <- coeftest(joy0, vcov. = vcovCL(joy0, type = 'HC0'))
positivex0 <- coeftest(positive0, vcov. = vcovCL(positive0, type = 'HC0'))
anticipationx0 <- coeftest(anticipation0, vcov. = vcovCL(anticipation0, type = 'HC0'))
fearx0 <- coeftest(fear0, vcov. = vcovCL(fear0, type = 'HC0'))
sadnessx0 <- coeftest(sadness0, vcov. = vcovCL(sadness0, type = 'HC0'))
trustx0 <- coeftest(trust0, vcov. = vcovCL(trust0, type = 'HC0'))
surprisex0 <- coeftest(surprise0, vcov. = vcovCL(surprise0, type = 'HC0'))

#stargazer(angerx0,angerx,disgustx0,disgustx,fearx0,fearx,sadnessx0,sadnessx,negativex0,negativex,surprisex0,surprisex,
#          trustx0,trustx,joyx0,joyx,positivex0,positivex, type = 'text', omit = c('user_id'), omit.stat = c('f', 'ser'))
write.csv(stargazer(angerx0,angerx,disgustx0,disgustx,fearx0,fearx,sadnessx0,sadnessx,negativex0,negativex,surprisex0,surprisex,
                    trustx0,trustx,joyx0,joyx,positivex0,positivex, omit = c('user_id'), omit.stat = c('f', 'ser')),
          paste(direc, 'emotion_il_tex.txt', sep = ''))
write.csv(stargazer(angerx0,angerx,disgustx0,disgustx,fearx0,fearx,sadnessx0,sadnessx,negativex0,negativex,surprisex0,surprisex,
                    trustx0,trustx,joyx0,joyx,positivex0,positivex, type = 'text', omit = c('user_id'), omit.stat = c('f', 'ser')),
          paste(direc, 'emotion_il_text.txt', sep = ''))

# Making a BTC time series figure

btc$Date <- as.Date(btc$Date, format = '%m/%d/%Y')

ggplot(data = btc, aes(x = Date, y = value)) +
  ggtitle('$BTC Time Series') +
  ylab('Price of Bitcoin in USD') +
  geom_line(aes(y = Close , col = '$BTC'), size = 1, alpha = 1) +
  scale_x_date(date_labels = '%Y %m %d', date_breaks = 'months') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(legend.position = 'none', plot.title = element_text(hjust = 0.5))

dev.copy(png, paste(direc, 'btc.png'))
dev.off()

# Cute sum stats table

keepers <- c('Compound', 'Positive', 'Negative', 'Neutral', 'Anger', 'Disgust', 'Negative_E', 'Joy',
             'Positive_E', 'Anticipation', 'Fear', 'Sadness', 'Trust', 'Surprise', 'Close', 'Change')

new_names <- c('Compound', 'Positive (Sentiment)', 'Negative (Sentiment)', 'Neutral', 'Anger',
               'Disgust', 'Negative (Emotion)', 'Joy', 'Positive (Emotion)', 'Anticipation',
               'Fear', 'Sadness', 'Trust', 'Surprise', 'Bitcoin', 'Change in Bitcoin')

sumdata <- dat[,names(dat) %in% keepers]
names(sumdata) <- new_names
datasummary_skim(sumdata, fmt = '%.3f')

