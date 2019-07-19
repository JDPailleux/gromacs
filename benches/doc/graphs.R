library(plotly)
library(ggplot2)

dat1 <- data.frame(
  Simd = factor(c("AVX2","NSIMD","AVX2","NSIMD")),
  pme_rank = factor(c("0","0","-1","-1"), levels=c("-1","0")),
  perf = c(2.537, 2.607, 2.530, 2.498)
)

dat2 <- data.frame(
  Simd = factor(c("AVX2","NSIMD","AVX2","NSIMD")),
  pme_rank = factor(c("0","0","-1","-1"), levels=c("-1","0")),
  perf = c(2285.657, 2224.498, 2291.430,2339.114)
)

dat3 <- data.frame(
  Simd = factor(c("AVX","NSIMD","AVX","NSIMD")),
  pme_rank = factor(c("0","0","-1","-1"), levels=c("-1","0")),
  perf = c(2.472, 2.496, 2.475, 2.516)
)
2.472
dat4 <- data.frame(
  Simd = factor(c("AVX","NSIMD","AVX","NSIMD")),
  pme_rank = factor(c("0","0","-1","-1"), levels=c("-1","0")),
  perf = c(2345.111, 2323.547 , 2342.610, 2304.658)
)


# Bar graph
p1 <- ggplot(data=dat1, aes(x=pme_rank, y=perf, fill=Simd)) +
  geom_bar(stat="identity", position=position_dodge()) + xlab("PME Rank") + ylab("Performance (ns/day)")
p1

p2 <- ggplot(data=dat2, aes(x=pme_rank, y=perf, fill=Simd)) +
  geom_bar(stat="identity", position=position_dodge()) + xlab("PME Rank") + ylab("Performance (Gcycles)")
p2
  